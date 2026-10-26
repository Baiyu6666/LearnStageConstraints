from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.stats import t as student_t

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.registry import load_env


MODE_ORDER = ("inactive", "eq", "lb", "ub")
MODE_LABELS = {
    "inactive": "Inactive",
    "eq": "Equality",
    "lb": "Lower bound",
    "ub": "Upper bound",
}
MODE_COLORS = {
    "inactive": "#34495E",
    "eq": "#8E5AA9",
    "lb": "#C0392B",
    "ub": "#1F7A5B",
}


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _feature_index(schema: list[dict], name: str) -> int:
    for index, spec in enumerate(schema):
        if str(spec.get("name", "")) == str(name):
            return int(spec.get("column_idx", index))
    raise ValueError(f"Feature '{name}' not found in schema.")


def _gaussian_pdf(values: np.ndarray, *, location: float, scale: float) -> np.ndarray:
    return norm.pdf(np.asarray(values, dtype=float), loc=float(location), scale=max(float(scale), 1e-12))


def _soft_half_t_pdf(
    values: np.ndarray,
    *,
    eta: float,
    scale: float,
    nu: float,
    alpha: float,
    mode: str,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    scale = max(float(scale), 1e-12)
    alpha = max(float(alpha), 1e-12)
    sign = 1.0 if str(mode).lower() == "lb" else -1.0
    slack = sign * (values - float(eta))
    half_t_at_zero = 2.0 * float(student_t.pdf(0.0, df=nu)) / scale
    normalizer = 1.0 + 2.0 * alpha * float(student_t.pdf(0.0, df=nu))
    density = np.empty_like(slack, dtype=float)
    feasible = slack >= 0.0
    density[feasible] = (2.0 / scale) * student_t.pdf(slack[feasible] / scale, df=nu) / normalizer
    density[~feasible] = half_t_at_zero * np.exp(slack[~feasible] / (alpha * scale)) / normalizer
    return density


def _mean_nll(density: np.ndarray) -> float:
    return float(-np.mean(np.log(np.maximum(np.asarray(density, dtype=float), 1e-300))))


def _fit_inactive(values: np.ndarray, *, scale_floor: float) -> dict:
    q10, location, q90 = np.quantile(values, [0.10, 0.50, 0.90])
    unit_width = float(norm.ppf(0.90) - norm.ppf(0.10))
    scale = max(float(q90 - q10) / unit_width, float(scale_floor), 1e-12)
    return {
        "mode": "inactive",
        "location": float(location),
        "scale": float(scale),
        "mean_nll": _mean_nll(_gaussian_pdf(values, location=float(location), scale=scale)),
    }


def _fit_equality(values: np.ndarray, *, scale_floor: float, scale_upper: float) -> dict:
    eta = float(np.median(values))
    rms = float(np.sqrt(np.mean(np.square(values - float(eta)))))
    scale = float(np.clip(rms, float(scale_floor), float(scale_upper)))
    return {
        "mode": "eq",
        "eta": float(eta),
        "scale": float(scale),
        "mean_nll": _mean_nll(_gaussian_pdf(values, location=float(eta), scale=scale)),
    }


def _fit_inequality(
    values: np.ndarray,
    *,
    nu: float,
    alpha: float,
    scale_floor: float,
    mode: str,
) -> dict:
    lower_scale = max(float(scale_floor), 1e-6)
    value_span = max(float(np.ptp(values)), float(np.std(values)), lower_scale)
    upper_scale = max(10.0 * value_span, 10.0 * lower_scale)
    eta_bounds = (
        float(np.min(values) - 0.25 * value_span),
        float(np.max(values) + 0.25 * value_span),
    )

    def objective(parameters: np.ndarray) -> float:
        candidate_eta = float(parameters[0])
        candidate_scale = math.exp(float(parameters[1]))
        candidate_density = _soft_half_t_pdf(
            values,
            eta=candidate_eta,
            scale=candidate_scale,
            nu=float(nu),
            alpha=float(alpha),
            mode=str(mode),
        )
        return _mean_nll(candidate_density)

    start_quantiles = (0.01, 0.05, 0.10, 0.25, 0.50) if str(mode).lower() == "lb" else (0.50, 0.75, 0.90, 0.95, 0.99)
    best_result = None
    for quantile in start_quantiles:
        initial_eta = float(np.quantile(values, quantile))
        sign = 1.0 if str(mode).lower() == "lb" else -1.0
        feasible_slack = np.maximum(sign * (values - initial_eta), 0.0)
        initial_scale = float(np.quantile(feasible_slack, 0.90)) / float(student_t.ppf(0.95, df=float(nu)))
        initial_scale = float(np.clip(initial_scale, lower_scale, upper_scale))
        result = minimize(
            objective,
            np.asarray([initial_eta, math.log(initial_scale)], dtype=float),
            method="L-BFGS-B",
            bounds=[eta_bounds, (math.log(lower_scale), math.log(upper_scale))],
            options={"maxiter": 500, "ftol": 1e-12},
        )
        if best_result is None or float(result.fun) < float(best_result.fun):
            best_result = result

    eta = float(best_result.x[0])
    scale = float(math.exp(float(best_result.x[1])))
    density = _soft_half_t_pdf(
        values,
        eta=float(eta),
        scale=scale,
        nu=float(nu),
        alpha=float(alpha),
        mode=str(mode),
    )
    return {
        "mode": str(mode),
        "eta": float(eta),
        "scale": float(scale),
        "mean_nll": _mean_nll(density),
        "parameter_estimation": "joint_nll",
        "optimization_success": bool(best_result.success),
    }


def _load_example(
    *,
    env_config: Path,
    result_dir: Path,
    feature_name: str,
    stage_index: int,
) -> dict:
    env_kwargs = _load_json(env_config)
    env_name = str(env_kwargs.pop("name"))
    env_kwargs.pop("method_overrides", None)
    bundle = load_env(env_name, **env_kwargs)
    schema = list(bundle.feature_schema or bundle.env.get_feature_schema())
    raw_feature_index = _feature_index(schema, feature_name)
    raw_features = [np.asarray(bundle.env.compute_all_features_matrix(demo), dtype=float) for demo in bundle.demos]
    full_stack = np.concatenate(raw_features, axis=0)
    feature_mean = np.mean(full_stack, axis=0)
    feature_std = np.std(full_stack, axis=0) + 1e-8

    segmentation = _load_json(result_dir / "segmentation.json")
    predicted_cutpoints = segmentation["predicted_cutpoints"]
    num_stages = len(predicted_cutpoints[0]) + 1
    if not 0 <= int(stage_index) < num_stages:
        raise ValueError(f"stage_index={stage_index} is outside the available stages.")

    values_by_demo = []
    intervals = []
    for demo_features, demo_cutpoints_raw in zip(raw_features, predicted_cutpoints):
        demo_cutpoints = [int(value) for value in demo_cutpoints_raw]
        start = 0 if int(stage_index) == 0 else demo_cutpoints[int(stage_index) - 1] + 1
        end = demo_cutpoints[int(stage_index)] if int(stage_index) < len(demo_cutpoints) else len(demo_features) - 1
        raw_values = demo_features[start : end + 1, raw_feature_index]
        standardized_values = (raw_values - feature_mean[raw_feature_index]) / feature_std[raw_feature_index]
        values_by_demo.append(np.asarray(standardized_values, dtype=float))
        intervals.append([int(start), int(end)])

    snapshot = _load_json(result_dir / "config_snapshot.json")
    method_kwargs = dict(snapshot.get("method_kwargs", {}))
    selected_features = list(method_kwargs.get("selected_raw_feature_ids") or [spec.get("name") for spec in schema])
    if feature_name not in selected_features:
        raise ValueError(f"Feature '{feature_name}' was not used by the saved method run.")

    return {
        "values": np.concatenate(values_by_demo, axis=0),
        "intervals": intervals,
        "method_kwargs": method_kwargs,
        "dataset_name": str(snapshot.get("dataset_name", env_name)),
        "method_name": str(snapshot.get("method_name", result_dir.parent.parent.name)),
    }


def plot_soft_half_t_example(
    *,
    env_config: Path,
    result_dir: Path,
    output_dir: Path,
    feature_name: str,
    stage_index: int,
    formats: list[str],
) -> list[Path]:
    example = _load_example(
        env_config=env_config,
        result_dir=result_dir,
        feature_name=feature_name,
        stage_index=int(stage_index),
    )
    values = np.asarray(example["values"], dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 3:
        raise ValueError("Need at least three finite observations in the recovered interval.")

    method_kwargs = dict(example["method_kwargs"])
    alpha = float(method_kwargs.get("truncated_z_soft_boundary_scale", 0.1))
    nu = float(method_kwargs.get("map_nu_ineq", 3.0))
    active_scale_floor = float(method_kwargs.get("map_active_sigma_floor", 0.003))
    configured_equality_scale_upper = float(method_kwargs.get("map_eq_sigma", 0.05))
    equality_scale_upper = 0.15
    inactive_scale_floor = float(method_kwargs.get("map_c_bg", 1.0)) * configured_equality_scale_upper
    inequality_scale_floor = max(
        active_scale_floor,
        float(method_kwargs.get("map_c_ineq", 0.0)) * configured_equality_scale_upper,
    )

    fits = {
        "inactive": _fit_inactive(values, scale_floor=inactive_scale_floor),
        "eq": _fit_equality(
            values,
            scale_floor=min(active_scale_floor, equality_scale_upper),
            scale_upper=equality_scale_upper,
        ),
        "lb": _fit_inequality(
            values,
            nu=nu,
            alpha=alpha,
            scale_floor=inequality_scale_floor,
            mode="lb",
        ),
        "ub": _fit_inequality(
            values,
            nu=nu,
            alpha=alpha,
            scale_floor=inequality_scale_floor,
            mode="ub",
        ),
    }
    selected_mode = min(MODE_ORDER, key=lambda mode: float(fits[mode]["mean_nll"]))

    active_etas = [float(fits[mode]["eta"]) for mode in ("eq", "lb", "ub")]
    x_min = min(float(np.min(values)), *active_etas)
    x_max = max(float(np.max(values)), *active_etas)
    padding = max(0.10 * (x_max - x_min), 0.08)
    grid = np.linspace(x_min - padding, x_max + padding, 600)
    bins = min(12, max(7, int(math.sqrt(values.size)) + 2))
    bin_edges = np.linspace(x_min - 0.02 * (x_max - x_min), x_max + 0.02 * (x_max - x_min), bins + 1)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axis = plt.subplots(figsize=(3.45, 1.58), constrained_layout=False)
    axis.hist(
        values,
        bins=bin_edges,
        density=True,
        color="#D9E5F2",
        edgecolor="#557A9B",
        linewidth=0.45,
        alpha=0.88,
        label="Data",
    )
    line_styles = {"inactive": ":", "eq": "--", "lb": "-", "ub": "-."}
    for mode in MODE_ORDER:
        fit = fits[mode]
        if mode == "inactive":
            density = _gaussian_pdf(grid, location=float(fit["location"]), scale=float(fit["scale"]))
        elif mode == "eq":
            density = _gaussian_pdf(grid, location=float(fit["eta"]), scale=float(fit["scale"]))
        else:
            density = _soft_half_t_pdf(
                grid,
                eta=float(fit["eta"]),
                scale=float(fit["scale"]),
                nu=nu,
                alpha=alpha,
                mode=mode,
            )
        label = f"{MODE_LABELS[mode]}: {float(fit['mean_nll']):.2f}"
        axis.plot(
            grid,
            density,
            color=MODE_COLORS[mode],
            linestyle=line_styles[mode],
            linewidth=1.85 if mode == selected_mode else 1.35,
            label=label,
            zorder=4 if mode == selected_mode else 3,
        )
    axis.set_xlabel("Standardized feature value", labelpad=1.5)
    axis.set_ylabel("Density", labelpad=2.0)
    axis.set_xlim(grid[0], grid[-1])
    axis.set_ylim(bottom=0.0)
    axis.grid(axis="y", color="#D0D5DD", linewidth=0.42, alpha=0.60)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(0.6)
    axis.spines["bottom"].set_linewidth(0.6)
    axis.tick_params(width=0.55, length=2.1, pad=1.3)
    legend = axis.legend(
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.86,
        ncol=1,
        handlelength=1.65,
        handletextpad=0.4,
        borderpad=0.25,
        labelspacing=0.16,
        title="Mean negative log-likelihood $\downarrow$",
        title_fontsize=6.2,
    )
    for mode, legend_text in zip(MODE_ORDER, legend.get_texts()[1:]):
        if mode == selected_mode:
            legend_text.set_fontweight("bold")
            legend_text.set_color("#9A4F0A")
    figure.subplots_adjust(left=0.13, right=0.985, bottom=0.25, top=0.97)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "soft_half_t_example_distribution"
    saved_paths = []
    for file_format in formats:
        output_path = output_dir / f"{stem}.{file_format}"
        figure.savefig(
            output_path,
            dpi=600 if file_format.lower() == "png" else None,
            bbox_inches="tight",
            pad_inches=0.02,
        )
        saved_paths.append(output_path)
    plt.close(figure)

    metadata = {
        "dataset_name": example["dataset_name"],
        "method_name": example["method_name"],
        "feature_name": feature_name,
        "stage_index": int(stage_index),
        "intervals_by_demo": example["intervals"],
        "n_observations": int(values.size),
        "alpha": alpha,
        "nu": nu,
        "equality_scale_upper": equality_scale_upper,
        "selected_mode_by_mean_nll": selected_mode,
        "fits": fits,
    }
    (output_dir / f"{stem}_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot four likelihood-mode fits to an aggregated stage distribution.")
    parser.add_argument("--env-config", type=Path, default=PROJECT_ROOT / "configs/envs/S3ObsAvoid.json")
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/map_balanced_pooled/S3ObsAvoid/method_seed_000",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/paper_figures/soft_half_t")
    parser.add_argument("--feature-name", default="obs_dist")
    parser.add_argument("--stage-index", type=int, default=0)
    parser.add_argument("--formats", default="png,pdf")
    arguments = parser.parse_args()

    formats = [item.strip().lower() for item in str(arguments.formats).split(",") if item.strip()]
    saved_paths = plot_soft_half_t_example(
        env_config=arguments.env_config,
        result_dir=arguments.result_dir,
        output_dir=arguments.output_dir,
        feature_name=str(arguments.feature_name),
        stage_index=int(arguments.stage_index),
        formats=formats,
    )
    for saved_path in saved_paths:
        print(saved_path)


if __name__ == "__main__":
    main()
