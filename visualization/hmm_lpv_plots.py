"""HMM-LPV diagnostics in original position coordinates; no rollout/planning."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
import numpy as np
from scipy.special import logsumexp

from visualization.io import save_figure
from visualization.learned_constraints_matrix import _prepare_constraints_matrix


STAGE_COLORS = ("#D55E00", "#0072B2", "#009E73", "#CC79A7", "#E69F00")
COMPONENT_COLORS = ("#2563EB", "#D97706", "#9333EA", "#059669", "#DB2777", "#0891B2")


def stage_velocity(parameters, stage_index, positions):
    """Evaluate the learned LPV field, using position-only GMM responsibilities.

    Both queries and returned velocities are in the original data coordinates.
    Velocity noise belongs to training emissions, not the execution gate h(x).
    """
    stage = parameters["stages"][stage_index]
    scale = float(parameters["normalization_scale"])
    x = (np.asarray(positions, dtype=float) - np.asarray(parameters["normalization_mean"])) / scale
    means = np.asarray(stage["position_means"])
    covariances = np.asarray(stage["position_covariances"])
    log_weights = []
    for prior, mean, covariance in zip(stage["priors"], means, covariances):
        chol = np.linalg.cholesky(covariance)
        white = np.linalg.solve(chol, (x - mean).T)
        log_weights.append(np.log(prior) - np.log(np.diag(chol)).sum()
                           - 0.5 * np.sum(white ** 2, axis=0))
    log_weights = np.stack(log_weights, axis=1)
    weights = np.exp(log_weights - logsumexp(log_weights, axis=1, keepdims=True))
    local = np.einsum("cij,nj->nci", np.asarray(stage["A"]), x - np.asarray(stage["attractor"]))
    return scale * np.einsum("nc,nci->ni", weights, local)


def _raw_stage(parameters, stage_index):
    stage = parameters["stages"][stage_index]
    offset = np.asarray(parameters["normalization_mean"])
    scale = float(parameters["normalization_scale"])
    return (offset + scale * np.asarray(stage["attractor"]),
            offset + scale * np.asarray(stage["position_means"]),
            scale ** 2 * np.asarray(stage["position_covariances"]))


def _labels_from_ends(demos, stage_ends, count):
    if len(demos) != len(stage_ends):
        raise ValueError("Plot demos and predicted stage ends must match.")
    labels = []
    for demo, ends in zip(demos, stage_ends):
        ends = np.asarray(ends, dtype=int)
        if len(ends) != count or ends[-1] != len(demo) - 1 or np.any(np.diff(np.r_[-1, ends]) <= 0):
            raise ValueError("Plot requires complete, ordered, nonempty stages.")
        labels.append(np.searchsorted(ends, np.arange(len(demo)), side="left"))
    return labels


def _limits(points):
    lo, hi = np.min(points, axis=0), np.max(points, axis=0)
    span = hi - lo
    padding = max(float(span.max()) * 0.10, 0.02)
    return lo - padding, hi + padding


def _set_axes(ax, bounds, dimension):
    lo, hi = bounds
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_xlabel("x", labelpad=2)
    ax.set_ylabel("y", labelpad=2)
    if dimension == 3:
        ax.set_zlim(lo[2], hi[2])
        ax.set_zlabel("z", labelpad=2)
        ax.set_box_aspect(hi - lo)
        ax.view_init(elev=25, azim=-65)
    else:
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.13)
    ax.tick_params(labelsize=8)


def _scatter(ax, points, **kwargs):
    return ax.scatter(*np.asarray(points).T, **kwargs)


def _draw_stage_samples(ax, demos, labels, stage_index):
    color = STAGE_COLORS[stage_index % len(STAGE_COLORS)]
    # A context trace helps relate the stage to the whole task without forcing
    # every local panel to use the full trajectory's spatial range.
    for x, z in zip(demos, labels):
        ax.plot(*x.T, color="#CBD5E1", lw=0.65, alpha=0.5, zorder=1)
        _scatter(ax, x[z == stage_index], color=color, s=7, alpha=0.7, zorder=3, linewidths=0)


def _draw_attractor(ax, goal):
    _scatter(ax, goal[None], marker="*", color="#111827", edgecolors="white", linewidths=0.6,
             s=145, zorder=6, label="LPV attractor")


def _draw_flow(ax, parameters, stage_index, demos, labels, bounds):
    dimension = int(parameters["dimension"])
    lo, hi = bounds
    _draw_stage_samples(ax, demos, labels, stage_index)
    if dimension == 2:
        xs, ys = np.linspace(lo[0], hi[0], 55), np.linspace(lo[1], hi[1], 55)
        xx, yy = np.meshgrid(xs, ys)
        velocity = stage_velocity(parameters, stage_index, np.column_stack([xx.ravel(), yy.ravel()]))
        ax.streamplot(xs, ys, velocity[:, 0].reshape(xx.shape), velocity[:, 1].reshape(xx.shape),
                      color="#73869B", density=1.05, linewidth=0.7, arrowsize=0.8, zorder=2)
    else:
        grid = np.meshgrid(*[np.linspace(lo[i], hi[i], 6) for i in range(3)], indexing="ij")
        points = np.column_stack([a.ravel() for a in grid])
        velocity = stage_velocity(parameters, stage_index, points)
        ax.quiver(*points.T, *velocity.T, length=0.075 * float((hi - lo).max()), normalize=True,
                  color="#73869B", linewidth=0.6, alpha=0.55)
    _draw_attractor(ax, _raw_stage(parameters, stage_index)[0])
    _set_axes(ax, bounds, dimension)
    kind = "DS flow" if dimension == 2 else "DS directions (unit arrows)"
    ax.set_title(f"Stage {stage_index + 1} · {kind}", fontweight="bold", fontsize=11)
    ax.legend(loc="upper right" if dimension == 2 else "upper left", fontsize=7, framealpha=0.9)


def _draw_gaussians(ax, parameters, stage_index, demos, labels, bounds):
    dimension = int(parameters["dimension"])
    goal, means, covariances = _raw_stage(parameters, stage_index)
    _draw_stage_samples(ax, demos, labels, stage_index)
    handles = []
    for c, (mean, covariance, prior) in enumerate(zip(means, covariances, parameters["stages"][stage_index]["priors"])):
        color = COMPONENT_COLORS[c % len(COMPONENT_COLORS)]
        eig, axes = np.linalg.eigh(covariance)
        if dimension == 2:
            principal = axes[:, -1]
            angle = np.degrees(np.arctan2(principal[1], principal[0]))
            # Mahalanobis radius 2, not a "95%" confidence contour in all D.
            ellipse = Ellipse(mean, 4 * np.sqrt(eig[-1]), 4 * np.sqrt(eig[0]), angle=angle,
                              facecolor=color, edgecolor=color, alpha=0.18, linewidth=1.5, zorder=2)
            ax.add_patch(ellipse)
            ax.add_patch(Ellipse(mean, ellipse.width, ellipse.height, angle=angle,
                                 fill=False, edgecolor=color, linewidth=1.2, zorder=3))
        else:
            u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 32), np.linspace(0, np.pi, 18))
            sphere = np.stack([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)], axis=-1)
            ellipsoid = mean + (sphere * (2 * np.sqrt(eig))) @ axes.T
            ax.plot_surface(*np.moveaxis(ellipsoid, -1, 0), color=color, alpha=0.12, linewidth=0)
            ax.plot_wireframe(*np.moveaxis(ellipsoid, -1, 0), color=color, alpha=0.4,
                              linewidth=0.45, rstride=4, cstride=5)
        _scatter(ax, mean[None], marker="+", color=color, s=70, linewidths=1.6, zorder=5)
        if dimension == 2:
            ax.annotate(f"C{c + 1}", mean, xytext=(5, 5), textcoords="offset points", color=color, fontsize=8)
        handles.append(Line2D([], [], color=color, lw=2, marker="+", label=f"C{c + 1} · π={prior:.2f}"))
    _draw_attractor(ax, goal)
    _set_axes(ax, bounds, dimension)
    ax.set_title(f"Stage {stage_index + 1} · position GMM (2σ)", fontweight="bold", fontsize=11)
    ax.legend(handles=handles, loc="best" if dimension == 2 else "upper left", fontsize=7, framealpha=0.9)


def _draw_overview(ax, demos, labels, count, env, dimension):
    for s in range(count):
        color = STAGE_COLORS[s % len(STAGE_COLORS)]
        points = np.vstack([x[z == s] for x, z in zip(demos, labels)])
        _scatter(ax, points, color=color, s=7, alpha=0.65, linewidths=0, label=f"Stage {s + 1}")
    if dimension == 2 and env is not None:
        circles = []
        if hasattr(env, "obs_center") and hasattr(env, "obs_radius"):
            circles.append((np.asarray(env.obs_center)[:2], env.obs_radius))
        circles.extend(zip(getattr(env, "stage1_aux_obstacle_centers", []), getattr(env, "stage1_aux_obstacle_radii", [])))
        for center, radius in circles:
            ax.add_patch(plt.Circle(center, radius, fill=False, color="#64748B", lw=1))
    _set_axes(ax, _limits(np.vstack(demos)), dimension)
    ax.set_title("HMM-LPV predicted stages", fontweight="bold", fontsize=11)
    ax.legend(fontsize=7, loc="best" if dimension == 2 else "upper left")


def _draw_boundaries(ax, demos, ends, true_cuts, metrics):
    for i, (x, stage_ends) in enumerate(zip(demos, ends)):
        for j, predicted in enumerate(stage_ends[:-1]):
            color = STAGE_COLORS[j % len(STAGE_COLORS)]
            pred_x = 100 * (predicted + 1) / len(x)
            ax.scatter(pred_x, i, marker="x", color=color, s=28, zorder=3)
            if true_cuts is not None and true_cuts[i] is not None and j < len(true_cuts[i]):
                true_x = 100 * (true_cuts[i][j] + 1) / len(x)
                ax.plot([true_x, pred_x], [i, i], color=color, lw=1)
                ax.scatter(true_x, i, facecolors="white", edgecolors=color, s=25, zorder=2)
    ax.set_xlim(0, 100)
    ax.set_ylim(len(demos) - 0.5, -0.5)
    ax.set_yticks(range(len(demos)))
    ax.set_ylabel("Demo index")
    ax.set_xlabel("Trajectory progress (%)")
    error = metrics.get("MeanAbsCutpointError", np.nan)
    ax.set_title(f"Cutpoints · MAE {error:.2f} samples", fontweight="bold", fontsize=11)
    ax.grid(alpha=0.15)
    ax.legend(handles=[Line2D([], [], marker="o", mfc="white", color="#64748B", ls="", label="True"),
                       Line2D([], [], marker="x", color="#64748B", ls="", label="Predicted")], fontsize=7)


def _draw_constraints(ax, metrics, dataset_name, learned):
    prefix = "ConstraintLearned" if learned else "Constraint"
    matrix, text, stages, features = _prepare_constraints_matrix(
        metrics, value_key=f"{prefix}ValueMatrix" if learned else "ConstraintTargetMatrix",
        semantics_key="ConstraintLearnedSemanticsMatrix" if learned else "ConstraintSemanticsMatrix",
        active_key="ConstraintPredictedActiveMask" if learned else "ConstraintTrueActiveMask",
        dataset_name=dataset_name,
    )
    ax.imshow(matrix, cmap=ListedColormap(["#F1F5F9", "#54A24B", "#E45756", "#4C78A8"]),
              vmin=0, vmax=3, aspect="auto")
    ax.set_xticks(range(len(stages)), stages)
    ax.set_yticks(range(len(features)), features)
    ax.tick_params(labelsize=8)
    for (row, col), code in np.ndenumerate(matrix):
        ax.text(col, row, text[row, col] if code else "—", ha="center", va="center",
                fontsize=8, color="white" if code else "#64748B")
    ax.set_title("MAP balanced pooled constraints" if learned else "True constraints", fontsize=11, fontweight="bold")


def _draw_parameter_errors(ax, metrics):
    values = np.asarray(metrics["ParameterErrorMatrix"], dtype=float).T
    image = ax.imshow(np.ma.masked_invalid(values), cmap="YlOrRd", vmin=0, aspect="auto")
    ax.set_facecolor("#E2E8F0")
    ax.set_xticks(range(values.shape[1]), [f"stage {s + 1}" for s in range(values.shape[1])])
    ax.set_yticks(range(values.shape[0]), metrics["ConstraintFeatureNames"])
    ax.tick_params(labelsize=8)
    for (row, col), value in np.ndenumerate(values):
        ax.text(col, row, f"{value:.3f}" if np.isfinite(value) else "—", ha="center", va="center", fontsize=8)
    ax.figure.colorbar(image, ax=ax, fraction=0.04, pad=0.03)
    ax.set_title("Normalized parameter error", fontsize=11, fontweight="bold")


def plot_hmm_lpv_summary(*, parameters, demos, stage_ends, history, metrics,
                         save_path, true_cutpoints=None, env=None, dataset_name=""):
    """One column per stage for field and Gaussian panels, wrapping after 3.

    Accepts the exported JSON representation as well as parameter_dict(), so
    saved runs can be replotted without reconstructing or fitting either model.
    """
    dimension = int(parameters["dimension"])
    if dimension not in (2, 3):
        raise ValueError("HMM-LPV plotting supports two or three position dimensions.")
    indices = parameters["position_indices"]
    positions = [np.asarray(x, dtype=float)[:, indices] for x in demos]
    count = len(parameters["stages"])
    labels = _labels_from_ends(positions, stage_ends, count)
    stage_rows = (count + 2) // 3
    rows = 2 + stage_rows * 2
    with plt.rc_context({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}):
        fig = plt.figure(figsize=(16.8, 3.6 * rows + 0.6), layout="constrained")
        grid = fig.add_gridspec(rows, 3)
        projection = {"projection": "3d"} if dimension == 3 else {}
        _draw_overview(fig.add_subplot(grid[0, 0], **projection), positions, labels, count, env, dimension)
        ax = fig.add_subplot(grid[0, 1])
        likelihood = [history["initial_loglik"], *history["loglik"]]
        ax.plot(range(len(likelihood)), likelihood, "o-", color="#334155", ms=3, lw=1.5)
        ax.set_title("HMM-LPV training likelihood", fontsize=11, fontweight="bold")
        ax.set_xlabel("EM iteration (0 = initialization)")
        ax.set_ylabel("Log likelihood")
        ax.grid(alpha=0.15)
        _draw_boundaries(fig.add_subplot(grid[0, 2]), positions, stage_ends, true_cutpoints, metrics)
        for s in range(count):
            goal, means, covariances = _raw_stage(parameters, s)
            spread = 2 * np.sqrt(np.diagonal(covariances, axis1=1, axis2=2))
            members = np.vstack([x[z == s] for x, z in zip(positions, labels)])
            bounds = _limits(np.vstack([members, goal[None], means - spread, means + spread]))
            row, col = 1 + 2 * (s // 3), s % 3
            _draw_flow(fig.add_subplot(grid[row, col], **projection), parameters, s, positions, labels, bounds)
            _draw_gaussians(fig.add_subplot(grid[row + 1, col], **projection), parameters, s, positions, labels, bounds)
        _draw_constraints(fig.add_subplot(grid[-1, 0]), metrics, dataset_name, False)
        _draw_constraints(fig.add_subplot(grid[-1, 1]), metrics, dataset_name, True)
        _draw_parameter_errors(fig.add_subplot(grid[-1, 2]), metrics)
        fig.suptitle(f"{dataset_name} · HMM-LPV + MAP balanced pooled\n"
                     f"Constraint IoU {metrics.get('SemanticConstraintIoU', np.nan):.3f}  ·  "
                     f"Parameter error {metrics.get('MeanParameterError', np.nan):.4f}",
                     fontsize=15, fontweight="bold")
        fig.supxlabel("Original position coordinates · DS fields are unmodified by obstacles · "
                      "2σ: Mahalanobis radius 2 · —: inactive or no matching constraint", fontsize=9)
        return save_figure(fig, Path(save_path), dpi=180)
