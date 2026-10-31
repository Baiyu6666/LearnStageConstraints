"""Generate new S3 trajectories from a saved HMM-LPV model; never fit a model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
import numpy as np

from envs.S3ObsAvoid import S3ObsAvoidEnv
from methods.backends.hmm_lpv_rollout import HMMLPVRollout


COLORS = ["#D55E00", "#0072B2", "#009E73"]


def render_rollouts(run_dir, *, starts, horizon=120., integration_dt=.02, output_dir=None,
                    correction_mode="paper_printed"):
    run_dir = Path(run_dir)
    parameters = json.loads((run_dir / "hmm_lpv_model.json").read_text())
    snapshot = json.loads((run_dir / "config_snapshot.json").read_text())
    if snapshot["dataset_name"] != "S3ObsAvoid" or parameters["dimension"] != 2:
        raise ValueError("This renderer is for the 2D S3 environment.")
    env = S3ObsAvoidEnv(**snapshot["dataset_kwargs"].get("env_kwargs", {}))
    model = HMMLPVRollout(parameters)
    output_dir = Path(output_dir or run_dir / "rollout")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, len(starts), figsize=(6 * len(starts), 10), squeeze=False,
                             gridspec_kw={"height_ratios": [2, 1, .55]}, constrained_layout=True)
    summaries = []
    for i, start in enumerate(starts):
        result = model.rollout(start, horizon=horizon, integration_dt=integration_dt, correction_mode=correction_mode)
        x, t, probabilities = result["positions"], result["time"], result["probabilities"]
        labels = probabilities.argmax(axis=1)
        ax, trace, committed = axes[:, i]
        ax.add_patch(Circle(env.obs_center, env.obs_radius, fc="#CBD5E1", ec="#475569", alpha=.65))
        for center, radius in zip(env.stage1_aux_obstacle_centers, env.stage1_aux_obstacle_radii):
            ax.add_patch(Circle(center, radius, fc="#CBD5E1", ec="#475569", alpha=.65))
        ax.add_patch(Circle(env.terminal_arc_center, env.terminal_arc_radius, fill=False,
                           ec="#94A3B8", ls="--", lw=1.2))
        segments = np.stack([x[:-1], x[1:]], axis=1)
        ax.add_collection(LineCollection(segments, colors=[COLORS[j] for j in labels[:-1]], linewidths=2))
        ax.scatter(*x[0], c="black", marker="o", s=45, zorder=5, label="Start")
        ax.scatter(*x[-1], c="black", marker="s", s=45, zorder=5, label="Last simulated point")
        goals = model.mean + model.scale * model.goals
        for s, goal in enumerate(goals):
            ax.scatter(*goal, c=COLORS[s], marker="*", s=130, edgecolors="white", zorder=5)
            ax.annotate(f"g{s + 1}", goal, xytext=(5, 5), textcoords="offset points")
        ax.update_datalim(np.vstack([x, goals, env.obs_center + [-.5, -.5], env.obs_center + [.5, .5]]))
        ax.autoscale_view()
        ax.set_aspect("equal", adjustable="box")
        ax.set(xlabel="x", ylabel="y", title=f"Start {tuple(start)}\n{result['stop_reason']} at {t[-1]:.1f} s")
        ax.grid(alpha=.15)
        ax.legend(loc="upper left", fontsize=8)
        for s in range(len(model.stages)):
            trace.plot(t, probabilities[:, s], color=COLORS[s], label=f"Stage {s + 1}")
        trace.set(xlabel="Time (s)", ylabel="Online stage probability", ylim=(-.03, 1.03))
        trace.legend(fontsize=8)
        trace.grid(alpha=.15)
        committed.step(t, result["current_stage"] + 1, where="post", color="#334155")
        committed.set(xlabel="Time (s)", ylabel="Committed stage", yticks=[1, 2, 3], ylim=(.8, 3.2))
        committed.grid(alpha=.15)
        for switch in result["switches"]:
            trace.axvline(switch["time"], color="#64748B", ls=":", lw=.8)
        centers = np.vstack([env.obs_center, env.stage1_aux_obstacle_centers])
        radii = np.r_[env.obs_radius, env.stage1_aux_obstacle_radii]
        clearance = np.min(np.linalg.norm(x[:, None] - centers[None], axis=2) - radii, axis=1)
        summary = {"start": list(start), "stop_reason": result["stop_reason"], "duration": float(t[-1]),
                   "switches": result["switches"], "final_position": x[-1].tolist(),
                   "distance_to_final_attractor": float(np.linalg.norm(x[-1] - goals[-1])),
                   "minimum_obstacle_surface_clearance": float(clearance.min()),
                   "collision_sample_fraction": float(np.mean(clearance < 0)),
                   "correction_sample_fraction": float(np.mean(np.linalg.norm(result["correction"], axis=1) > 1e-10)),
                   "negative_inward_progress_count": int(np.sum(result["inward_progress"] < -1e-10)),
                   "dominant_weight_change_count": int(np.sum(np.diff(labels) != 0)),
                   "max_speed": float(np.linalg.norm(result["velocity"], axis=1).max())}
        summaries.append(summary)
        np.savez_compressed(output_dir / f"rollout_{i:02d}.npz",
                            **{k: v for k, v in result.items() if isinstance(v, np.ndarray)})
    components = len(parameters["stages"][0]["priors"])
    fig.suptitle(f"S3: new HMM-LPV rollouts (C={components})\n"
                 f"Correction: {correction_mode}\nNo MAP constraint or obstacle correction", fontsize=14)
    fig.savefig(output_dir / "hmm_lpv_rollouts.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    metadata = {"model_path": str(run_dir / "hmm_lpv_model.json"), "training_performed": False,
                "observation_dt": model.period, "integration_dt": integration_dt, "horizon": horizon,
                "switch_tolerance": 1e-6, "goal_tolerance": .01, "correction_gain": 1.,
                "correction_formula": correction_mode,
                "filter_formula": "eq16_source_emission_then_pair_restricted_transition",
                "constraint_projection": False, "rollouts": summaries}
    (output_dir / "rollout_summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    print(output_dir / "hmm_lpv_rollouts.png")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--start", type=float, nargs=2, action="append")
    parser.add_argument("--horizon", type=float, default=120.)
    parser.add_argument("--integration-dt", type=float, default=.02)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--correction-mode", choices=["paper_printed", "lyapunov_projection"], default="paper_printed")
    args = parser.parse_args()
    render_rollouts(args.run_dir, starts=args.start or [[-1.6, .3], [-1.6, -.3], [-1.1, .65]],
                    horizon=args.horizon, integration_dt=args.integration_dt, output_dir=args.output_dir,
                    correction_mode=args.correction_mode)


if __name__ == "__main__":
    main()
