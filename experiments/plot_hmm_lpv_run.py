"""Replot an exported HMM-LPV run without fitting segmentation or constraints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from envs import load_env
from experiments.artifacts import dataset_fingerprint
from visualization.hmm_lpv_plots import plot_hmm_lpv_summary


def plot_saved_run(run_dir: Path):
    run_dir = Path(run_dir)
    read = lambda name: json.loads((run_dir / name).read_text(encoding="utf-8"))
    snapshot, metadata = read("config_snapshot.json"), read("metadata.json")
    parameters = read("hmm_lpv_model.json")
    history, segmentation = read("hmm_lpv_training.json"), read("segmentation.json")
    metrics = read("metrics.json")["all_metrics"]
    # Existing artifacts contain seeds/configuration, not raw trajectories.
    # Use the standard loader and reject changed data rather than plotting a
    # saved model against a different demonstration set.
    dataset = load_env(snapshot["dataset_name"], **snapshot["dataset_kwargs"])
    if dataset_fingerprint(dataset) != metadata["dataset_fingerprint"]:
        raise ValueError("Dataset fingerprint changed; cannot replot this saved run faithfully.")
    return plot_hmm_lpv_summary(
        parameters=parameters, demos=dataset.demos, stage_ends=segmentation["predicted_stage_ends"],
        history=history, metrics=metrics, true_cutpoints=dataset.true_cutpoints, env=dataset.env,
        dataset_name=dataset.name, save_path=run_dir / "training_summary_posthoc_final.png",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(plot_saved_run(args.run_dir))


if __name__ == "__main__":
    main()
