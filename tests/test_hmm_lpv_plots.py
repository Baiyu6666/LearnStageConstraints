import numpy as np
import pytest

from visualization.hmm_lpv_plots import _raw_stage, plot_hmm_lpv_summary, stage_velocity


def _parameters(dimension=2):
    means = np.zeros((2, dimension))
    means[1, 0] = 2
    stage = {
        "attractor": np.zeros(dimension),
        "A": np.stack([-np.eye(dimension), -2 * np.eye(dimension)]),
        "priors": [0.25, 0.75],
        "position_means": means,
        "position_covariances": np.tile(np.eye(dimension), (2, 1, 1)),
    }
    return {"dimension": dimension, "position_indices": list(range(dimension)),
            "normalization_mean": np.arange(dimension) + 5.0,
            "normalization_scale": 3.0, "stages": [stage]}


@pytest.mark.parametrize("dimension", [2, 3])
def test_flow_uses_position_gate_and_restores_physical_coordinates(dimension):
    parameters = _parameters(dimension)
    origin = parameters["normalization_mean"]
    midpoint = origin.copy()
    midpoint[0] += 3  # Equal component densities; gate equals the mixture priors.
    expected = -1.75 * (midpoint - origin)
    np.testing.assert_allclose(stage_velocity(parameters, 0, midpoint[None])[0], expected)
    np.testing.assert_allclose(stage_velocity(parameters, 0, origin[None]), 0)
    far = np.full((1, dimension), 1e4)
    assert np.isfinite(stage_velocity(parameters, 0, far)).all()
    goal, means, covariance = _raw_stage(parameters, 0)
    np.testing.assert_allclose(goal, origin)
    np.testing.assert_allclose(means[1], origin + np.r_[6, np.zeros(dimension - 1)])
    np.testing.assert_allclose(covariance[0], 9 * np.eye(dimension))


@pytest.mark.parametrize("dimension", [2, 3])
def test_summary_renders_fields_gaussians_and_constraints_without_fitting(tmp_path, dimension):
    parameters = _parameters(dimension)
    origin = parameters["normalization_mean"]
    demo = origin + np.linspace(2, 0.1, 20)[:, None] * np.ones((1, dimension))
    metrics = {
        "ConstraintFeatureNames": ["distance"],
        "ConstraintTargetMatrix": [[0.2]],
        "ConstraintLearnedValueMatrix": [[0.21]],
        "ConstraintSemanticsMatrix": [["target_value"]],
        "ConstraintLearnedSemanticsMatrix": [["target_value"]],
        "ConstraintTrueActiveMask": [[True]],
        "ConstraintPredictedActiveMask": [[True]],
        "ParameterErrorMatrix": [[0.01]],
        "MeanAbsCutpointError": 0.0,
        "SemanticConstraintIoU": 1.0,
        "MeanParameterError": 0.01,
    }
    path = plot_hmm_lpv_summary(
        parameters=parameters, demos=[demo], stage_ends=[[19]],
        history={"initial_loglik": -5.0, "loglik": [-2.0, -1.0]}, metrics=metrics,
        true_cutpoints=[[]], save_path=tmp_path / f"summary_{dimension}d.png", dataset_name="Synthetic",
    )
    assert path.stat().st_size > 10000
