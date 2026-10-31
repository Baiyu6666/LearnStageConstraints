import numpy as np
import pytest

from methods.backends.hmm_lpv_rollout import HMMLPVRollout


def parameters(dimension=2, count=1):
    stages = [{"attractor": np.zeros(dimension).tolist(), "A": [-np.eye(dimension)],
               "priors": [1.], "position_means": [np.zeros(dimension)],
               "position_covariances": [np.eye(dimension)],
               "termination_precision": np.eye(dimension)} for _ in range(count)]
    return {"normalization_mean": np.arange(dimension), "normalization_scale": 2.,
            "dimension": dimension, "dt": .5, "stages": stages}


@pytest.mark.parametrize("dimension", [2, 3])
def test_single_stage_rollout_matches_analytic_solution_in_raw_coordinates(dimension):
    model = HMMLPVRollout(parameters(dimension))
    start = model.mean + np.ones(dimension)
    result = model.rollout(start, horizon=1., integration_dt=.02)
    expected = model.mean + np.exp(-result["time"][:, None]) * (start - model.mean)
    np.testing.assert_allclose(result["positions"], expected, atol=1e-8)
    assert result["stop_reason"] == "time_limit"
    assert not np.any(result["correction"])


def test_pair_filter_bayes_update_and_certain_transition_at_goal():
    model = HMMLPVRollout(parameters(count=3))
    p, current = model.update_probabilities(np.array([1., 0.]), np.array([1., 0., 0.]), 0, 1e-6)
    np.testing.assert_allclose(p, [1 - np.exp(-1), np.exp(-1), 0.])
    assert current == 0
    p, current = model.update_probabilities(np.zeros(2), p, 0, 1e-6)
    np.testing.assert_array_equal(p, [0., 1., 0.])
    assert current == 1  # No skipping over the next state.


def test_filter_clock_independent_of_integration_substeps():
    model = HMMLPVRollout(parameters(count=2))
    # Hold position fixed to isolate filtering from integration accuracy.
    model.matrices[:] = 0
    start = model.mean + [2., 0.]
    a = model.rollout(start, horizon=1., integration_dt=.1, correction_gain=0)
    b = model.rollout(start, horizon=1., integration_dt=.02, correction_gain=0)
    np.testing.assert_allclose(a["probabilities"][-1], b["probabilities"][-1])
    np.testing.assert_allclose(a["probabilities"][-1, 0], (1 - np.exp(-1)) ** 3)


def test_eq16_source_likelihood_does_not_gate_new_transition_by_destination_density():
    model = HMMLPVRollout(parameters(count=2))
    model.means[1] = 100.  # Destination likelihood is effectively zero here.
    p, current = model.update_probabilities(np.array([1., 0.]), np.array([1., 0.]), 0, 1e-6)
    # With certain source state, its likelihood cancels after normalization.
    np.testing.assert_allclose(p, [1 - np.exp(-1), np.exp(-1)])
    assert current == 0


def test_eq16_matches_explicit_source_sum_and_restricted_transition_matrix():
    model = HMMLPVRollout(parameters(count=3))
    model.means[1] = [.2, -.3]
    model.goals[1] = [.4, 0.]
    x, old = np.array([1., .2]), np.array([.7, .3, 0.])
    likelihood = np.exp(model.fields(x)[0][:2])
    beta = np.exp(-np.sum((x - model.goals[:2]) ** 2, axis=1))
    matrix = np.array([[1 - beta[0], beta[0]], [0., 1 - beta[1]]])
    expected = (old[:2] * likelihood) @ matrix
    expected /= expected.sum()
    actual, current = model.update_probabilities(x, old, 0, 1e-6)
    np.testing.assert_allclose(actual[:2], expected)
    assert actual[2] == 0 and current == 0


def test_printed_correction_and_goal_stop():
    model = HMMLPVRollout(parameters())
    model.matrices[:] = np.eye(2)  # Deliberately adverse field tests correction branch.
    velocity, correction = model.velocity(np.array([.5, 0]), np.ones(1), 0, 1.)
    np.testing.assert_allclose(correction, [-.75, 0.])
    np.testing.assert_allclose(velocity, [-.25, 0.])
    result = model.rollout(model.mean)
    assert result["stop_reason"] == "final_attractor_reached"
    assert len(result["positions"]) == 1


def test_projection_comparison_restores_inward_progress_when_printed_formula_fails():
    model = HMMLPVRollout(parameters())
    model.matrices[:] = 10 * np.eye(2)
    x = np.array([.1, 0.])
    printed, _ = model.velocity(x, np.ones(1), 0, 1.)
    projected, _ = model.velocity(x, np.ones(1), 0, 1., "lyapunov_projection")
    assert (-x) @ printed < 0
    np.testing.assert_allclose((-x) @ projected, x @ x)
