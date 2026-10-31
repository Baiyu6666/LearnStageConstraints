import itertools
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import logsumexp

from envs.base import TaskBundle
from experiments.artifacts import save_run_artifacts
from experiments.config_loader import inherit_map_posthoc_parameters, load_json
from methods.backends.hmm_lpv import (
    HMMLPV, LPVStage, _forward_backward, _ordered_viterbi,
    fit_lpv_surrogate, segment_with_hmm_lpv,
)
from methods.base import SegmentationResult, labels_to_cutpoints
from methods.registry import method_pipeline_kind


def test_time_dependent_inference_matches_enumerated_complete_paths():
    rng = np.random.default_rng(14)
    t_count, stages = 6, 3
    emissions = rng.normal(size=(t_count, stages)) - 1000
    transition = np.full((t_count - 1, stages, stages), -np.inf)
    for s in range(stages - 1):
        probability = rng.uniform(0.05, 0.95, t_count - 1)
        transition[:, s, s] = np.log1p(-probability)
        transition[:, s, s + 1] = np.log(probability)
    transition[:, -1, -1] = 0
    paths, scores = [], []
    for cuts in itertools.combinations(range(1, t_count), stages - 1):
        path = np.searchsorted(cuts, np.arange(t_count), side="right")
        score = emissions[np.arange(t_count), path].sum()
        score += transition[np.arange(t_count - 1), path[:-1], path[1:]].sum()
        paths.append(path)
        scores.append(score)
    likelihood = logsumexp(scores)
    expected_gamma = np.zeros_like(emissions)
    expected_xi = np.zeros_like(transition)
    for path, probability in zip(paths, np.exp(scores - likelihood)):
        expected_gamma[np.arange(t_count), path] += probability
        expected_xi[np.arange(t_count - 1), path[:-1], path[1:]] += probability
    gamma, xi, ll = _forward_backward(emissions, transition)
    np.testing.assert_allclose(ll, likelihood, atol=1e-10)
    np.testing.assert_allclose(gamma, expected_gamma, atol=1e-10)
    np.testing.assert_allclose(xi, expected_xi, atol=1e-10)
    np.testing.assert_array_equal(_ordered_viterbi(emissions, transition), paths[np.argmax(scores)])
    np.testing.assert_allclose(xi.sum(axis=2), gamma[:-1], atol=1e-10)
    np.testing.assert_allclose(xi.sum(axis=1), gamma[1:], atol=1e-10)


@pytest.mark.parametrize("dimension", [2, 3])
def test_lpv_surrogate_recovers_unknown_attractor_and_full_stable_matrices(dimension):
    rng = np.random.default_rng(19)
    goal = np.arange(dimension) * 0.3 + 0.2
    a = -np.eye(dimension)
    a[0, 1], a[1, 0] = 0.3, -0.3
    matrices = np.stack([a, 0.7 * a])
    x = rng.normal(size=(160, dimension))
    component = np.arange(len(x)) % 2
    velocity = np.einsum("nij,nj->ni", matrices[component], x - goal)
    weights = np.eye(2)[component]
    learned_goal, learned_a = fit_lpv_surrogate(x, velocity, weights)
    np.testing.assert_allclose(learned_goal, goal, atol=2e-4)
    np.testing.assert_allclose(learned_a, matrices, atol=2e-4)
    assert max(np.linalg.eigvalsh(m + m.T).max() for m in learned_a) <= -1e-6 + 1e-10


def test_joint_emission_termination_gradient_and_shared_goal():
    rng = np.random.default_rng(41)
    model = HMMLPV.__new__(HMMLPV)
    model.dimension, model.n_components = 3, 2
    model.stability_margin, model.precision_floor = 1e-6, 1e-8
    stage = LPVStage(
        np.array([0.2, 0.4, -0.1]), np.stack([-np.eye(3), -2 * np.eye(3)]),
        np.array([0.4, 0.6]), np.zeros((2, 3)), np.tile(np.eye(3), (2, 1, 1)),
        np.tile(np.eye(3), (2, 1, 1)), np.eye(3) * 0.7,
    )
    parameters = model._pack(stage, True)
    x, v = rng.normal(size=(20, 3)), rng.normal(size=(20, 3))
    weights = rng.uniform(0.1, 1.0, size=(20, 2))
    edge_x = x[:12]
    stay, leave = rng.uniform(size=12), rng.uniform(size=12)
    args = (x, v, weights, np.tile(np.eye(3), (2, 1, 1)), edge_x, stay, leave)
    _, analytic = model._joint_objective(parameters, *args)
    numeric = np.empty_like(parameters)
    for j in range(len(parameters)):
        delta = np.zeros_like(parameters)
        delta[j] = 1e-6
        numeric[j] = (model._joint_objective(parameters + delta, *args)[0]
                      - model._joint_objective(parameters - delta, *args)[0]) / 2e-6
    np.testing.assert_allclose(analytic, numeric, rtol=2e-5, atol=2e-6)
    model.stages, model.num_stages = [stage, stage], 2
    at_goal = model._transitions(np.stack([stage.goal, x[0]]))
    assert at_goal[0, 0, 1] == 0  # P(termination | attractor) = 1 exactly.
    assert np.isneginf(at_goal[0, 0, 0])
    transitions = model._transitions(x)
    np.testing.assert_allclose(np.exp(transitions).sum(axis=2), 1.0)
    assert np.ptp(np.exp(transitions[:, 0, 1])) > 0.01


@pytest.fixture(scope="module")
def trained_3d():
    class PositionOnlyEnv:
        dt = 0.12

        def compute_all_features_matrix(self, *_):
            raise AssertionError("HMM-LPV must not read environmental features")

    demos = []
    for index in range(3):
        x = np.array([-0.8, -0.2, 0.1]) + index * 0.05
        trajectory = []
        for goal in (np.array([0.6, 0.2, 0.7]), np.array([-0.1, 0.9, 0.0])):
            for _ in range(20):
                trajectory.append(x.copy())
                x = goal + (x - goal) * np.exp(-PositionOnlyEnv.dt * 1.5)
        # Non-position channels must not enter the dynamics or normalization.
        demos.append(np.column_stack([trajectory, np.full(len(trajectory), np.nan)]))
    labels, model, history, gammas = segment_with_hmm_lpv(
        demos, env=PositionOnlyEnv(), n_stages=2, n_components=1,
        init_mode="uniform", max_iter=3, joint_max_iter=60, verbose=False,
    )
    return demos, labels, model, history, gammas


def test_3d_training_feasible_monotone_and_position_only(trained_3d):
    demos, labels, model, history, gammas = trained_3d
    assert model.dimension == 3
    assert model.position_indices_ == [0, 1, 2]
    assert model.dt_ == 0.12
    assert history["optimization"][0]["accepted"]
    ll = [history["initial_loglik"], *history["loglik"]]
    assert np.min(np.diff(ll)) >= 0
    assert ll[-1] > ll[0]
    for x, z, gamma, ends in zip(demos, labels, gammas, model.stage_ends_):
        np.testing.assert_array_equal(np.unique(z), [0, 1])
        assert np.all(np.diff(z) >= 0)
        assert ends == np.flatnonzero(np.diff(z)).tolist() + [len(x) - 1]
        np.testing.assert_allclose(gamma.sum(axis=1), 1)
    for stage in model.stages:
        assert stage.matrices.shape == (1, 3, 3)
        for matrix in stage.matrices:
            assert np.linalg.eigvalsh(matrix + matrix.T).max() <= -model.stability_margin + 1e-10
        assert np.linalg.eigvalsh(stage.termination_precision).min() > 0
        assert np.linalg.eigvalsh(stage.noise).min() >= model.covariance_floor * 0.999
    expected_velocity = np.gradient(demos[0][:, :3], 0.12, axis=0) / model.position_scale_
    np.testing.assert_allclose(model.velocities[0], expected_velocity)


def test_registration_config_and_artifacts(trained_3d, tmp_path):
    demos, labels, model, history, _ = trained_3d
    assert method_pipeline_kind("hmm_lpv") == "sequential"
    config = load_json("configs/methods/hmm_lpv.json")
    inherited = inherit_map_posthoc_parameters(
        "hmm_lpv", config, {"map_balanced_pooled": {"map_eq_sigma": 0.03}},
        {"map_balanced_pooled": {"map_eq_sigma": 0.1}},
    )
    assert inherited["posthoc_constraint"]["posthoc_training_mode"] == "map_balanced_pooled"
    assert inherited["posthoc_constraint"]["map_eq_sigma"] == 0.1
    dataset = TaskBundle("synthetic", demos, true_cutpoints=[np.array([19])] * len(demos))
    segmentation = SegmentationResult("hmm_lpv", labels, labels_to_cutpoints(labels), model=model,
                                      extras={"segmentation_history": history})
    result = {"dataset": dataset, "segmentation": segmentation,
              "constraints": {"metrics": {}, "model": SimpleNamespace(posthoc_total_objective_=1.0)}}
    files = save_run_artifacts(
        run_dir=tmp_path, dataset_name="synthetic", method_name="hmm_lpv", result=result,
        dataset_kwargs={}, method_kwargs=inherited,
    )
    parameters = json.loads(files["segmentation_model"].read_text())
    assert parameters["dimension"] == 3
    np.testing.assert_allclose(parameters["stages"][0]["A"], model.stages[0].matrices)
    saved_history = json.loads(files["segmentation_training"].read_text())
    assert saved_history["loglik"] == history["loglik"]
    objectives = json.loads(files["objectives"].read_text())
    assert objectives["SegmentationLogLikelihoodFinal"] == history["loglik"][-1]


@pytest.mark.parametrize("dt", [0, -1, float("nan")])
def test_invalid_sample_interval_rejected(dt):
    with pytest.raises(ValueError, match="dt"):
        segment_with_hmm_lpv([np.ones((20, 3))], dt=dt)
