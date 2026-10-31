"""Offline HMM-LPV segmentation (Medina & Billard, CoRL 2017, sections 2–3).

LPV initialization follows jrmout/lpv-em's inverse-attractor / forward-refit
algorithm. Here A is the *negative* stable matrix, as in the MATLAB code,
whereas the paper writes -A. HMM emissions are p(x) p(velocity | x, stage).
Transitions use exp(-(x-g).T Q (x-g)), with the SAME learned attractor g.

The unpublished HMM optimizer is reconstructed as generalized Baum-Welch:
the source LPV surrogate proposes parameters, followed by joint optimization
of forward dynamics, attractor and termination precision. Feasible matrix
factorizations replace FMINSDP; likelihood-decreasing updates are rejected.
Only complete, left-to-right demonstrations are supported. No rollout code.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
from typing import Any

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
from scipy.special import logsumexp

from ..base import compute_cutpoint_metrics, format_training_log


def _spd(matrix: np.ndarray, floor: float) -> np.ndarray:
    eigenvalues, vectors = np.linalg.eigh((matrix + matrix.T) * 0.5)
    return (vectors * np.maximum(eigenvalues, floor)) @ vectors.T


def _log_gaussian(residual: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    chol = np.linalg.cholesky(covariance)
    whitened = solve_triangular(chol, residual.T, lower=True, check_finite=False)
    return -0.5 * (residual.shape[1] * np.log(2 * np.pi)
                   + 2 * np.log(np.diag(chol)).sum() + (whitened ** 2).sum(axis=0))


def _kmeans(values: np.ndarray, count: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Seeded k-means++, with ten restarts as in the author's initializer."""
    if len(values) < count:
        raise ValueError("Fewer initialization samples than mixture components.")
    best = None
    for _ in range(10):
        centers = [values[rng.integers(len(values))].copy()]
        for _ in range(1, count):
            dist = np.min(np.sum((values[:, None] - np.asarray(centers)) ** 2, axis=2), axis=1)
            idx = rng.choice(len(values), p=dist / dist.sum()) if dist.sum() > 0 else rng.integers(len(values))
            centers.append(values[idx].copy())
        centers = np.asarray(centers)
        for _ in range(50):
            distances = np.sum((values[:, None] - centers) ** 2, axis=2)
            labels = np.argmin(distances, axis=1)
            updated = centers.copy()
            for k in range(count):
                members = values[labels == k]
                updated[k] = members.mean(axis=0) if len(members) else values[np.argmax(distances.min(axis=1))]
            if np.allclose(updated, centers, rtol=0, atol=1e-8):
                break
            centers = updated
        distances = np.sum((values[:, None] - centers) ** 2, axis=2)
        labels = np.argmin(distances, axis=1)
        cost = float(distances[np.arange(len(values)), labels].sum())
        if best is None or cost < best[0]:
            best = cost, centers.copy(), labels.copy()
    return best[1], best[2]


def _ordered_viterbi(emissions: np.ndarray, transitions: np.ndarray) -> np.ndarray:
    t_count, stages = emissions.shape
    score = np.full((t_count, stages), -np.inf)
    back = np.zeros((t_count, stages), dtype=int)
    score[0, 0] = emissions[0, 0]
    for t in range(1, t_count):
        candidates = score[t - 1, :, None] + transitions[t - 1]
        back[t] = np.argmax(candidates, axis=0)
        score[t] = emissions[t] + np.max(candidates, axis=0)
    if not np.isfinite(score[-1, -1]):
        raise RuntimeError("No finite complete left-to-right HMM-LPV path.")
    labels = np.empty(t_count, dtype=int)
    labels[-1] = stages - 1
    for t in range(t_count - 1, 0, -1):
        labels[t - 1] = back[t, labels[t]]
    return labels


def _forward_backward(emissions: np.ndarray, transitions: np.ndarray):
    """Condition on start=0 and final=S-1, as for other ordered baselines."""
    t_count, stages = emissions.shape
    alpha = np.full_like(emissions, -np.inf)
    alpha[0, 0] = emissions[0, 0]
    for t in range(1, t_count):
        alpha[t] = emissions[t] + logsumexp(alpha[t - 1, :, None] + transitions[t - 1], axis=0)
    ll = float(alpha[-1, -1])
    if not np.isfinite(ll):
        raise RuntimeError("No finite complete HMM-LPV trajectory likelihood.")
    beta = np.full_like(emissions, -np.inf)
    beta[-1, -1] = 0.0
    for t in range(t_count - 2, -1, -1):
        beta[t] = logsumexp(transitions[t] + emissions[t + 1][None, :] + beta[t + 1][None, :], axis=1)
    gamma = np.exp(alpha + beta - ll)
    gamma /= gamma.sum(axis=1, keepdims=True)
    xi = np.exp(alpha[:-1, :, None] + transitions + emissions[1:, None, :] + beta[1:, None, :] - ll)
    return gamma, xi, ll


def fit_lpv_surrogate(x, velocity, weights, *, stability_margin=1e-6, inverse_margin=0.5, solver="CLARABEL"):
    """Port of estimate_stable_mix_lds_inv_max.m (MIT; see lpv_em.LICENSE).

    Preserve the author's component-mass weighting: N_c * sum_i w_ic e_ic²
    (a common overall factor does not change either optimum). Full matrices,
    not symmetric-only variables, allow rotational dynamics in 2D and 3D.
    """
    import cvxpy as cp

    x, velocity, weights = map(lambda a: np.asarray(a, dtype=float), (x, velocity, weights))
    n, d = x.shape
    if velocity.shape != x.shape or weights.ndim != 2 or weights.shape[0] != n:
        raise ValueError("LPV surrogate expects x/velocity (N,D) and weights (N,C).")
    if not all(np.isfinite(a).all() for a in (x, velocity, weights)) or np.any(weights < 0):
        raise ValueError("LPV inputs must be finite with nonnegative weights.")
    if stability_margin <= 0 or inverse_margin <= 0 or weights.sum() <= 0:
        raise ValueError("LPV stability margins and total responsibility must be positive.")
    if solver not in {"CLARABEL", "SCS"} or solver not in cp.installed_solvers():
        raise ValueError("HMM-LPV requires an installed CLARABEL or SCS SDP solver.")
    masses = weights.sum(axis=0)
    normalized = weights / weights.sum()
    root_weights = np.sqrt(normalized * (masses / weights.sum())[None, :])
    goal = cp.Variable(d)
    inverse = [cp.Variable((d, d)) for _ in masses]
    objective, constraints = 0, []
    for c, b in enumerate(inverse):
        objective += cp.sum_squares(cp.multiply(root_weights[:, c, None], x + velocity @ b.T - goal))
        constraints.append(b + b.T >> inverse_margin * np.eye(d))

    def solve(objective, constraints):
        problem = cp.Problem(cp.Minimize(objective), constraints)
        options = {"eps": 1e-7, "max_iters": 20000} if solver == "SCS" else {
            "tol_gap_abs": 1e-9, "tol_feas": 1e-9, "tol_gap_rel": 1e-9,
        }
        problem.solve(solver=solver, **options)
        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(f"LPV SDP failed: {problem.status}")

    solve(objective, constraints)
    g = np.asarray(goal.value, dtype=float)
    b_values = np.asarray([b.value for b in inverse], dtype=float)
    if not np.isfinite(g).all() or not np.isfinite(b_values).all():
        raise RuntimeError("LPV inverse SDP returned nonfinite parameters.")
    if min(np.linalg.eigvalsh(b + b.T).min() for b in b_values) < inverse_margin - 1e-6:
        raise RuntimeError("LPV inverse SDP violated its matrix inequality.")

    forward = [cp.Variable((d, d)) for _ in masses]
    objective, constraints = 0, []
    for c, a in enumerate(forward):
        objective += cp.sum_squares(cp.multiply(root_weights[:, c, None], (x - g) @ a.T - velocity))
        constraints.append(a + a.T << -stability_margin * np.eye(d))
    solve(objective, constraints)
    matrices = np.asarray([a.value for a in forward], dtype=float)
    if not np.isfinite(matrices).all():
        raise RuntimeError("LPV forward SDP returned nonfinite matrices.")
    # Repair only solver roundoff, never silently accept an unstable solve.
    for a in matrices:
        violation = np.linalg.eigvalsh(a + a.T).max() + stability_margin
        if violation > 1e-6:
            raise RuntimeError("LPV forward SDP violated its stability inequality.")
        if violation > 0:
            a -= (0.5 * violation + 1e-12) * np.eye(d)
    return g, matrices


@dataclass
class LPVStage:
    goal: np.ndarray
    matrices: np.ndarray
    priors: np.ndarray
    means: np.ndarray
    covariances: np.ndarray
    noise: np.ndarray
    termination_precision: np.ndarray


class HMMLPV:
    def __init__(self, positions, velocities, *, n_stages, n_components=7, seed=0,
                 covariance_floor=1e-5, stability_margin=1e-6, inverse_margin=0.5,
                 solver="CLARABEL", joint_max_iter=100, init_mode="kmeans"):
        self.positions = [np.asarray(x, dtype=float) for x in positions]
        self.velocities = [np.asarray(v, dtype=float) for v in velocities]
        self.num_stages, self.n_components = int(n_stages), int(n_components)
        self.covariance_floor = float(covariance_floor)
        self.stability_margin, self.inverse_margin = float(stability_margin), float(inverse_margin)
        self.solver, self.joint_max_iter = str(solver), int(joint_max_iter)
        self.precision_floor = 1e-8
        if not self.positions or len(self.positions) != len(self.velocities):
            raise ValueError("Provide matching nonempty position and velocity sequences.")
        if self.num_stages < 1 or self.n_components < 1 or self.joint_max_iter < 1:
            raise ValueError("Stage/component/optimizer iteration counts must be positive.")
        if min(self.covariance_floor, self.stability_margin, self.inverse_margin) <= 0:
            raise ValueError("Covariance and stability floors must be positive.")
        self.dimension = self.positions[0].shape[1]
        for x, v in zip(self.positions, self.velocities):
            if x.ndim != 2 or v.shape != x.shape or x.shape[1] != self.dimension:
                raise ValueError("Inconsistent position/velocity dimensions.")
            if len(x) < max(self.num_stages, 2) or not np.isfinite(x).all() or not np.isfinite(v).all():
                raise ValueError("Each demonstration must be finite and long enough for all stages.")
        self.rng = np.random.default_rng(seed)
        self.loss_loglik = []
        self.optimization_history_ = []
        self.converged_ = False
        self.stop_reason_ = "max_iter"
        self._initialize(init_mode)

    def _initial_labels(self, init_mode):
        if init_mode == "uniform":
            return [np.minimum(np.arange(len(x)) * self.num_stages // len(x), self.num_stages - 1)
                    for x in self.positions]
        if init_mode != "kmeans":
            raise ValueError("HMM-LPV init_mode must be kmeans or uniform.")
        observations = [np.hstack([x, 0.5 * v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)])
                        for x, v in zip(self.positions, self.velocities)]
        stacked = np.vstack(observations)
        centers, labels = _kmeans(stacked, self.num_stages, self.rng)
        progress = np.concatenate([np.linspace(0, 1, len(x)) for x in self.positions])
        order = np.argsort([progress[labels == s].mean() if np.any(labels == s) else 1
                            for s in range(self.num_stages)], kind="stable")
        centers = centers[order]
        results = []
        for values in observations:
            costs = -np.sum((values[:, None] - centers) ** 2, axis=2)
            transition = np.full((len(values) - 1, self.num_stages, self.num_stages), -np.inf)
            for s in range(self.num_stages):
                transition[:, s, s] = 0
                if s + 1 < self.num_stages:
                    transition[:, s, s + 1] = 0
            results.append(_ordered_viterbi(costs, transition))
        return results

    def _initialize(self, init_mode):
        labels = self._initial_labels(init_mode)
        self.stage_ends_ = self._stage_ends(labels)
        self.initial_stage_ends_ = copy.deepcopy(self.stage_ends_)
        self.segmentation_history_ = [copy.deepcopy(self.stage_ends_)]
        x, v, z = np.vstack(self.positions), np.vstack(self.velocities), np.concatenate(labels)
        self.stages = []
        for s in range(self.num_stages):
            xs, vs = x[z == s], v[z == s]
            direction = vs / np.maximum(np.linalg.norm(vs, axis=1, keepdims=True), 1e-12)
            _, components = _kmeans(np.hstack([xs, 0.5 * direction]), self.n_components, self.rng)
            weights = np.eye(self.n_components)[components] + 1e-8
            goal, matrices = fit_lpv_surrogate(xs, vs, weights, stability_margin=self.stability_margin,
                                              inverse_margin=self.inverse_margin, solver=self.solver)
            covariance = _spd(np.atleast_2d(np.cov(xs.T, bias=True)), self.covariance_floor)
            stage = LPVStage(goal, matrices, np.ones(self.n_components) / self.n_components,
                             np.zeros((self.n_components, self.dimension)),
                             np.tile(covariance, (self.n_components, 1, 1)),
                             np.tile(np.eye(self.dimension), (self.n_components, 1, 1)),
                             np.linalg.inv(covariance))
            self._update_gaussians(stage, xs, vs, weights)
            self.stages.append(stage)

    @staticmethod
    def _stage_ends(labels):
        return [np.flatnonzero(np.diff(z)).tolist() + [len(z) - 1] for z in labels]

    def _update_gaussians(self, stage, x, v, weights):
        counts = weights.sum(axis=0)
        stage.priors = np.maximum(counts, 1e-10)
        stage.priors /= stage.priors.sum()
        for c, count in enumerate(counts):
            if count < 1e-8:
                continue  # Keep a valid density for a temporarily unoccupied component.
            w = weights[:, c] / count
            stage.means[c] = w @ x
            dx = x - stage.means[c]
            stage.covariances[c] = _spd((dx.T * w) @ dx, self.covariance_floor)
            error = v - (x - stage.goal) @ stage.matrices[c].T
            stage.noise[c] = _spd((error.T * w) @ error, self.covariance_floor)

    def _emissions(self, x, v):
        components = np.empty((len(x), self.num_stages, self.n_components))
        for s, stage in enumerate(self.stages):
            for c in range(self.n_components):
                error = v - (x - stage.goal) @ stage.matrices[c].T
                components[:, s, c] = (np.log(stage.priors[c])
                    + _log_gaussian(x - stage.means[c], stage.covariances[c])
                    + _log_gaussian(error, stage.noise[c]))
        return logsumexp(components, axis=2), components

    def _transitions(self, x):
        transitions = np.full((len(x) - 1, self.num_stages, self.num_stages), -np.inf)
        for s, stage in enumerate(self.stages[:-1]):
            delta = x[:-1] - stage.goal
            q = np.maximum(np.einsum("ni,ij,nj->n", delta, stage.termination_precision, delta), 0)
            transitions[:, s, s + 1] = -q
            with np.errstate(divide="ignore"):
                transitions[:, s, s] = np.log(-np.expm1(-q))
        transitions[:, -1, -1] = 0  # Last stage is absorbing for complete noncyclic tasks.
        return transitions

    def _expectation(self):
        gammas, xis, weights, total = [], [], [], 0.0
        for x, v in zip(self.positions, self.velocities):
            emissions, components = self._emissions(x, v)
            gamma, xi, ll = _forward_backward(emissions, self._transitions(x))
            gammas.append(gamma)
            xis.append(xi)
            weights.append(gamma[:, :, None] * np.exp(components - emissions[:, :, None]))
            total += ll
        return gammas, xis, weights, total

    def _pack(self, stage, with_termination):
        d = self.dimension
        tril = np.tril_indices(d)
        values = [stage.goal]
        for a in stage.matrices:
            values.append(((a - a.T) * 0.5).ravel())
            positive = _spd(-(a + a.T) * 0.5 - self.stability_margin * np.eye(d) / 2, 1e-10)
            values.append(np.linalg.cholesky(positive)[tril])
        if with_termination:
            positive = _spd(stage.termination_precision - self.precision_floor * np.eye(d), 1e-10)
            values.append(np.linalg.cholesky(positive)[tril])
        return np.concatenate(values)

    def _unpack(self, parameters, with_termination):
        d = self.dimension
        tril = np.tril_indices(d)
        n_lower = len(tril[0])
        offset, matrices, factors = d, [], []
        for _ in range(self.n_components):
            raw = parameters[offset:offset + d * d].reshape(d, d)
            offset += d * d
            lower = np.zeros((d, d))
            lower[tril] = parameters[offset:offset + n_lower]
            offset += n_lower
            matrices.append((raw - raw.T) * 0.5 - lower @ lower.T - self.stability_margin * np.eye(d) / 2)
            factors.append(lower)
        q_lower = np.zeros((d, d))
        if with_termination:
            q_lower[tril] = parameters[offset:offset + n_lower]
        precision = q_lower @ q_lower.T + self.precision_floor * np.eye(d)
        return parameters[:d], np.asarray(matrices), factors, precision, q_lower

    def _joint_objective(self, parameters, x, v, weights, noise_precisions, edge_x, stay, leave):
        """Expected negative log likelihood and analytic gradient (fixed covariances).

        A = skew(R) - L L.T - eps I/2, Q = U U.T + eps_q I keep all
        optimization iterates feasible. g is shared by emission and termination.
        No heuristic transition weight: both terms are the Baum-Welch Q terms.
        """
        active = len(edge_x) > 0
        goal, matrices, factors, precision, q_lower = self._unpack(parameters, active)
        delta = x - goal
        value, goal_gradient, matrix_gradients = 0.0, np.zeros(self.dimension), []
        for c, a in enumerate(matrices):
            error = v - delta @ a.T
            weighted = (error @ noise_precisions[c]) * weights[:, c, None]
            value += 0.5 * np.sum(weighted * error)
            goal_gradient += weighted.sum(axis=0) @ a
            matrix_gradients.append(-weighted.T @ delta)
        precision_gradient = np.zeros_like(precision)
        if active:
            edge_delta = edge_x - goal
            q = np.maximum(np.einsum("ni,ij,nj->n", edge_delta, precision, edge_delta), 1e-14)
            # log(1-exp(-q)) and its derivative, without overflow for large q.
            survival = -np.expm1(-q)
            value += np.dot(leave, q) - np.dot(stay, np.log(survival))
            dq = leave - stay * np.exp(-q) / survival
            goal_gradient -= 2 * np.sum(dq[:, None] * (edge_delta @ precision), axis=0)
            precision_gradient = (edge_delta.T * dq) @ edge_delta
        gradient = [goal_gradient]
        tril = np.tril_indices(self.dimension)
        for ga, lower in zip(matrix_gradients, factors):
            gradient.append(((ga - ga.T) * 0.5).ravel())
            gradient.append((-(ga + ga.T) @ lower)[tril])
        if active:
            gradient.append(((precision_gradient + precision_gradient.T) @ q_lower)[tril])
        scale = max(float(weights.sum()), 1.0)
        return float(value / scale), np.concatenate(gradient) / scale

    def _maximization(self, xis, weights):
        x, v, w = np.vstack(self.positions), np.vstack(self.velocities), np.concatenate(weights)
        edge_x = np.vstack([values[:-1] for values in self.positions])
        diagnostics = []
        for s, stage in enumerate(self.stages):
            active = s + 1 < self.num_stages
            stay = np.concatenate([xi[:, s, s] for xi in xis]) if active else np.empty(0)
            leave = np.concatenate([xi[:, s, s + 1] for xi in xis]) if active else np.empty(0)
            args = (x, v, w[:, s], np.linalg.inv(stage.noise), edge_x if active else np.empty((0, self.dimension)), stay, leave)
            old_parameters = self._pack(stage, active)
            # Source LPV-EM surrogate is a proposal, not an exact HMM M-step.
            proposed = copy.deepcopy(stage)
            proposed.goal, proposed.matrices = fit_lpv_surrogate(
                x, v, w[:, s], stability_margin=self.stability_margin,
                inverse_margin=self.inverse_margin, solver=self.solver)
            proposal = self._pack(proposed, active)
            old_value = self._joint_objective(old_parameters, *args)[0]
            proposal_value = self._joint_objective(proposal, *args)[0]
            initial = proposal if proposal_value < old_value else old_parameters
            initial_value = min(old_value, proposal_value)
            result = minimize(self._joint_objective, initial, args=args, jac=True, method="L-BFGS-B",
                              options={"maxiter": self.joint_max_iter, "ftol": 1e-10, "gtol": 1e-6, "maxls": 40})
            final_value = self._joint_objective(result.x, *args)[0]
            accepted = bool(np.isfinite(final_value) and final_value <= initial_value)
            chosen = result.x if accepted else initial
            stage.goal, stage.matrices, _, precision, _ = self._unpack(chosen, active)
            if active:
                stage.termination_precision = precision
            self._update_gaussians(stage, x, v, w[:, s])
            diagnostics.append({"stage": s, "optimizer_success": bool(result.success),
                                "optimizer_message": str(result.message), "iterations": int(result.nit),
                                "joint_nll_before": old_value, "joint_nll_after": min(initial_value, final_value) if accepted else initial_value,
                                "surrogate_selected": bool(proposal_value < old_value)})
        return diagnostics

    def fit(self, *, max_iter=30, tol=1e-4, verbose=True, true_cutpoints=None):
        if int(max_iter) < 1 or float(tol) < 0:
            raise ValueError("max_iter must be positive and tol nonnegative.")
        gammas, xis, weights, previous = self._expectation()
        self.initial_loglik_ = previous
        for iteration in range(int(max_iter)):
            old_stages = copy.deepcopy(self.stages)
            diagnostics = self._maximization(xis, weights)
            new_gammas, new_xis, new_weights, total = self._expectation()
            accepted = total >= previous
            if not accepted:
                self.stages = old_stages
                total = previous
                self.stop_reason_ = "likelihood_update_rejected"
            else:
                gammas, xis, weights = new_gammas, new_xis, new_weights
            self.loss_loglik.append(float(total))
            self.optimization_history_.append({"accepted": accepted, "stages": diagnostics})
            labels = self.decode()
            self.stage_ends_ = self._stage_ends(labels)
            self.segmentation_history_.append(copy.deepcopy(self.stage_ends_))
            improvement = total - previous
            metrics = compute_cutpoint_metrics([ends[:-1] for ends in self.stage_ends_], true_cutpoints, self.positions)
            if verbose:
                print(format_training_log("HMM-LPV", iteration, losses={"loglik": total}, metrics=metrics,
                                          extras={"improvement": round(improvement, 6), "accepted": accepted}), flush=True)
            if not accepted:
                break
            if improvement <= float(tol) * (1 + abs(previous)):
                self.converged_, self.stop_reason_ = True, "likelihood_tolerance"
                break
            previous = total
        history = {"initial_loglik": self.initial_loglik_, "loglik": list(self.loss_loglik),
                   "stage_ends": self.segmentation_history_, "optimization": self.optimization_history_,
                   "stop_reason": self.stop_reason_, "converged": self.converged_}
        return self.decode(), gammas, history

    def decode(self):
        return [_ordered_viterbi(self._emissions(x, v)[0], self._transitions(x))
                for x, v in zip(self.positions, self.velocities)]

    def parameter_dict(self):
        return {"model": "HMM-LPV", "topology": "complete_left_to_right", "dimension": self.dimension,
                "normalization_mean": self.position_mean_, "normalization_scale": self.position_scale_,
                "position_indices": self.position_indices_, "dt": self.dt_,
                "stability_margin": self.stability_margin, "inverse_margin": self.inverse_margin,
                "covariance_floor": self.covariance_floor,
                "matrix_convention": "velocity = A @ (position - attractor)",
                "stages": [{"attractor": s.goal, "A": s.matrices, "priors": s.priors,
                            "position_means": s.means, "position_covariances": s.covariances,
                            "velocity_noise_covariances": s.noise, "termination_precision": s.termination_precision}
                           for s in self.stages]}


def segment_with_hmm_lpv(demos, *, env=None, n_stages=3, n_components=7, dt=None,
                         position_indices=None, standardize=True, seed=0, max_iter=30,
                         tol=1e-4, covariance_floor=1e-5, stability_margin=1e-6,
                         inverse_margin=0.5, solver="CLARABEL", joint_max_iter=100,
                         init_mode="kmeans", verbose=True, true_cutpoints=None):
    if not demos:
        raise ValueError("HMM-LPV needs at least one demonstration.")
    if position_indices is None:
        position_indices = list(range(min(3, np.asarray(demos[0]).shape[1])))
    indices = np.asarray(position_indices, dtype=int)
    if len(indices) not in (2, 3) or len(set(indices.tolist())) != len(indices) or np.any(indices < 0):
        raise ValueError("Select exactly two or three distinct position columns.")
    time_step = float(dt if dt is not None else getattr(env, "dt", 1.0))
    if not np.isfinite(time_step) or time_step <= 0:
        raise ValueError("HMM-LPV dt must be finite and positive.")
    positions = []
    for demo in demos:
        values = np.asarray(demo, dtype=float)
        if values.ndim != 2 or len(values) < max(2, int(n_stages)) or indices.max() >= values.shape[1]:
            raise ValueError("Invalid demonstration length or position columns.")
        x = values[:, indices].copy()
        if not np.isfinite(x).all():
            raise ValueError("Position observations contain nonfinite values.")
        positions.append(x)
    stacked = np.vstack(positions)
    mean = stacked.mean(axis=0) if standardize else np.zeros(len(indices))
    # One scalar spatial scale preserves the Euclidean Lyapunov inequality.
    # Position and its derivative must use the SAME scale, with no velocity centering.
    scale = max(float(np.sqrt(np.mean((stacked - mean) ** 2))), 1e-8) if standardize else 1.0
    positions = [(x - mean) / scale for x in positions]
    velocities = [np.gradient(x, time_step, axis=0, edge_order=1) for x in positions]
    model = HMMLPV(positions, velocities, n_stages=n_stages, n_components=n_components,
                   seed=seed, covariance_floor=covariance_floor, stability_margin=stability_margin,
                   inverse_margin=inverse_margin, solver=solver, joint_max_iter=joint_max_iter,
                   init_mode=init_mode)
    model.position_mean_, model.position_scale_ = mean, scale
    model.position_indices_, model.dt_ = indices.tolist(), time_step
    labels, gammas, history = model.fit(max_iter=max_iter, tol=tol, verbose=verbose, true_cutpoints=true_cutpoints)
    return labels, model, history, gammas
