"""Saved-model HMM-LPV execution, Medina & Billard (2017), eqs. 15–19.

No demonstrations, constraint learner or obstacle projector enter this policy.
Filtering uses the training observation period; RK4 substeps only integrate
the velocity field. Equation 16 weights the SOURCE state by its position
likelihood before applying transitions. Proposition 2 restricts the sum to the
current/next pair: mass leaving that pair is omitted, then renormalized. Only
the final state of the complete left-to-right model is absorbing.
Equation 17's exact probability-one test uses an explicit numerical tolerance.
The correction follows eq. 19 as PRINTED (denominator ||l||, not ||l||²).
Consequently we record inward-progress failures rather than assuming a global
stability guarantee for this printed expression or its discrete integration.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


class HMMLPVRollout:
    def __init__(self, parameters):
        self.mean = np.asarray(parameters["normalization_mean"], dtype=float)
        self.scale = float(parameters["normalization_scale"])
        self.dimension = int(parameters["dimension"])
        self.period = float(parameters["dt"])
        self.stages = parameters["stages"]
        if self.scale <= 0 or self.period <= 0 or not self.stages:
            raise ValueError("Invalid model scale, observation period or stages.")
        self.goals = np.asarray([s["attractor"] for s in self.stages])
        self.matrices = np.asarray([s["A"] for s in self.stages])
        self.means = np.asarray([s["position_means"] for s in self.stages])
        cov = np.asarray([s["position_covariances"] for s in self.stages])
        self.precisions = np.linalg.inv(cov)
        signs, logdet = np.linalg.slogdet(cov)
        if np.any(signs <= 0):
            raise ValueError("Position covariances must be positive definite.")
        self.log_constants = (np.log(np.asarray([s["priors"] for s in self.stages]))
                              - .5 * (logdet + self.dimension * np.log(2 * np.pi)))
        self.termination = np.asarray([s["termination_precision"] for s in self.stages])

    def fields(self, x):
        """Position marginal likelihood and component-gated DS in normalized units."""
        delta = x - self.means
        logs = self.log_constants - .5 * np.einsum("sci,scij,scj->sc", delta, self.precisions, delta)
        density = logsumexp(logs, axis=1)
        gates = np.exp(logs - density[:, None])
        local = np.einsum("scij,sj->sci", self.matrices, x - self.goals)
        return density, np.einsum("sc,sci->si", gates, local)

    def update_probabilities(self, x, probabilities, current, switch_tolerance):
        if current == len(self.stages) - 1:
            return probabilities, current
        delta = x - self.goals[current]
        q = max(float(delta @ self.termination[current] @ delta), 0.)
        log_beta = -q
        log_stay = np.log(-np.expm1(-q)) if q > 0 else -np.inf
        with np.errstate(divide="ignore"):
            log_p = np.log(probabilities[current:current + 2])
        # The printed eq. 16 is p(x_i | s_i), NOT p(x_i | s_destination).
        source = log_p + self.fields(x)[0][current:current + 2]
        next_log_stay = 0.
        if current + 1 < len(self.stages) - 1:
            next_delta = x - self.goals[current + 1]
            next_q = max(float(next_delta @ self.termination[current + 1] @ next_delta), 0.)
            next_log_stay = np.log(-np.expm1(-next_q)) if next_q > 0 else -np.inf
        posterior = np.array([source[0] + log_stay,
                              np.logaddexp(source[1] + next_log_stay, source[0] + log_beta)])
        if not np.isfinite(logsumexp(posterior)):
            raise RuntimeError("No probability mass remains in the current/next pair.")
        posterior = np.exp(posterior - logsumexp(posterior))
        updated = np.zeros_like(probabilities)
        updated[current:current + 2] = posterior
        if posterior[1] >= 1 - switch_tolerance:
            current += 1
            updated[:] = 0
            updated[current] = 1
        return updated, current

    def velocity(self, x, probabilities, current, correction_gain, correction_mode="paper_printed"):
        mixed = probabilities @ self.fields(x)[1]
        direction = self.goals[current] - x
        norm = float(np.linalg.norm(direction))
        progress = float(direction @ mixed)
        correction = np.zeros(self.dimension)
        if progress <= 0 and norm > 1e-12 and correction_gain > 0:
            # The squared norm yields l.T(v+corr) = gain * ||l||² > 0.
            # Keep this mathematical repair explicit, separate from the paper.
            denominator = norm ** 2 if correction_mode == "lyapunov_projection" else norm
            correction = (-progress / denominator + correction_gain) * direction
        return mixed + correction, correction

    def rollout(self, start, *, horizon=120., integration_dt=.02,
                switch_tolerance=1e-6, goal_tolerance=.01, correction_gain=1.,
                correction_mode="paper_printed"):
        start = np.asarray(start, dtype=float)
        if start.shape != (self.dimension,) or not np.isfinite(start).all():
            raise ValueError("Start must be a finite position of the model dimension.")
        if (horizon <= 0 or integration_dt <= 0 or not 0 < switch_tolerance < .5
                or goal_tolerance <= 0 or correction_gain < 0):
            raise ValueError("Invalid rollout integration settings.")
        if correction_mode not in {"paper_printed", "lyapunov_projection"}:
            raise ValueError("Unknown correction mode.")
        x = (start - self.mean) / self.scale
        probabilities = np.zeros(len(self.stages))
        probabilities[0] = 1.
        current, t, next_observation = 0, 0., 0.
        rows, switches = [], []
        reason = "time_limit"
        while True:
            if t >= next_observation - 1e-10:
                old = current
                probabilities, current = self.update_probabilities(x, probabilities, current, switch_tolerance)
                if current != old:
                    switches.append({"time": t, "from_stage": old + 1, "to_stage": current + 1})
                next_observation += self.period
            velocity, correction = self.velocity(x, probabilities, current, correction_gain, correction_mode)
            progress = float((self.goals[current] - x) @ velocity)
            rows.append((t, x.copy(), probabilities.copy(), current, velocity, correction, progress))
            if current == len(self.stages) - 1 and np.linalg.norm(x - self.goals[-1]) * self.scale <= goal_tolerance:
                reason = "final_attractor_reached"
                break
            if t >= horizon - 1e-10:
                break
            dt = min(integration_dt, next_observation - t, horizon - t)
            field = lambda point: self.velocity(point, probabilities, current, correction_gain, correction_mode)[0]
            k1 = velocity
            k2 = field(x + .5 * dt * k1)
            k3 = field(x + .5 * dt * k2)
            k4 = field(x + dt * k3)
            x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += dt
            if not np.isfinite(x).all() or np.linalg.norm(x) > 1e6:
                raise RuntimeError("Rollout diverged; no trajectory was silently clipped.")
        return {
            "time": np.asarray([r[0] for r in rows]),
            "positions": self.mean + self.scale * np.asarray([r[1] for r in rows]),
            "probabilities": np.asarray([r[2] for r in rows]),
            "current_stage": np.asarray([r[3] for r in rows]),
            "velocity": self.scale * np.asarray([r[4] for r in rows]),
            "correction": self.scale * np.asarray([r[5] for r in rows]),
            "inward_progress": np.asarray([r[6] for r in rows]),
            "switches": switches, "stop_reason": reason,
        }
