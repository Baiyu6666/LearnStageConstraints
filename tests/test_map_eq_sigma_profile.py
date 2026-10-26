import numpy as np
import pytest

from methods.cores.map import StageWiseMAPConstraintLearningModel


def _model(*, upper_bound=0.1, lower_bound=0.003):
    model = object.__new__(StageWiseMAPConstraintLearningModel)
    model.map_eq_sigma = upper_bound
    model.map_active_sigma_floor = lower_bound
    model.map_eq_distribution = "gaussian"
    return model


def test_eq_sigma_profiles_observed_dispersion():
    model = _model()

    sigma = model._eq_profile_scale(np.array([-0.02, 0.02]), eta=0.0)

    assert np.isclose(sigma, 0.02)


def test_eq_sigma_is_clipped_to_configured_bounds():
    model = _model()

    assert np.isclose(model._eq_profile_scale(np.zeros(3), eta=0.0), 0.003)
    assert np.isclose(model._eq_profile_scale(np.array([-1.0, 1.0]), eta=0.0), 0.1)


def test_removed_eq_sigma_mode_is_rejected():
    with pytest.raises(ValueError, match="remove map_eq_sigma_mode"):
        StageWiseMAPConstraintLearningModel(map_eq_sigma_mode="fixed")
