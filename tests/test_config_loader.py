from experiments.config_loader import inherit_map_posthoc_parameters


def test_posthoc_balanced_pooled_inherits_matching_method_and_dataset_config():
    method_cfg = {
        "posthoc_constraint": {
            "posthoc_training_mode": "map_balanced_pooled",
            "plot_dir": "baseline-plots",
        }
    }
    map_method_cfgs = {
        "map": {"map_eq_sigma": 0.05, "map_c_bg": 5.0},
        "map_pooled": {"map_eq_sigma": 0.07, "map_c_bg": 4.0},
        "map_balanced_pooled": {"map_eq_sigma": 0.1, "map_c_bg": 1.0},
    }
    dataset_overrides = {
        "map": {"map_c_ineq": 5.0},
        "map_pooled": {"map_c_ineq": 2.0},
        "map_balanced_pooled": {"map_c_ineq": 0.25},
    }

    result = inherit_map_posthoc_parameters(
        "gmmhmm",
        method_cfg,
        map_method_cfgs,
        dataset_overrides,
    )

    posthoc = result["posthoc_constraint"]
    assert posthoc["posthoc_training_mode"] == "map_balanced_pooled"
    assert posthoc["map_eq_sigma"] == 0.1
    assert posthoc["map_c_bg"] == 1.0
    assert posthoc["map_c_ineq"] == 0.25
    assert posthoc["plot_dir"] == "baseline-plots"


def test_posthoc_pooled_inherits_map_pooled_config():
    result = inherit_map_posthoc_parameters(
        "arhsmm",
        {"posthoc_constraint": {"posthoc_training_mode": "pooled"}},
        {
            "map": {"map_c_ineq": 1.0},
            "map_pooled": {"map_c_ineq": 2.0},
            "map_balanced_pooled": {"map_c_ineq": 3.0},
        },
        {},
    )

    assert result["posthoc_constraint"]["map_c_ineq"] == 2.0
