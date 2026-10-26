import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_task_entrypoints_keep_demo_ids_zero_based():
    inspect_source = (
        PROJECT_ROOT / "experiments/inspect_bar_inspection_demo_starts.py"
    ).read_text()
    render_source = (PROJECT_ROOT / "experiments/render_s5_demonstrations.py").read_text()
    map_plot_source = (PROJECT_ROOT / "visualization/map_plots.py").read_text()

    assert "demo_index + 1" not in inspect_source
    assert '"demo_local_index"' not in render_source
    assert "set_yticklabels([str(idx + 1)" not in map_plot_source


def test_gui_recording_labels_default_to_demo_zero():
    gui_pages = [
        PROJECT_ROOT / "robot/stage_cons_iiwa14/host_gui/web/index.html",
        PROJECT_ROOT
        / "robot/stage_cons_iiwa14/ros_ws/src/stage_demo_gui/web/index.html",
    ]
    for gui_page in gui_pages:
        source = gui_page.read_text()
        assert 'value="demo_00"' in source


def test_bar_inspect_active_artifacts_use_demo_zero_names():
    processed_dir = PROJECT_ROOT / "robot/stage_cons_iiwa14/data/processed"
    expected = [
        processed_dir / "demo_00_10hz_coarse.npz",
        processed_dir / "demo_00_5hz_coarse.npz",
        processed_dir / "demo_00_coarse_segmentation.png",
        PROJECT_ROOT / "configs/data/BarInspect_demo_00_annotations.json",
        PROJECT_ROOT / "experiments/bar_inspection_demo00_analysis.ipynb",
    ]
    assert all(path.exists() for path in expected)

    scene_config = json.loads(
        (
            PROJECT_ROOT
            / "robot/stage_cons_iiwa14/ros_ws/src/stage_iiwa_sim/config/demo_scene.json"
        ).read_text()
    )
    assert scene_config["source"]["ik_seed_processed_demo"].endswith(
        "/demo_00_5hz_coarse.npz"
    )
    with np.load(expected[1], allow_pickle=True) as archive:
        assert str(archive["source_processed_file"]).endswith(
            "/demo_00_10hz_coarse.npz"
        )
