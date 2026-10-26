import methods

from evaluation.metrics import _compute_constraint_set_iou


def test_constraint_set_iou_counts_semantic_matches_over_union():
    assert _compute_constraint_set_iou(3, 4, 5) == 0.5


def test_constraint_set_iou_handles_empty_sets():
    assert _compute_constraint_set_iou(0, 0, 0) == 1.0
    assert _compute_constraint_set_iou(0, 0, 2) == 0.0
