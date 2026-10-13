from scripts.project_v3_labels import project


def test_conflicting_projection_preserves_original_prediction():
    rows = list(project([{"example_id": "x", "prediction": "CONFLICTING"}],
                        {"CONFLICTING": "NOT ENOUGH INFO"}))
    assert rows[0]["prediction"] == "NOT ENOUGH INFO"
    assert rows[0]["metadata"]["pre_projection_prediction"] == "CONFLICTING"
