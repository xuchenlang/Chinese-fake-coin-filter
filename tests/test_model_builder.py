from scripts.model_builder import build_model

def test_build_model():
    model = build_model(num_classes=2)
    assert model is not None, "Model should build successfully"
    assert len(model.layers) > 0, "Model should have layers"
