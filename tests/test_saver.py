from scripts.saver import save_model, load_model
from scripts.model_builder import build_model

def test_save_and_load_model():
    model = build_model(num_classes=2)
    save_model(model, save_path="test_model")
    loaded_model = load_model("test_model")
    assert loaded_model is not None, "Model should load successfully"
