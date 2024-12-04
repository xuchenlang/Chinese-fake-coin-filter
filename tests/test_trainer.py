from scripts.trainer import train_model, evaluate_model
from scripts.data_loader import load_data

def test_train_and_evaluate():
    data_dir = "dataset/train"
    train_ds, val_ds, _ = load_data(data_dir)
    from scripts.model_builder import build_model
    model = build_model(num_classes=2)
    history = train_model(model, train_ds, val_ds, epochs=1)
    assert 'accuracy' in history.history, "Training should include accuracy metric"
