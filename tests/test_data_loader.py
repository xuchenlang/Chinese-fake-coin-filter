from scripts.data_loader import load_data, load_test_data

def test_load_data():
    data_dir = "dataset/train"
    train_ds, val_ds, class_names = load_data(data_dir)
    assert len(class_names) > 0, "Class names should not be empty"
    assert train_ds is not None, "Training dataset should load"
    assert val_ds is not None, "Validation dataset should load"

def test_load_test_data():
    test_dir = "dataset/test"
    test_ds = load_test_data(test_dir)
    assert test_ds is not None, "Test dataset should load"
