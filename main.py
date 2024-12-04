from scripts import (
    load_data,
    load_test_data,
    build_model,
    train_model,
    evaluate_model,
    plot_training_results,
    save_model,
    load_model,
    predict_image,
)

# Or import everything with *
from scripts import *

# Usage remains the same
data_dir = "dataset/train"
test_dir = "dataset/test"

train_ds, val_ds, class_names = load_data(data_dir)
test_ds = load_test_data(test_dir)
model = build_model(num_classes=len(class_names))
history = train_model(model, train_ds, val_ds)
test_loss, test_accuracy = evaluate_model(model, test_ds)
plot_training_results(history)
save_model(model)
