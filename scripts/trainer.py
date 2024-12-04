def train_model(model, train_ds, val_ds, epochs=10):
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )
    return history

def evaluate_model(model, test_ds):
    test_loss, test_accuracy = model.evaluate(test_ds)
    return test_loss, test_accuracy
