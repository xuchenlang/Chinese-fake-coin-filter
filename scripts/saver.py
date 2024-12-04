def save_model(model, save_path="coin_classifier_model"):
    model.save(save_path)

def load_model(save_path="coin_classifier_model"):
    from tensorflow.keras.models import load_model
    return load_model(save_path)
