from sklearn.neural_network import MLPClassifier


def create_fnn(random_state: int = 42) -> MLPClassifier:
    # FNN = feed-forward neural network via sklearn MLPClassifier
    return MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=50,
        early_stopping=True,
        n_iter_no_change=5,
        random_state=random_state,
        verbose=False,
    )


def param_grid() -> dict:
    # Keep it small/fast; expand later if needed
    return {
        "model__hidden_layer_sizes": [(128, 64), (256, 128), (128, 64, 32)],
        "model__alpha": [1e-5, 1e-4, 1e-3],
        "model__learning_rate_init": [1e-4, 1e-3],
    }


