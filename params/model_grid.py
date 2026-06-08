get_model_grid = {

    "logistic": {
        "estimator__C": [0.01, 0.1, 1.0],
        "estimator__class_weight": [None, "balanced"]
    },

    "random_forest": {
        "estimator__n_estimators": [100, 200, 300],
        "estimator__max_depth": [5, 10],
        "estimator__max_features": ["sqrt"],
        "estimator__class_weight": ["balanced"]
    },

    "mlp": {
        "hidden_layer_sizes": [
            (256, 128),
            (512, 256)
        ],
        "learning_rate_init": [
            0.001,
            0.0005
        ],
        "alpha": [1e-5, 1e-4, 1e-3],
        "max_iter": [40],
        "activation": ['relu'],
        "solver": ['adam'],
        "batch_size": [64],
        "early_stopping": [True],
    }
}