get_model_grid = {
    "logistic": {
    },

    "random_forest": {
        "balanced": [True, False],
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "max_features_rf": ["sqrt", 0.8]
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
        "batch_size": [32, 64],
        "max_iter": [40, 80]
    }
}