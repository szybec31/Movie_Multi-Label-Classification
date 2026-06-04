get_model_grid = {

    "logistic": {
        "estimator__C": [0.01, 0.1, 1.0, 10.0],
        "estimator__class_weight": [None, "balanced"]
    },

    "random_forest": {
        "estimator__n_estimators": [100, 200, 300],
        "estimator__max_depth": [3, 5, 10],
        # "estimator__max_features": ["sqrt", 0.8],
        # "estimator__min_samples_split": [2, 5],
        "estimator__class_weight": ["balanced"]
    },

    "mlp": {
        "hidden_layer_sizes": [
            (256,),
            (256, 128),
            (512, 256)
        ],
        "learning_rate_init": [
            0.001,
            0.0005
        ],
        "alpha": [
            1e-5,
            1e-4,
            1e-3
        ],
        "max_iter": [
            80
        ]
    }
}