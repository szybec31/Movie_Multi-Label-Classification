get_model_grid = {

    "logistic": {
        "estimator__C": [0.01, 0.1, 1.0, 10.0],
        "estimator__class_weight": [None, "balanced"]
    },

    "random_forest": {
        "estimator__n_estimators": [100, 200, 500],
        "estimator__max_depth": [3, 5, 10, None],
        "estimator__max_features": ["sqrt", 0.8],
        "estimator__min_samples_split": [2, 5],
        "estimator__min_samples_leaf": [1, 2],
        "estimator__class_weight": [None, "balanced"]
    },

    "mlp": {
        "estimator__hidden_layer_sizes": [
            (256,),
            (256, 128),
            (512, 256)
        ],
        "estimator__learning_rate_init": [
            0.001,
            0.0005
        ],
        "estimator__batch_size": [
            32,
            64
        ],
        "estimator__alpha": [
            1e-5,
            1e-4,
            1e-3
        ],
        "estimator__max_iter": [
            40,
            80
        ]
    }
}