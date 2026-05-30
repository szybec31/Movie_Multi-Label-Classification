def freeze_configs(type):
    if type == "text":
        vectorizer = ["distilbert"]
        configs = [
            {
                "type": type,
                "vectorizers": vectorizer,
                "models": ["logistic"],
                "balanced": False,
                "threshold": 0.3
            },
            {
                "type": type,
                "vectorizers": vectorizer,
                "models": ["svm"],
                "balanced": True,
                "max_iter_svm": 5000
            },
            {
                "type": type,
                "vectorizers": vectorizer,
                "models": ["random_forest"],
                "balanced": True,
                "n_estimators": 200,
                "max_depth": 5,
                "max_features_rf": "sqrt"
            },
            {
                "type": type,
                "vectorizers": vectorizer,
                "models": ["mlp"],
                "hidden_layer_sizes": (256, 128),
                "learning_rate_init": 0.001,
                "batch_size": 64,
                "max_iter": 80
            }
        ]
    elif type == "graphics":
        vectorizer = ['resnet50']
        configs = [
            {
                "type": type,
                "vectorizers": vectorizer,
                "models": ["logistic"],
                "balanced": True,
                "threshold": 0.2
            },
            {
                "type": type,
                "vectorizers": vectorizer,
                "models": ["random_forest"],
                "balanced": True,
                "n_estimators": 200,
                "max_depth": 3,
                "max_features_rf": "sqrt"
            },
            {
                "type": type,
                "vectorizers": vectorizer,
                "models": ["mlp"],
                "hidden_layer_sizes": (512, 256),
                "learning_rate_init": 0.001,
                "batch_size": 32,
                "max_iter": 40
            }
        ]

    elif type == "early-fusion":
        vectorizer = ['distilbert', 'resnet50']
        configs = [
            {
                "type": type,
                "vectorizers": vectorizer,
                "models": ["logistic"],
                "balanced": False,
                "threshold": 0.3
            },
            {
                "type": type,
                "vectorizers": vectorizer,
                "models": ["random_forest"],
                "balanced": True,
                "n_estimators": 200,
                "max_depth": 5,
                "max_features_rf": "sqrt"
            },
            {
                "type": type,
                "vectorizers": vectorizer,
                "models": ["mlp"],
                "hidden_layer_sizes": (512, 256),
                "learning_rate_init": 0.001,
                "batch_size": 32,
                "max_iter": 40
            }
        ]

    else:
        exit()

    return configs
