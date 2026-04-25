def run_experiment(df, y, split=None, **config):
	# ========================
	# type: str = "text" or "graphics" or "early-fusion" or "late-fusion"
	# subtype: str = "text" or "title" or "overview" (only for type = "text")
	# vectorizer: str = "tfidf" or "distilbert" (for "text") or "resnet18" or "resnet50" (for "graphics")
	# vectorizers: list[str] = up to 2 vectorizers from list above (only for early or late fusion)
	# model: str = "logistic" or "svm" or "random_forest" or "mlp"
	# models: list = up to 2 models from list above (only for "late-fusion")
	# balanced: bool = True or False
	# balanced_list: list[bool] = list of using balanced params in models (only for "late-fusion")
	# threshold: float = 0.2, 0.3, 0.5 (only for tfidf vectorizer), base value = 0.5 for tfidf or None (for other vect)
	# thresholds: list[float] = list of thresholds for late-fusion when using min one model based on tfidf vectorizer
	# max_features_tfidf: int = base 20000,
	# ngram_range_tfidf: tuple = base (1,2),
	# n_estimators: int = base 200, for random_forest
    # max_depth: int = base 20, for random_forest
	# max_features_rf: str = base 'sqrt', for random_forest
	# hidden_layer_sizes: tuple = base (256, 128), for mlp
    # max_iter: int = base 20, for mlp
    # batch_size: int = base 64, for mlp
    # learning_rate_init: float = base 0.001, for mlp
	# ========================


	# ========================
	# VALIDATE all data
    # ========================
	if "type" not in config:
		raise ValueError("Unknown type")

	if config["type"] in ["text", "graphics"]:
		if "model" not in config or "vectorizer" not in config:
			raise ValueError("Set model and vectorizer param")

	elif config["type"] in ["early-fusion", "late-fusion"]:
		if "vectorizers" not in config:
			raise ValueError("Set vectorizers[text, graphics] param")
		if "models" not in config and config["type"] == "late-fusion":
			raise ValueError("Set models[text, graphics] param")
		if "model" not in config and "models" not in config and config["type"] == "early-fusion":
			raise ValueError("Set model or models[] param")
    
	if "models" not in config:
		config["models"] = [config["model"]]
		
	if "vectorizers" not in config:
		config["vectorizers"] = [config["vectorizer"]]
	
	if "thresholds" not in config:
		config["thresholds"] = []
		if "threshold" in config:
			config["thresholds"] = [config["threshold"]]
		else:
			config["thresholds"] = [None, None]
			
	if "balanced_list" not in config:
		if "balanced" in config:
			config["balanced_list"] = [config["balanced"]]
		else:
			config["balanced_list"] = [False, False]

	## LISTS:
	X_list = []
    
	if config["type"] in ["text", "early-fusion", "late-fusion"]:
		if config["vectorizers"][0] not in ["tfidf", "distilbert"]:
			raise ValueError("Wrong vectorizer to chosen type; choose tfidf or distilbert")

        # ========================
        # SUBTYPE (for text only)
        # ========================
		if "subtype" not in config:
			config["subtype"] = "text"
		
		if config["subtype"] == "text":
			X1 = df["title"].fillna('') + " " + df["overview"].fillna('')
		elif config["subtype"] in ["title", "overview"]:
			X1 = df[config["subtype"]]
			
		X_list.append(X1)
			
			
	if config["type"] == "graphics":
		if config["vectorizers"][0] not in ["resnet18", "resnet50"]:
			raise ValueError("Wrong vectorizer to chosen type; choose resnet18 or resnet50")
		X1 = df["poster_path"]
		X_list.append(X1)
		
	if config["type"] in ["early-fusion", "late-fusion"]:
		if config["vectorizers"][1] not in ["resnet18", "resnet50"]:
			raise ValueError("Wrong vectorizer to chosen type; choose resnet18 or resnet50")
		X2 = df["poster_path"]
		X_list.append(X2)

    # ========================
    # SPLIT
    # ========================
	if split is None:
		from sklearn.model_selection import train_test_split
		idx = np.arange(len(df))
		train_idx, test_idx = train_test_split(
			idx, test_size=0.2, random_state=42
		)
	else:
		train_idx, test_idx = split

	X_train_list = [X.iloc[train_idx] for X in X_list]
	X_test_list  = [X.iloc[test_idx] for X in X_list]

	y_train = y[train_idx]
	y_test  = y[test_idx]


    # ========================
    # FEATURES / VECTORIZER
    # ========================
	features_train = []
	features_test = []
	
	for i, vec in enumerate(config["vectorizers"]):

		if vec == "tfidf":
			from .features.tfidf import build_tfidf
			Xt, Xv, _ = build_tfidf(X_train_list[i], X_test_list[i], **config)

		elif vec == "distilbert":
			from .features.distilbert import build_distilbert_embedding
			Xt, Xv, = build_distilbert_embedding(X_list[i],split=split)

		elif vec == "resnet50":
			from .features.resnet50 import build_image_features
			Xt, Xv, _ = build_image_features(df, (train_idx, test_idx))

		elif vec == "resnet18":
			from .features.resnet18 import build_image_features
			Xt, Xv, _ = build_image_features(df, (train_idx, test_idx))

		else:
			raise ValueError("Unknown vectorizer")

		features_train.append(Xt)
		features_test.append(Xv)


	# ========================
    # FUSION
    # ========================

	# if early-fusion then X+X2 and one model, if late-fusion then two models
	
	if config["type"] == "early-fusion":
		import numpy as np
		X_train_final = np.hstack(features_train)
		X_test_final  = np.hstack(features_test)

		features_train = [X_train_final]
		features_test  = [X_test_final]

    # ========================
    # MODELS AND PREDICTIONS
    # ========================
	
	preds = []

	for i, model_name in enumerate(config["models"]):

		Xtr = features_train[i]
		Xte = features_test[i]

		if model_name == "logistic":
			from .models.logistic import train_logistic
			model = train_logistic(Xtr, y_train, config["balanced_list"][i])

			threshold = config["thresholds"][i] or 0.5
			y_proba = model.predict_proba(Xte)
			y_pred = (y_proba > threshold).astype(int)

		elif model_name == "svm":
			from .models.svm import train_svm
			model = train_svm(Xtr, y_train, config["balanced_list"][i])
			y_pred = model.predict(Xte)

		elif model_name == "random_forest":
			from .models.randomforest import train_random_forest
			from .utils.remake_config import clean_model_config
			print(f"Balanced: {config["balanced_list"][i]}")
			model = train_random_forest(Xtr, y_train, balanced=config["balanced_list"][i], **clean_model_config(config, ["balanced"]))
			y_pred = model.predict(Xte)

		elif model_name == "mlp":
			from .models.mlp import train_mlp
			model = train_mlp(Xtr, y_train, **config)
			y_pred = model.predict(Xte)

		else:
			raise ValueError("Unknown model")

		preds.append(y_pred)

    # ========================
    # LATE FUSION
    # ========================
	if config["type"] == "late-fusion":
        # np. OR / average
		y_pred = np.mean(preds, axis=0) > 0.5
		y_pred = y_pred.astype(int)

	else:
		y_pred = preds[0]

    # ========================
    # METRICS
    # ========================
	from .utils.metrics import evaluate
	return evaluate(y_test, y_pred)