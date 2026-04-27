import os

def save_model_info(config, avg_list, std_list, dir_name):
    os.makedirs(dir_name, exist_ok=True)

    names = config.get("models", config.get("model", None)) + [config.get("models", None)]*3
    types = ["text", "graphics", "late-fusion-or", "late-fusion-and", "late-fusion-avg"] if "models" in config else config["type"]
    vectorizers = (config.get("vectorizers", config.get("vectorizer", None)) if config["type"] != "early-fusion" else [config.get("vectorizers", None)]) + [config.get("vectorizers", None)]*3

    for i, (avg, std) in enumerate(zip(avg_list, std_list)):
        name = names[i]

        file_path = os.path.join(
            dir_name,
            f"{types[i]}_{name}_{vectorizers[i]}.txt"
        )

        output = []
        output.append(str(config))

        for k in avg:
            output.append(f"{k}: {avg[k]:.4f} ± {std[k]:.4f}")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(output))