import os

def save_model_info(config, avg, std, dir_name):
    os.makedirs(dir_name, exist_ok=True)
    file_path = os.path.join(dir_name, f"{config.get('model', config.get('models', [None])[0])}_{config.get('type', None)}_{config.get('vectorizer', config.get("vectorizers", [None])[0])}_{config.get('balanced', None)}.txt")

    output = []
    output.append(str(config))
    for k in avg:
        output.append(f"{k}: {avg[k]:.4f} ± {std[k]:.4f}")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output))