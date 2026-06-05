import pandas as pd
from EDA import TextEDA
from label_transform import LabelTransform
from baselines.run_cv import run_cv
# from baselines.utils.save_model import save_model_info
# from baselines.grid_search import run_grid_search
# from baselines.freeze_models import freeze_model
# from baselines.utils.test_configs import freeze_configs
from params.models import get_model_name
from params.vectorizers import get_vectorizer_name
from params.threshold_grid import get_thresholds
from params.model_grid import get_model_grid

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 20)

def load():
    df = pd.read_csv("movies.csv")

    # EDA - podstawowe informacje, usunięcie null
    eda = TextEDA(df, True)
    # eda.display_dataset_basic_info()
    df = eda.drop_na()

    # Transformacja etykiet do wektorów 1 na 18
    lt = LabelTransform(df)
    return df, lt, eda

def info():
    _, lt, eda = load()

    y = lt.preprocessing()
    y_label = lt.y_labels
    y_count = lt.y_count
    # Podsumowanie informacji na temat zbioru
    eda.display_summary(y=y, y_labels=y_label, y_count=y_count)
    eda.chart_summary()
    eda.class_distribution()
    leak_df = eda.check_label_leakage(y, y_label)
    print(leak_df)

def main(type: str = "text", use_threshold_grid: bool = True, use_model_grid: bool = False) -> None:
    if type == "info":
        info()
        return

    df, lt, _ = load()
    y = lt.preprocessing()

    models = get_model_name[type]
    vectorizers = get_vectorizer_name[type]
    
    for vect1 in vectorizers[0]:
        for vect2 in vectorizers[1]:
            for mt1 in models[0]:
                for mt2 in models[1]:
                    thresholds_text = get_thresholds[type][vect1] if type != "graphics" else [None]
                    thresholds_graphics = get_thresholds[type][vect1] if type == "graphics" else get_thresholds[type][vect2] if type in ["early-fusion", "late-fusion"] else [None]
                    grid = [
                        get_model_grid.get(mt1, {}),
                        get_model_grid.get(mt2, {})
                    ] if use_model_grid else [{}, {}]

                    config = {
                        "type": type,
                        "vectorizers": [vect1] if vect2 is None else [vect1, vect2],
                        "models": [mt1] if mt2 is None else [mt1, mt2],
                        "balanced_list": [True, True],
                        "thresholds_text": thresholds_text,
                        "thresholds_graphics": thresholds_graphics,
                        "outer_cv": 10,
                        "inner_cv": 3,
                        "use_threshold_grid": use_threshold_grid,
                        "use_model_grid": use_model_grid,
                        "grid": grid,
                        "save_more_metadata": True,
                        "all_thresholds": True,
                        # "use_subset": True,
                        # "subset_size": 0.1,
                        # "plot_show": True, 
                    }

                    print(config)

                    results = run_cv(df, y, **config)

if __name__ == "__main__":
    # type = "text"    # "graphics", "late-fusion", "text", "early-fusion" or "info" # freeze - soon return
    # main(
    #     type = type,
    #     use_threshold_grid = True,
    #     use_model_grid = True
    # )

    type = "graphics"  # "graphics", "late-fusion", "text", "early-fusion" or "info" # freeze - soon return
    main(
        type=type,
        use_threshold_grid=True,
        use_model_grid=True
    )
