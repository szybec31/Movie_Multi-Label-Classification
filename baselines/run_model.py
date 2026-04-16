from .tf_idf import model as tfidf

def run_model(type, df, y, y_label, opt = "text"):
    df["text"] = df["title"].fillna('') + " " + df["overview"].fillna('')

    if type == "tfidf":
        tfidf(df, y, y_label, opt)