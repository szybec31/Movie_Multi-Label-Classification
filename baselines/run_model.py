from .tf_idf import model as tfidf

def run_model(type, df, y, y_label, opt = ["text", 0.5, True]):
    df["text"] = df["title"].fillna('') + " " + df["overview"].fillna('')

    if type == "tfidf":
        tfidf(df, y, y_label, opt[0], opt[1], opt[2])
        # print("th: 0.3")
        # tfidf(df, y, y_label, opt, 0.3)
        # print("th: 0.2")
        # tfidf(df, y, y_label, opt, 0.2)