# features/tfidf.py

from sklearn.feature_extraction.text import TfidfVectorizer

def build_tfidf(X_train, X_test, **config):
    vectorizer = TfidfVectorizer(
        max_features=config.get("max_features_tfidf", 20000),
        ngram_range=config.get("ngram_range_tfidf", (1,2)),
        stop_words='english'
    )

    X_train_vec = vectorizer.fit_transform(X_train).toarray()
    X_test_vec  = vectorizer.transform(X_test).toarray()

    print("TFIDF Ilość cech:", X_train_vec.shape[1])

    return X_train_vec, X_test_vec, vectorizer