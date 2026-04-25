# features/tfidf.py

from sklearn.feature_extraction.text import TfidfVectorizer

def build_tfidf(X_train, X_test, max_features=20000, ngram_range=(1,2)):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words='english'
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print("TFIDF Ilość cech: ",X_train_vec.shape[1])

    return X_train_vec, X_test_vec, vectorizer