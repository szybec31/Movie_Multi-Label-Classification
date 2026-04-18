from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from sklearn.model_selection import train_test_split

def model(df, y, type, threshold = 0.5, balanced = True):
    
    X = df[type]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    tfidf = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1,2),
        stop_words='english'
    )

    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    model = OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight='balanced') if balanced else LogisticRegression(max_iter=1000))
    model.fit(X_train_tfidf, y_train)

    y_proba = model.predict_proba(X_test_tfidf)
    y_pred = (y_proba > threshold).astype(int)

    return y_test, y_pred