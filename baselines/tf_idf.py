from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, classification_report, recall_score
from sklearn.metrics import hamming_loss

from sklearn.model_selection import train_test_split

def model(df, y, y_label, type, threshold = 0.5, balanced = True):
    
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

    print("F1 micro:", f1_score(y_test, y_pred, average='micro'))
    print("F1 macro:", f1_score(y_test, y_pred, average='macro'))
    print("Recall weighted:", recall_score(y_test, y_pred, average='weighted'))

    print("Avg labels per sample (true):", y_test.sum(axis=1).mean())
    print("Avg labels per sample (pred):", y_pred.sum(axis=1).mean())
    print("Hamming Loss:", hamming_loss(y_test, y_pred))

    #print(classification_report(y_test, y_pred, target_names=y_label, zero_division=0))