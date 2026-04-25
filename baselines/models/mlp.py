from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier

def train_mlp(X_train, y_train, balanced=True):
    model = OneVsRestClassifier(
        MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation='relu',
            solver='adam',
            max_iter=20,
            batch_size=64,
            learning_rate_init=0.001,
            early_stopping=True,
            random_state=42
        )
    )

    model.fit(X_train, y_train)
    return model