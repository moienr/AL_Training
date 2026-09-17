"""The two classifiers used in the notebooks.

LinearProbe  the model behind the simulation figures in the EarthQuery report:
             one linear layer on standardised embeddings, trained with heavy
             input dropout (the ALFM "linear_classifier" with dropout 0.75).
make_rf      the random forest that the EarthQuery tool trains inside Earth
             Engine (100 trees, 21 of 64 features per split, half the rows
             per tree), here as a scikit-learn model for local use.

Both expose fit(X, y) and predict_proba(X) with two columns
[P(class 0), P(class 1)], so the notebooks can swap them freely.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class LinearProbe:
    """Linear classifier on standardised features with input dropout.

    Training follows the ALFM setup used for the report: standardise the
    features with the mean and standard deviation of the labelled points
    (BatchNorm without affine parameters), drop 75% of the input features
    at random in every step, cross-entropy loss, AdamW with learning rate
    0.01 and weight decay 0.01, 100 full-batch epochs.
    """

    def __init__(self, dropout=0.75, lr=1e-2, weight_decay=1e-2, epochs=100, seed=0):
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.seed = seed

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        self.classes_ = np.array([0, 1])
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-5
        Xs = (X - self.mean_) / self.std_
        Y = np.eye(2)[y]                       # one-hot targets

        W = np.zeros((d, 2))
        b = np.zeros(2)
        mW, vW, mb, vb = np.zeros_like(W), np.zeros_like(W), np.zeros_like(b), np.zeros_like(b)
        b1, b2, eps = 0.9, 0.999, 1e-8
        keep = 1.0 - self.dropout
        for t in range(1, self.epochs + 1):
            mask = rng.random(Xs.shape) < keep      # input dropout
            Xd = Xs * mask / keep
            P = _softmax(Xd @ W + b)
            G = (P - Y) / n                          # gradient of the mean cross-entropy
            gW, gb = Xd.T @ G, G.sum(axis=0)
            # AdamW
            mW, vW = b1 * mW + (1 - b1) * gW, b2 * vW + (1 - b2) * gW ** 2
            mb, vb = b1 * mb + (1 - b1) * gb, b2 * vb + (1 - b2) * gb ** 2
            cW, cb = mW / (1 - b1 ** t), mb / (1 - b1 ** t)
            dW, db = vW / (1 - b2 ** t), vb / (1 - b2 ** t)
            W -= self.lr * (cW / (np.sqrt(dW) + eps) + self.weight_decay * W)
            b -= self.lr * cb / (np.sqrt(db) + eps)
        self.W_, self.b_ = W, b
        return self

    def predict_proba(self, X):
        Xs = (np.asarray(X, dtype=np.float64) - self.mean_) / self.std_
        return _softmax(Xs @ self.W_ + self.b_)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


def make_rf(seed=0, n_train=None):
    """The EarthQuery random forest (Earth Engine's smileRandomForest settings).

    Each tree sees half of the rows; with fewer than 20 labelled rows every
    tree sees all of them, which avoids trees built on one or two points.
    """
    half = 0.5 if (n_train is None or n_train >= 20) else None
    return RandomForestClassifier(n_estimators=100, max_features=21, max_samples=half,
                                  random_state=seed, n_jobs=-1)


def make_model(name="probe", seed=0, n_train=None):
    """'probe' for the linear probe, 'rf' for the random forest."""
    if name == "probe":
        return LinearProbe(seed=seed)
    if name == "rf":
        return make_rf(seed=seed, n_train=n_train)
    raise ValueError(f"unknown model {name!r}; use 'probe' or 'rf'")
