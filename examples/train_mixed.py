"""Runnable example: python examples/train_mixed.py

All samples are synthetic and intentionally use a temporally ordered holdout.
Do not treat training accuracy on this toy dataset as meaningful.
"""
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline

from MLSD import SData, SDataFrame
from MLSD.model_selection import split_frame


def main():
    rng = np.random.default_rng(42)
    n = 80
    signals = []
    news = []
    labels = []
    for i in range(n):
        direction = int(i % 3 != 0)
        signal = rng.normal(loc=direction * .6, size=12)
        signals.append(signal)
        news.append("optimistic outlook" if direction else "cautious outlook")
        labels.append(direction)

    frame = SDataFrame([
        SData(signals, column="signal", dtype="Series"),
        SData(news, column="news", dtype="Text"),
    ])
    train, y_train, test, y_test = split_frame(
        frame, np.asarray(labels), test_size=.25, shuffle=False)
    train_features = train.fit_transform()
    test_features = train.transform(test)
    # Estimation, imputation, and fitting happen exclusively on train rows.
    model = make_pipeline(SimpleImputer(strategy="median"),
                          LogisticRegression(max_iter=1000))
    model.fit(train_features.sparse.to_dense(), y_train)
    predictions = model.predict(test_features.sparse.to_dense())
    print(f"train={len(train)} test={len(test)}")
    print(f"held-out accuracy={accuracy_score(y_test, predictions):.3f}")


if __name__ == "__main__":
    main()
