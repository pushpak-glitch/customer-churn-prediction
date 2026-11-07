"""End-to-end training script: python src/train.py
Compares Logistic Regression vs Gradient Boosting (5-fold CV + hold-out), saves metrics and the best model."""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, classification_report, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from churn import RANDOM_STATE, ROOT, build_models, clean, engineer, load_raw, split_xy


def main() -> None:
    df = engineer(clean(load_raw()))
    X, y = split_xy(df)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    rows, fitted = [], {}
    for name, pipe in build_models(X).items():
        cv_auc = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc")
        pipe.fit(X_tr, y_tr)
        p = pipe.predict_proba(X_te)[:, 1]
        pred = (p >= 0.5).astype(int)
        rows.append({"model": name, "cv_roc_auc_mean": cv_auc.mean(), "cv_roc_auc_std": cv_auc.std(),
                     "test_roc_auc": roc_auc_score(y_te, p), "test_pr_auc": average_precision_score(y_te, p),
                     "precision@0.5": precision_score(y_te, pred), "recall@0.5": recall_score(y_te, pred),
                     "f1@0.5": f1_score(y_te, pred)})
        fitted[name] = pipe
        print(f"{name}: CV AUC {cv_auc.mean():.3f}±{cv_auc.std():.3f} | test AUC {rows[-1]['test_roc_auc']:.3f}")
        print(classification_report(y_te, pred, digits=3))
    res = pd.DataFrame(rows).sort_values("cv_roc_auc_mean", ascending=False)
    (ROOT / "outputs").mkdir(exist_ok=True); (ROOT / "models").mkdir(exist_ok=True)
    res.round(4).to_csv(ROOT / "outputs" / "model_comparison_train_script.csv", index=False)
    best = res.iloc[0]["model"]
    joblib.dump(fitted[best], ROOT / "models" / "churn_model.joblib")
    print(f"Best by CV ROC-AUC: {best}")


if __name__ == "__main__":
    main()
