"""Baseline models — no graph structure, just node features."""

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score


# ---------------------------------------------------------------------------
# Logistic Regression
# ---------------------------------------------------------------------------

class LogisticRegressionBaseline:
    """Balanced logistic regression on raw features."""

    def __init__(self, max_iter=1000, C=1.0):
        self.model = LogisticRegression(
            class_weight="balanced", max_iter=max_iter, C=C, solver="lbfgs"
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def assess(self, X, y):
        """Compute AUC, AP, and F1 on a given split."""
        proba = self.predict_proba(X)
        preds = self.predict(X)
        return {
            "auc": roc_auc_score(y, proba),
            "ap": average_precision_score(y, proba),
            "f1": f1_score(y, preds),
        }


# ---------------------------------------------------------------------------
# SVM (RBF kernel)
# ---------------------------------------------------------------------------

class SVMBaseline:
    """SVM with RBF kernel on raw features."""

    def __init__(self, C=1.0):
        self.model = SVC(
            kernel="rbf", C=C, class_weight="balanced",
            probability=True, random_state=42,
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def assess(self, X, y):
        proba = self.predict_proba(X)
        preds = self.predict(X)
        return {
            "auc": roc_auc_score(y, proba),
            "ap": average_precision_score(y, proba),
            "f1": f1_score(y, preds),
        }


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

class RandomForestBaseline:
    """Random Forest classifier (200 trees, balanced weights)."""

    def __init__(self, n_estimators=200):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def assess(self, X, y):
        proba = self.predict_proba(X)
        preds = self.predict(X)
        return {
            "auc": roc_auc_score(y, proba),
            "ap": average_precision_score(y, proba),
            "f1": f1_score(y, preds),
        }


# ---------------------------------------------------------------------------
# Gradient Boosting
# ---------------------------------------------------------------------------

class GradientBoostingBaseline:
    """Gradient Boosting classifier (200 estimators)."""

    def __init__(self, n_estimators=200):
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators, random_state=42,
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def assess(self, X, y):
        proba = self.predict_proba(X)
        preds = self.predict(X)
        return {
            "auc": roc_auc_score(y, proba),
            "ap": average_precision_score(y, proba),
            "f1": f1_score(y, preds),
        }


# ---------------------------------------------------------------------------
# MLP Baseline
# ---------------------------------------------------------------------------

class MLPBaseline(nn.Module):
    """Simple feed-forward network on tabular features."""

    def __init__(self, input_dim=110, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def predict_proba(self, x):
        self.train(False)
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_mlp(model, X_train, y_train, X_val, y_val, lr=1e-3,
              epochs=100, batch_size=512, patience=10, device="cpu"):
    """Standard training loop for the MLP baseline."""
    model = model.to(device)

    # handle class imbalance
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    pos_weight = torch.tensor([neg_count / pos_count], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)

    best_auc = 0.0
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        # mini-batch training
        perm = torch.randperm(len(X_train_t))
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(X_train_t), batch_size):
            idx = perm[i:i + batch_size]
            logits = model(X_train_t[idx])
            loss = criterion(logits, y_train_t[idx])

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            epoch_loss += loss.item()
            n_batches += 1

        # validation
        model.train(False)
        with torch.no_grad():
            val_proba = model.predict_proba(X_val_t).cpu().numpy()
        val_auc = roc_auc_score(y_val, val_proba)

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    return {"best_val_auc": best_auc, "epochs_trained": epoch + 1}
