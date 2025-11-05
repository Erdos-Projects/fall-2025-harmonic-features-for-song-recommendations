from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import cross_validate, learning_curve
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.tree import plot_tree

def train_logistic_regression(X, y, cv, target_type='multiclass', C=1.0, penalty='l2',
                              solver='lbfgs', max_iter=1000, random_state=42):
    """
    Train and evaluate logistic regression with cross-validation.

    Args:
        X: feature DataFrame or array
        y: target array/Series
        cv: cross-validator object (e.g. StratifiedKFold)
        target_type: one of 'multiclass', 'binary', 'regression'
        C: inverse of regularization strength (smaller = stronger regularization)
        penalty: regularization type ('l1', 'l2', 'elasticnet', 'none')
        solver: optimization algorithm ('lbfgs', 'saga', 'liblinear', etc.)
        max_iter: maximum iterations for solver
        random_state: random seed

    Returns:
        dict: cross-validation results with metrics
    """

    target_type = target_type.lower()
    if target_type not in {'multiclass', 'binary', 'regression'}:
        raise ValueError("target_type must be one of 'multiclass', 'binary', 'regression'")

    # For regression, use Ridge instead of LogisticRegression
    if target_type == 'regression':
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=1.0 / C, random_state=random_state, max_iter=max_iter))
        ])
        scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
    else:
        # For classification (binary or multiclass)
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('logistic', LogisticRegression(
                C=C,
                penalty=penalty,
                solver=solver,
                max_iter=max_iter,
                random_state=random_state,
                n_jobs=-1
            ))
        ])

        if target_type == 'multiclass':
            scoring = ['accuracy', 'precision_micro', 'recall_micro', 'f1_micro']
        else:  # binary
            scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    print(f"\nTraining Logistic Regression/Ridge...")
    print(f"Cross-validation folds: {cv.get_n_splits()}")

    # Perform cross-validation
    scores = cross_validate(
        model, X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )

    # Print results
    model_name = "Ridge Regression" if target_type == 'regression' else "Logistic Regression"
    params_str = f"C={C}, penalty={penalty}, solver={solver}" if target_type != 'regression' else f"alpha={1.0/C}"
    print_cv_results(scores, model_name, target_type, params_str)

    return scores




def train_lasso(X, y, cv, target_type='multiclass', alpha=1.0, max_iter=5000,
                random_state=42, plot_confusion_matrix=False):
    """
    Train and evaluate Lasso (or L1-regularized LogisticRegression) with cross-validation,
    optionally plotting confusion matrices.

    Args:
        X: feature DataFrame or array
        y: target array/Series
        cv: cross-validator object (e.g. StratifiedKFold)
        target_type: one of 'multiclass', 'binary', 'regression'
        alpha: regularization strength (larger = stronger regularization)
        max_iter: maximum iterations for solver
        random_state: random seed
        plot_confusion_matrix: if True, plots the average confusion matrix for classification

    Returns:
        dict: cross-validation results with metrics
    """

    target_type = target_type.lower()
    if target_type not in {'multiclass', 'binary', 'regression'}:
        raise ValueError("target_type must be one of 'multiclass', 'binary', 'regression'")

    # --- Model definition ---
    if target_type == 'regression':
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('lasso', Lasso(alpha=alpha, random_state=random_state, max_iter=max_iter))
        ])
        scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']

    else:
        # Classification: L1-regularized LogisticRegression
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('lasso', LogisticRegression(
                penalty='l1',
                solver='saga',
                C=1.0/alpha,
                max_iter=max_iter,
                random_state=random_state,
                n_jobs=-1
            ))
        ])
        if target_type == 'multiclass':
            scoring = ['accuracy', 'precision_micro', 'recall_micro', 'f1_micro']
        else:  # binary
            scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    print(f"\n📐 Training Lasso...")
    print(f"Cross-validation folds: {cv.get_n_splits()}")

    # --- Confusion matrix plotting for classification ---
    if plot_confusion_matrix and target_type != 'regression':
        fold_cm = []
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
            X_train, y_train = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx], \
                               y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
            X_test, y_test = X.iloc[test_idx] if hasattr(X, 'iloc') else X[test_idx], \
                             y.iloc[test_idx] if hasattr(y, 'iloc') else y[test_idx]

            fold_model = clone(model)
            fold_model.fit(X_train, y_train)
            y_pred = fold_model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            fold_cm.append(cm)

        avg_cm = np.mean(np.array(fold_cm), axis=0)

        plt.figure(figsize=(6,5))
        sns.heatmap(avg_cm, annot=True, fmt=".1f", cmap="Blues", cbar=True)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Average Confusion Matrix Across Folds")
        plt.show()

    # --- Cross-validation metrics ---
    scores = cross_validate(
        model, X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )

    # --- Print results ---
    params_str = f"alpha={alpha}, max_iter={max_iter}"
    print_cv_results(scores, "Lasso", target_type, params_str)

    return scores


def train_random_forest(
    X, y, cv, target_type='multiclass', n_estimators=100, max_depth=None,
    min_samples_split=2, min_samples_leaf=1, random_state=42,
    plot_confusion_matrix=False, plot_tree_flag=False
):
    """
    Train and evaluate Random Forest with cross-validation, optionally plotting confusion matrices
    and visualizing a single tree from the forest.

    Args:
        X: feature DataFrame or array
        y: target array/Series
        cv: cross-validator object (e.g. StratifiedKFold)
        target_type: one of 'multiclass', 'binary', 'regression'
        n_estimators: number of trees in the forest
        max_depth: maximum depth of trees (None = unlimited)
        min_samples_split: minimum samples required to split an internal node
        min_samples_leaf: minimum samples required to be at a leaf node
        random_state: random seed
        plot_confusion_matrix: if True, plots the average confusion matrix
        plot_tree_flag: if True, plots one tree from the fitted forest

    Returns:
        dict: cross-validation results with metrics
    """

    target_type = target_type.lower()
    if target_type not in {'multiclass', 'binary', 'regression'}:
        raise ValueError("target_type must be one of 'multiclass', 'binary', 'regression'")

    # Regression
    if target_type == 'regression':
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1
        )
        scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']

        scores = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=True, n_jobs=-1)
        params_str = f"n_estimators={n_estimators}, max_depth={max_depth}, min_samples_leaf={min_samples_leaf}"
        print_cv_results(scores, "Random Forest", target_type, params_str)
        return scores

    # Classification
    else:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1
        )

        if target_type == 'multiclass':
            scoring = ['accuracy', 'precision_micro', 'recall_micro', 'f1_micro']
        else:  # binary
            scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

        print(f"\nTraining Random Forest...")
        print(f"Cross-validation folds: {cv.get_n_splits()}")

        # Plot confusion matrices if requested
        if plot_confusion_matrix:
            fold_cm = []
            for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
                X_train, y_train = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx], \
                                   y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
                X_test, y_test = X.iloc[test_idx] if hasattr(X, 'iloc') else X[test_idx], \
                                 y.iloc[test_idx] if hasattr(y, 'iloc') else y[test_idx]
                fold_model = clone(model)
                fold_model.fit(X_train, y_train)
                y_pred = fold_model.predict(X_test)
                cm = confusion_matrix(y_test, y_pred)
                fold_cm.append(cm)

            avg_cm = np.mean(np.array(fold_cm), axis=0)
            plt.figure(figsize=(6,5))
            sns.heatmap(avg_cm, annot=True, fmt=".1f", cmap="Blues", cbar=True)
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.title("Average Confusion Matrix Across Folds")
            plt.show()

        # Compute cross-validated metrics
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=True, n_jobs=-1)
        params_str = f"n_estimators={n_estimators}, max_depth={max_depth}, min_samples_leaf={min_samples_leaf}"
        print_cv_results(scores, "Random Forest", target_type, params_str)

        # --- Plot one tree from the forest ---
        if plot_tree_flag:
            # Fit on full dataset
            model.fit(X, y)
            plt.figure(figsize=(20,10))
            plot_tree(
                model.estimators_[0],
                feature_names=X.columns if hasattr(X, 'columns') else None,
                class_names=[str(c) for c in model.classes_],
                filled=True,
                rounded=True,
                fontsize=10
            )
            plt.title("Single Tree from Random Forest")
            plt.show()

        return scores

def evaluate_dummy_baseline(X, y, cv, target_type='multiclass', target_variable=None, random_state=0):
    """
    Run cross-validated dummy baseline.
    Args:
      X: feature DataFrame or array
      y: target array/Series
      cv: cross-validator object (e.g. StratifiedKFold)
      target_type: one of 'multiclass', 'binary', 'regression'
      target_variable: name of target variable (optional, for display)
      random_state: seed for dummy classifier
    Returns:
      dict: result from sklearn.model_selection.cross_validate
    """
    target_type = target_type.lower()
    if target_type not in {'multiclass', 'binary', 'regression'}:
        raise ValueError("target_type must be one of 'multiclass', 'binary', 'regression'")

    if target_type in {'multiclass', 'binary'}:
        model = DummyClassifier(strategy='most_frequent', random_state=random_state)
        scoring = ['accuracy', 'precision_micro'] if target_type == 'multiclass' else ['accuracy', 'precision']
    else:  # regression
        model = DummyRegressor(strategy='mean')
        scoring = ['neg_mean_squared_error', 'r2']

    print(f"\nEvaluating Dummy Baseline...")
    print(f"Cross-validation folds: {cv.get_n_splits()}")

    scores = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=False)

    # Print results
    model_name = f"Dummy Baseline ({target_variable})" if target_variable else "Dummy Baseline"
    strategy = 'most_frequent' if target_type in {'multiclass', 'binary'} else 'mean'
    params_str = f"strategy={strategy}"
    print_cv_results(scores, model_name, target_type, params_str)

    return scores

def print_cv_results(scores, model_name, target_type, params_str=None):
    """
    Print cross-validation results.

    Args:
        scores: dict from cross_validate
        model_name: name of the model (e.g., 'Random Forest', 'Lasso')
        target_type: 'multiclass', 'binary', or 'regression'
        params_str: optional string describing key parameters
    """
    print(f"\n{'=' * 60}")
    print(f"{model_name} - {target_type.capitalize()} Target")
    if params_str:
        print(f"Parameters: {params_str}")
    print(f"{'=' * 60}")

    # Test Performance
    print("\nTest Performance (Out-of-Sample):")
    print("-" * 60)
    for metric, values in sorted(scores.items()):
        if metric.startswith('test_'):
            metric_name = metric.replace('test_', '').replace('_', ' ').title()
            mean_score = values.mean()
            std_score = values.std()
            print(f"{metric_name:30s}: {mean_score:7.4f} (+/- {std_score:.4f})")

    # Train Performance (if available)
    has_train = any(k.startswith('train_') for k in scores.keys())
    if has_train:
        print("\nTrain Performance (In-Sample):")
        print("-" * 60)
        for metric, values in sorted(scores.items()):
            if metric.startswith('train_'):
                metric_name = metric.replace('train_', '').replace('_', ' ').title()
                mean_score = values.mean()
                std_score = values.std()
                print(f"{metric_name:30s}: {mean_score:7.4f} (+/- {std_score:.4f})")

    print(f"\n{'=' * 60}\n")
