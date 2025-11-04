from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import cross_validate, learning_curve
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def train_random_forest(X, y, cv, target_type='multiclass', n_estimators=100, max_depth=None,
<<<<<<< Updated upstream
<<<<<<< Updated upstream
                       min_samples_split=2, min_samples_leaf=1, random_state=42,
                       class_weight=None):
=======
                       min_samples_split=2, min_samples_leaf=1, random_state=42, print_cv = True):
>>>>>>> Stashed changes
=======
                       min_samples_split=2, min_samples_leaf=1, random_state=42, print_cv = True):
>>>>>>> Stashed changes
    """
    Train and evaluate Random Forest with cross-validation.

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

    Returns:
        dict: cross-validation results with metrics
    """

    target_type = target_type.lower()
    if target_type not in {'multiclass', 'binary', 'regression'}:
        raise ValueError("target_type must be one of 'multiclass', 'binary', 'regression'")

    # For regression, use RandomForestRegressor
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
    else:
        # For classification (binary or multiclass)
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,
            class_weight=class_weight
        )

        if target_type == 'multiclass':
            scoring = ['accuracy', 'precision_micro', 'recall_micro', 'f1_micro']
        else:  # binary
            scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    if print_cv:
        print(f"\nTraining Random Forest...")
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
    params_str = f"n_estimators={n_estimators}, max_depth={max_depth}, min_samples_leaf={min_samples_leaf}"
    if print_cv:
        print_cv_results(scores, "Random Forest", target_type, params_str)

    return scores


def train_lasso(X, y, cv, target_type='multiclass', alpha=1.0, max_iter=5000, random_state=42, print_cv = True):
    """
    Train and evaluate Lasso with cross-validation.

    Args:
        X: feature DataFrame or array
        y: target array/Series
        cv: cross-validator object (e.g. StratifiedKFold)
        target_type: one of 'multiclass', 'binary', 'regression'
        alpha: regularization strength (larger = stronger regularization)
        max_iter: maximum iterations for solver
        random_state: random seed

    Returns:
        dict: cross-validation results with metrics
    """

    target_type = target_type.lower()
    if target_type not in {'multiclass', 'binary', 'regression'}:
        raise ValueError("target_type must be one of 'multiclass', 'binary', 'regression'")

    # For regression, use Lasso regression
    if target_type == 'regression':
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('lasso', Lasso(alpha=alpha, random_state=random_state, max_iter=max_iter))
        ])
        scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
    else:
        # For classification (binary or multiclass), use LogisticRegression with L1 penalty
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('lasso', LogisticRegression(
                penalty='l1',
                solver='saga',
                C=1.0/alpha,  # Convert alpha to C (inverse relationship)
                max_iter=max_iter,
                random_state=random_state,
                n_jobs=-1
            ))
        ])

        if target_type == 'multiclass':
            scoring = ['accuracy', 'precision_micro', 'recall_micro', 'f1_micro']
        else:  # binary
            scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    if print_cv:
        print(f"\n📐 Training Lasso...")
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
    params_str = f"alpha={alpha}, max_iter={max_iter}"
    if print_cv:
        print_cv_results(scores, "Lasso", target_type, params_str)

    return scores


def train_logistic_regression(X, y, cv, target_type='multiclass', C=1.0, penalty='l2',
                              solver='lbfgs', max_iter=1000, random_state=42, print_cv = True):
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

    if print_cv:
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

    if print_cv:
        print_cv_results(scores, model_name, target_type, params_str)

    return scores


def evaluate_dummy_baseline(X, y, cv, target_type='multiclass', target_variable=None, random_state=0, print_cv =  True):
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

    if print_cv:
        print(f"\nEvaluating Dummy Baseline...")
        print(f"Cross-validation folds: {cv.get_n_splits()}")

    scores = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=False)

    # Print results
    model_name = f"Dummy Baseline ({target_variable})" if target_variable else "Dummy Baseline"
    strategy = 'most_frequent' if target_type in {'multiclass', 'binary'} else 'mean'
    params_str = f"strategy={strategy}"

    if print_cv:
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
