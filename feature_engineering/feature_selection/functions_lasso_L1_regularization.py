from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def run_classification_lasso(data, predictor_columns, target_column, output_path, is_binary=False, k=5, savefile=True):
    """
    Run Lasso (L1) regularized logistic regression with k-fold cross-validation for classification tasks.

    Trains a logistic regression model with L1 penalty using stratified k-fold cross-validation,
    evaluates performance, and identifies important features based on absolute coefficients.
    Generates confusion matrix, classification report, and feature importance visualizations.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset containing both features and target variable.
    predictor_columns : list of str
        Column names to use as predictor features for the model.
    target_column : str
        Column name of the target variable to predict.
    output_path : pathlib.Path or str
        Directory path where feature importance CSV file will be saved.
    is_binary : bool, optional
        Whether the classification task is binary (True) or multi-class (False).
        Affects how coefficients are aggregated for feature importance. Default is False.
    k : int, optional
        Number of folds for cross-validation. Default is 5.
    savefile : bool, optional
        Whether to save feature importance results to a CSV file. Default is True.

    Returns
    -------
    pd.DataFrame
        DataFrame containing feature names and their importance scores (absolute coefficients),
        sorted in descending order of importance.
    """

    # Prepare data
    X = data[predictor_columns]
    y = data[target_column]

    # Drop NaN if needed
    if y.isna().any():
        valid_indices = y.notna()
        X = X[valid_indices]
        y = y[valid_indices]

    print(f"Training Lasso (L1) for: {target_column}")
    print(f"Total data size: {X.shape[0]}")
    print(f"Using {k}-fold cross-validation")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Setup model
    lasso = LogisticRegression(penalty='l1', solver='saga', C=0.1, random_state=42,
                               max_iter=5000, n_jobs=-1)

    # K-fold cross-validation
    kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    # Get cross-validation scores
    cv_scores = cross_val_score(lasso, X_scaled, y, cv=kfold, scoring='accuracy', n_jobs=-1)
    print(f"\nCross-validation accuracy scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Get predictions for confusion matrix and classification report
    y_pred = cross_val_predict(lasso, X_scaled, y, cv=kfold, n_jobs=-1)

    print(f"\nClassification Report:")
    print(classification_report(y, y_pred, zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    fig_size = (6, 5) if is_binary else (10, 8)
    plt.figure(figsize=fig_size)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {target_column} (Lasso, {k}-fold CV)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

    # Train on full dataset for feature importance
    lasso.fit(X_scaled, y)

    # Feature importance
    if is_binary:
        coef_abs = np.abs(lasso.coef_[0])
    else:
        coef_abs = np.abs(lasso.coef_).sum(axis=0)

    selected_features = np.array(predictor_columns)[coef_abs > 0].tolist()
    print(f"\nNumber of features selected: {len(selected_features)} out of {len(predictor_columns)}")

    feature_importance = pd.DataFrame({
        'feature': predictor_columns,
        'importance': coef_abs
    }).sort_values('importance', ascending=False)
    feature_importance['importance'] = feature_importance['importance'].round(4)

    print(f"\nTop 20 features for {target_column}:")
    print(feature_importance.head(20))

    # Save feature importance to CSV
    if savefile==True:
        feature_selection_path = output_path / 'feature_selection_csv'
        feature_selection_path.mkdir(parents=True, exist_ok=True)
        output_file = feature_selection_path / f'lasso_feature_importance_{target_column}.csv'
        feature_importance.to_csv(output_file, index=False)
        print(f"\nFeature importance saved to: {output_file}")

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    top_20 = feature_importance.head(20)
    sns.barplot(data=top_20, x='importance', y='feature', hue='feature', legend=False, palette='viridis')
    plt.title(f'Top 20 Feature Importances - {target_column} (Lasso L1, {k}-fold CV)')
    plt.xlabel('Absolute Coefficient Sum' if not is_binary else 'Absolute Coefficient')
    plt.tight_layout()
    plt.show()

    return feature_importance


def run_regression_lasso(data, predictor_columns, target_column, output_path, k=5, savefile=True):
    """
    Run Lasso (L1) regularized regression with k-fold cross-validation for continuous target prediction.

    Trains a Lasso regression model using k-fold cross-validation, evaluates performance with
    regression metrics (R², RMSE, MAE), and identifies important features based on absolute coefficients.
    Generates actual vs predicted scatter plot and feature importance visualizations.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset containing both features and target variable.
    predictor_columns : list of str
        Column names to use as predictor features for the model.
    target_column : str
        Column name of the continuous target variable to predict.
    output_path : pathlib.Path or str
        Directory path where feature importance CSV file will be saved.
    k : int, optional
        Number of folds for cross-validation. Default is 5.
    savefile : bool, optional
        Whether to save feature importance results to a CSV file. Default is True.

    Returns
    -------
    pd.DataFrame
        DataFrame containing feature names and their importance scores (absolute coefficients),
        sorted in descending order of importance.
    """

    # Prepare data
    X = data[predictor_columns]
    y = data[target_column]

    # Drop NaN
    if y.isna().any():
        valid_indices = y.notna()
        X = X[valid_indices]
        y = y[valid_indices]

    print(f"Training Lasso (L1) for: {target_column}")
    print(f"Total data size: {X.shape[0]}")
    print(f"Using {k}-fold cross-validation")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Setup model
    lasso = Lasso(alpha=0.1, random_state=42, max_iter=5000)

    # K-fold cross-validation
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)

    # Get cross-validation scores (R² score)
    cv_scores = cross_val_score(lasso, X_scaled, y, cv=kfold, scoring='r2', n_jobs=-1)
    print(f"\nCross-validation R² scores: {cv_scores}")
    print(f"Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Get cross-validation scores for negative MSE
    cv_mse_scores = -cross_val_score(lasso, X_scaled, y, cv=kfold, scoring='neg_mean_squared_error', n_jobs=-1)
    cv_rmse = np.sqrt(cv_mse_scores.mean())
    print(f"Mean CV RMSE: {cv_rmse:.4f}")

    # Get predictions for scatter plot
    y_pred = cross_val_predict(lasso, X_scaled, y, cv=kfold, n_jobs=-1)

    # Calculate metrics on predictions
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print(f"\nOverall metrics on cross-validated predictions:")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")

    # Scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y, y_pred, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel(f'Actual {target_column}')
    plt.ylabel(f'Predicted {target_column}')
    plt.title(f'Actual vs Predicted - {target_column} (Lasso, {k}-fold CV)')
    plt.tight_layout()
    plt.show()

    # Train on full dataset for feature importance
    lasso.fit(X_scaled, y)

    # Feature importance
    coef_abs = np.abs(lasso.coef_)
    selected_features = np.array(predictor_columns)[coef_abs > 0].tolist()

    print(f"\nNumber of features selected: {len(selected_features)} out of {len(predictor_columns)}")

    feature_importance = pd.DataFrame({
        'feature': predictor_columns,
        'importance': coef_abs
    }).sort_values('importance', ascending=False)
    feature_importance['importance'] = feature_importance['importance'].round(4)

    print(f"\nTop 20 features for {target_column}:")
    print(feature_importance.head(20))

    # Save feature importance to CSV
    if savefile==True:
        feature_selection_path = output_path / 'feature_selection_csv'
        feature_selection_path.mkdir(parents=True, exist_ok=True)
        output_file = feature_selection_path / f'lasso_feature_importance_{target_column}.csv'
        feature_importance.to_csv(output_file, index=False)
        print(f"\nFeature importance saved to: {output_file}")

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    top_20 = feature_importance.head(20)
    sns.barplot(data=top_20, x='importance', y='feature', hue='feature', legend=False, palette='viridis')
    plt.title(f'Top 20 Feature Importances - {target_column} (Lasso L1, {k}-fold CV)')
    plt.xlabel('Absolute Coefficient')
    plt.tight_layout()
    plt.show()

    return feature_importance
