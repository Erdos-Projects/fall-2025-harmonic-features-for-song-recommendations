import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_feature_importance_df(feature_importance_df: pd.DataFrame, top_n: int | None = None,
                               style='line', figsize=(10, 8)):
    """
    Plot feature importance from a DataFrame.

    Args:
        feature_importance_df: DataFrame with columns 'feature' and 'importance'
        top_n: number of top features to plot (None = all)
        style: 'line' for line plot or 'bar' for horizontal bar chart
        figsize: figure size

    Returns:
        matplotlib figure
    """
    if not isinstance(feature_importance_df, pd.DataFrame):
        raise TypeError("feature_importance_df must be a pandas DataFrame")
    if 'feature' not in feature_importance_df.columns or 'importance' not in feature_importance_df.columns:
        raise ValueError("DataFrame must contain 'feature' and 'importance' columns")

    df = feature_importance_df.copy()
    df['feature'] = df['feature'].astype(str)
    df['importance'] = pd.to_numeric(df['importance'], errors='coerce').fillna(0)

    df = df.sort_values('importance', ascending=False).reset_index(drop=True)
    if top_n is not None:
        if not isinstance(top_n, int) or top_n <= 0:
            raise ValueError("top_n must be a positive integer or None")
        df = df.head(top_n).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=figsize)

    if style == 'line':
        x = np.arange(len(df))
        ax.plot(x, df['importance'].values, marker='o', linestyle='-', linewidth=2, markersize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(df['feature'].values, rotation=45, ha='right')
        ax.set_xlabel('Feature', fontsize=12)
        ax.set_ylabel('Importance', fontsize=12)
    else:  # bar chart
        bars = ax.barh(range(len(df)), df['importance'])
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df['feature'])
        ax.invert_yaxis()
        ax.set_xlabel('Importance', fontsize=12)

        # Color bars by importance
        norm = plt.Normalize(vmin=df['importance'].min(), vmax=df['importance'].max())
        colors = plt.cm.viridis(norm(df['importance']))
        for bar, color in zip(bars, colors):
            bar.set_color(color)

    title = f"Top {len(df)} Feature Importances" if top_n is not None else "Feature Importances"
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig
