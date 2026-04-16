import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')

COLORS = {
    'primary': '#6366F1',
    'secondary': '#EC4899',
    'accent1': '#F59E0B',
    'accent2': '#10B981',
    'accent3': '#EF4444',
    'accent4': '#8B5CF6',
    'dark': '#1E1B4B',
    'light': '#F8FAFC',
}

EMOTION_COLORS = {
    'anger': '#EF4444',
    'sadness': '#3B82F6',
    'joy': '#F59E0B',
    'neutral': '#6B7280',
    'fear': '#8B5CF6',
    'disgust': '#10B981',
    'surprise': '#EC4899',
}

CLUSTER_COLORS = ['#6366F1', '#EC4899', '#F59E0B', '#10B981', '#8B5CF6']

def setup_style():
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': '#E2E8F0',
        'axes.linewidth': 1.5,
        'axes.grid': True,
        'grid.color': '#F1F5F9',
        'grid.alpha': 0.8,
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.titlepad': 15,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '#E2E8F0',
    })

def plot_elbow_method(K_range, inertia, save_path=None):
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(K_range, inertia, 
            color=COLORS['primary'], 
            linewidth=3,
            marker='o',
            markersize=10,
            markerfacecolor='white',
            markeredgecolor=COLORS['primary'],
            markeredgewidth=2)
    
    ax.fill_between(K_range, inertia, alpha=0.15, color=COLORS['primary'])
    
    ax.set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Inertia', fontsize=12, fontweight='bold')
    ax.set_title('Elbow Method for Optimal k Selection', fontsize=16, fontweight='bold', pad=20)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig

def plot_pca_clusters(df_pca, labels_col='cluster', save_path=None):
    setup_style()
    fig, ax = plt.subplots(figsize=(12, 8))
    
    unique_clusters = df_pca[labels_col].unique()
    
    for i, cluster in enumerate(sorted(unique_clusters)):
        mask = df_pca[labels_col] == cluster
        ax.scatter(df_pca.loc[mask, 'pca_1'],
                  df_pca.loc[mask, 'pca_2'],
                  c=CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
                  s=80,
                  alpha=0.7,
                  edgecolors='white',
                  linewidths=0.5,
                  label=f'Cluster {cluster}')
    
    ax.set_xlabel('Principal Component 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('Principal Component 2', fontsize=12, fontweight='bold')
    ax.set_title('Character Clusters\n(PCA Projection)', fontsize=16, fontweight='bold', pad=20)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper right', framealpha=0.95, fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig

def plot_emotional_arc(df_char, emotions, title, save_path=None, smooth=True, window=5):
    setup_style()
    fig, ax = plt.subplots(figsize=(14, 7))
    
    df_plot = df_char.copy()
    
    if smooth:
        for emo in emotions:
            df_plot[emo] = df_plot[emo].rolling(window=window, min_periods=1).mean()
    
    for emo in emotions:
        color = EMOTION_COLORS.get(emo, COLORS['primary'])
        ax.plot(df_plot['time_step'], df_plot[emo],
                color=color,
                linewidth=2.5,
                label=emo.capitalize(),
                alpha=0.9)
        ax.fill_between(df_plot['time_step'], df_plot[emo], 
                        alpha=0.15, color=color)
    
    ax.set_xlabel('Narrative Progression (Scenes)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Emotion Intensity', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper right', framealpha=0.95, fontsize=10, ncol=2)
    ax.set_ylim(0, None)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig

def plot_multiple_arcs(df_main_arcs, main_chars, emotions, title, save_path=None, window=5):
    setup_style()
    
    n_chars = len(main_chars)
    fig, axes = plt.subplots(n_chars, 1, figsize=(14, 5 * n_chars), sharex=True)
    
    if n_chars == 1:
        axes = [axes]
    
    for idx, (ax, char) in enumerate(zip(axes, main_chars)):
        df_char = df_main_arcs[df_main_arcs['character_name'] == char].copy()
        df_char = df_char.sort_values('time_step')
        
        for emo in emotions:
            color = EMOTION_COLORS.get(emo, COLORS['primary'])
            smoothed = df_char[emo].rolling(window=window, min_periods=1).mean()
            ax.plot(df_char['time_step'], smoothed,
                    color=color,
                    linewidth=2.5,
                    label=emo.capitalize(),
                    alpha=0.9)
            ax.fill_between(df_char['time_step'], smoothed, 
                           alpha=0.12, color=color)
        
        ax.set_ylabel('Emotion Intensity', fontsize=11, fontweight='bold')
        ax.set_title(f'{char}', fontsize=14, fontweight='bold', loc='left')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc='upper right', framealpha=0.95, fontsize=9, ncol=len(emotions))
        ax.set_ylim(0, None)
        
        if idx == 0:
            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    axes[-1].set_xlabel('Narrative Progression (Scenes)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig

def plot_emotion_comparison(df_char1, df_char2, emotions, labels, save_path=None, window=5):
    setup_style()
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    for ax, df_char, label in zip(axes, [df_char1, df_char2], labels):
        df_plot = df_char.copy()
        for emo in emotions:
            color = EMOTION_COLORS.get(emo, COLORS['primary'])
            df_plot[emo] = df_plot[emo].rolling(window=window, min_periods=1).mean()
            ax.plot(df_plot['time_step'], df_plot[emo],
                   color=color, linewidth=2, label=emo.capitalize(), alpha=0.85)
            ax.fill_between(df_plot['time_step'], df_plot[emo], 
                          alpha=0.1, color=color)
        
        ax.set_ylabel('Emotion Intensity', fontsize=11, fontweight='bold')
        ax.set_title(label, fontsize=14, fontweight='bold', loc='left', color=COLORS['primary'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc='upper right', framealpha=0.95, fontsize=9, ncol=len(emotions))
        ax.set_ylim(0, None)
    
    axes[-1].set_xlabel('Narrative Progression (Scenes)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig

def plot_cluster_heatmap(cluster_summary, save_path=None):
    setup_style()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    emotion_cols = ['anger', 'disgust', 'fear', 'sadness', 'neutral', 'surprise', 'joy']
    
    data = cluster_summary[emotion_cols]
    
    sns.heatmap(data, 
                annot=True, 
                fmt='.2f', 
                cmap='RdYlBu_r',
                center=0.15,
                linewidths=0.5,
                linecolor='white',
                cbar_kws={'label': 'Mean Intensity', 'shrink': 0.8},
                ax=ax)
    
    ax.set_xlabel('Emotion', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cluster', fontsize=12, fontweight='bold')
    ax.set_title('Emotion Profile by Cluster', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticklabels([e.capitalize() for e in emotion_cols], rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig
