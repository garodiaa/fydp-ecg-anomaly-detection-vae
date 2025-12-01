"""
Generate comparison visualizations across all models
"""
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import pandas as pd

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Define models to compare
MODELS = {
    'ba_vae': 'Beat-Aligned VAE',
    'cae': 'Convolutional AE',
    'hlvae': 'Hierarchical Latent VAE',
    'st_vae': 'ST-VAE',
    'vae_bilstm_attn': 'VAE BiLSTM Attn',
    'vae_bilstm_mha': 'VAE BiLSTM MHA',
    'vae_gru': 'VAE GRU'
}

def load_all_models():
    """Load comprehensive JSON for all models"""
    data = {}
    for model_id in MODELS.keys():
        json_path = Path(f'outputs/{model_id}/{model_id}_comprehensive.json')
        with open(json_path, 'r') as f:
            data[model_id] = json.load(f)
    return data

def plot_metrics_comparison(all_data):
    """Compare key metrics across all models"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    model_names = [MODELS[mid] for mid in all_data.keys()]
    
    # 1. F1-Score Comparison
    ax1 = axes[0, 0]
    f1_scores = [all_data[mid]['evaluation']['f1_score'] for mid in all_data.keys()]
    colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
    bars = ax1.barh(model_names, f1_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax1.set_xlim([0, 1])
    ax1.set_xlabel('F1-Score', fontsize=12, fontweight='bold')
    ax1.set_title('F1-Score Comparison', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars, f1_scores)):
        ax1.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=10, fontweight='bold')
    
    # 2. AUC-ROC Comparison
    ax2 = axes[0, 1]
    auc_roc = [all_data[mid]['evaluation']['auc_roc'] for mid in all_data.keys()]
    bars = ax2.barh(model_names, auc_roc, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax2.set_xlim([0, 1])
    ax2.set_xlabel('AUC-ROC', fontsize=12, fontweight='bold')
    ax2.set_title('AUC-ROC Comparison', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars, auc_roc)):
        ax2.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=10, fontweight='bold')
    
    # 3. Precision vs Recall
    ax3 = axes[1, 0]
    precisions = [all_data[mid]['evaluation']['precision'] for mid in all_data.keys()]
    recalls = [all_data[mid]['evaluation']['recall'] for mid in all_data.keys()]
    
    for i, (mid, name) in enumerate(MODELS.items()):
        ax3.scatter(recalls[i], precisions[i], s=300, alpha=0.6, 
                   color=colors[i], edgecolor='black', linewidth=2, label=name)
        ax3.annotate(name, (recalls[i], precisions[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax3.set_xlim([0.7, 1.0])
    ax3.set_ylim([0.6, 1.0])
    ax3.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax3.set_title('Precision vs Recall', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.plot([0.7, 1.0], [0.7, 1.0], 'r--', alpha=0.3, label='Perfect Balance')
    
    # 4. Inference Time
    ax4 = axes[1, 1]
    inference_times = [all_data[mid]['latency']['inference_time_ms_per_sample'] 
                      for mid in all_data.keys()]
    bars = ax4.barh(model_names, inference_times, color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=2)
    ax4.set_xlabel('Inference Time (ms/sample)', fontsize=12, fontweight='bold')
    ax4.set_title('Inference Time Comparison', fontsize=14, fontweight='bold')
    ax4.grid(axis='x', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars, inference_times)):
        ax4.text(val + 0.05, i, f'{val:.2f}', va='center', fontsize=10, fontweight='bold')
    
    plt.suptitle('Model Performance Comparison', fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_path = Path('outputs/visualizations/model_comparison_metrics.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print('✓ Saved model comparison metrics')

def plot_training_convergence(all_data):
    """Compare training convergence across models"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(MODELS)))
    
    # Training loss convergence
    ax1 = axes[0]
    for i, (mid, name) in enumerate(MODELS.items()):
        training = all_data[mid]['training']
        epochs = list(range(1, len(training['train_loss']) + 1))
        ax1.plot(epochs, training['train_loss'], label=name, linewidth=2, 
                color=colors[i], alpha=0.7)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Training Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training Loss Convergence', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Validation loss convergence
    ax2 = axes[1]
    for i, (mid, name) in enumerate(MODELS.items()):
        training = all_data[mid]['training']
        epochs = list(range(1, len(training['val_loss']) + 1))
        ax2.plot(epochs, training['val_loss'], label=name, linewidth=2, 
                color=colors[i], alpha=0.7)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Validation Loss Convergence', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.suptitle('Training Convergence Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = Path('outputs/visualizations/training_convergence_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print('✓ Saved training convergence comparison')

def plot_architecture_comparison(all_data):
    """Compare model architectures"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    model_names = [MODELS[mid] for mid in all_data.keys()]
    colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
    
    # Parameter count
    ax1 = axes[0]
    params = [all_data[mid]['architecture_info']['total_parameters'] / 1e6 
             for mid in all_data.keys()]
    bars = ax1.barh(model_names, params, color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=2)
    ax1.set_xlabel('Parameters (Millions)', fontsize=12, fontweight='bold')
    ax1.set_title('Model Size (Parameter Count)', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars, params)):
        ax1.text(val + 0.1, i, f'{val:.2f}M', va='center', fontsize=10, fontweight='bold')
    
    # Efficiency (F1-Score per Million Parameters)
    ax2 = axes[1]
    f1_scores = [all_data[mid]['evaluation']['f1_score'] for mid in all_data.keys()]
    efficiency = [f1 / (param if param > 0 else 1) for f1, param in zip(f1_scores, params)]
    bars = ax2.barh(model_names, efficiency, color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=2)
    ax2.set_xlabel('F1-Score per Million Parameters', fontsize=12, fontweight='bold')
    ax2.set_title('Model Efficiency', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars, efficiency)):
        ax2.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=10, fontweight='bold')
    
    plt.suptitle('Architecture Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = Path('outputs/visualizations/architecture_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print('✓ Saved architecture comparison')

def plot_comprehensive_radar(all_data):
    """Create radar chart for comprehensive model comparison"""
    from math import pi
    
    # Metrics to compare (normalized to 0-1)
    metrics = ['F1-Score', 'AUC-ROC', 'Precision', 'Recall', 'Speed\n(inverse)']
    
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
    
    angles = [n / float(len(metrics)) * 2 * pi for n in range(len(metrics))]
    angles += angles[:1]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(MODELS)))
    
    for i, (mid, name) in enumerate(MODELS.items()):
        eval_data = all_data[mid]['evaluation']
        latency = all_data[mid]['latency']['inference_time_ms_per_sample']
        
        # Normalize speed (inverse of latency, scaled)
        max_latency = max([all_data[m]['latency']['inference_time_ms_per_sample'] 
                          for m in all_data.keys()])
        speed_score = 1 - (latency / max_latency) if max_latency > 0 else 1
        
        values = [
            eval_data['f1_score'],
            eval_data['auc_roc'],
            eval_data['precision'],
            eval_data['recall'],
            speed_score
        ]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=name, color=colors[i], alpha=0.7)
        ax.fill(angles, values, alpha=0.15, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    plt.title('Comprehensive Model Comparison\n(Radar Chart)', 
             fontsize=16, fontweight='bold', pad=20)
    
    output_path = Path('outputs/visualizations/radar_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print('✓ Saved radar chart comparison')

def create_summary_table(all_data):
    """Create a summary table of all models"""
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data
    rows = []
    for mid, name in MODELS.items():
        data = all_data[mid]
        eval_data = data['evaluation']
        arch_info = data['architecture_info']
        latency = data['latency']
        training = data['training']
        
        rows.append([
            name,
            f"{eval_data['f1_score']:.3f}",
            f"{eval_data['precision']:.3f}",
            f"{eval_data['recall']:.3f}",
            f"{eval_data['auc_roc']:.3f}",
            f"{eval_data['auc_pr']:.3f}",
            f"{arch_info['total_parameters']/1e6:.2f}M",
            f"{latency['inference_time_ms_per_sample']:.2f}",
            f"{training['best_epoch']}/{training['epochs']}"
        ])
    
    columns = ['Model', 'F1-Score', 'Precision', 'Recall', 'AUC-ROC', 
              'AUC-PR', 'Parameters', 'Latency\n(ms)', 'Best/Total\nEpochs']
    
    table = ax.table(cellText=rows, colLabels=columns, cellLoc='center', loc='center',
                    colWidths=[0.20, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(len(columns)):
        cell = table[(0, i)]
        cell.set_facecolor('#4CAF50')
        cell.set_text_props(weight='bold', color='white', fontsize=11)
    
    # Style rows with alternating colors
    colors_alt = ['#f0f0f0', '#ffffff']
    for i in range(1, len(rows) + 1):
        for j in range(len(columns)):
            cell = table[(i, j)]
            cell.set_facecolor(colors_alt[i % 2])
            cell.set_text_props(fontsize=10)
    
    # Highlight best values
    # F1-Score (column 1)
    f1_scores = [float(row[1]) for row in rows]
    best_f1_idx = f1_scores.index(max(f1_scores))
    table[(best_f1_idx + 1, 1)].set_facecolor('#90EE90')
    table[(best_f1_idx + 1, 1)].set_text_props(weight='bold')
    
    # AUC-ROC (column 4)
    auc_scores = [float(row[4]) for row in rows]
    best_auc_idx = auc_scores.index(max(auc_scores))
    table[(best_auc_idx + 1, 4)].set_facecolor('#90EE90')
    table[(best_auc_idx + 1, 4)].set_text_props(weight='bold')
    
    plt.title('Model Performance Summary Table', fontsize=18, fontweight='bold', pad=20)
    
    output_path = Path('outputs/visualizations/summary_table.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print('✓ Saved summary table')

def plot_confusion_matrix_comparison(all_data):
    """Create a grid of normalized confusion matrices"""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, (mid, name) in enumerate(MODELS.items()):
        cm = all_data[mid]['confusion_matrix']
        
        # Normalize confusion matrix
        confusion = np.array([
            [cm['TN'], cm['FP']],
            [cm['FN'], cm['TP']]
        ])
        
        # Normalize by row (true labels)
        confusion_norm = confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis]
        
        ax = axes[idx]
        sns.heatmap(confusion_norm, annot=True, fmt='.2%', cmap='Blues', 
                   square=True, linewidths=2, linecolor='black',
                   xticklabels=['Normal', 'Abnormal'],
                   yticklabels=['Normal', 'Abnormal'],
                   cbar=True, ax=ax, vmin=0, vmax=1)
        
        ax.set_title(f'{name}\nF1: {all_data[mid]["evaluation"]["f1_score"]:.3f}', 
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=10, fontweight='bold')
        ax.set_ylabel('True', fontsize=10, fontweight='bold')
    
    # Remove extra subplot
    fig.delaxes(axes[7])
    
    plt.suptitle('Normalized Confusion Matrices (All Models)', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = Path('outputs/visualizations/confusion_matrices_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print('✓ Saved confusion matrices comparison')

def main():
    """Generate all comparison visualizations"""
    print('=' * 60)
    print('GENERATING MODEL COMPARISON VISUALIZATIONS')
    print('=' * 60)
    
    # Load all model data
    print('\nLoading model data...')
    all_data = load_all_models()
    print(f'✓ Loaded data for {len(all_data)} models\n')
    
    # Generate comparisons
    print('Generating comparison visualizations...')
    plot_metrics_comparison(all_data)
    plot_training_convergence(all_data)
    plot_architecture_comparison(all_data)
    plot_comprehensive_radar(all_data)
    create_summary_table(all_data)
    plot_confusion_matrix_comparison(all_data)
    
    print('\n' + '=' * 60)
    print('ALL COMPARISON VISUALIZATIONS GENERATED!')
    print('=' * 60)
    print('\nGenerated comparison files:')
    print('  - model_comparison_metrics.png')
    print('  - training_convergence_comparison.png')
    print('  - architecture_comparison.png')
    print('  - radar_comparison.png')
    print('  - summary_table.png')
    print('  - confusion_matrices_comparison.png')
    print(f'\nAll saved to: outputs/visualizations/')

if __name__ == '__main__':
    main()
