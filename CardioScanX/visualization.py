import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import io
from matplotlib import pyplot as plt
try:
    from src.visualization_cae import plot_full_multilead as plot_full_cae
except Exception:
    plot_full_cae = None

try:
    from src.visualization_vae import plot_full_multilead as plot_full_vae
except Exception:
    plot_full_vae = None


def plot_12_lead_plotly(orig, recon=None):
    # orig: [12, T]
    T = orig.shape[1]
    fig = make_12_lead_subplots()
    x = np.arange(T)
    for i in range(12):
        fig.add_trace(go.Scatter(x=x, y=orig[i], name=f'Lead {i+1}', mode='lines', line=dict(color='black')), row=i+1, col=1)
        if recon is not None:
            fig.add_trace(go.Scatter(x=x, y=recon[i], name=f'Recon {i+1}', mode='lines', line=dict(color='blue', dash='dash')), row=i+1, col=1)
    fig.update_layout(height=1600, showlegend=False,
                      plot_bgcolor='white', paper_bgcolor='white', font_color='black')
    return fig


def make_12_lead_subplots():
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=12, cols=1, shared_xaxes=True, vertical_spacing=0.01)
    # Ensure white background and dark text for readability
    fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', font_color='black')
    return fig


def plot_anomaly_timeline(mse):
    # mse: [T, 12]
    T = mse.shape[0]
    fig = go.Figure()
    x = np.arange(T)
    mean_score = mse.mean(axis=1)
    fig.add_trace(go.Scatter(x=x, y=mean_score, mode='lines', name='Anomaly Score'))
    fig.update_layout(height=300, plot_bgcolor='white', paper_bgcolor='white', font_color='black')
    return fig


def plot_heatmap(mse):
    # mse: [T,12]
    fig = px.imshow(mse.T, labels=dict(x='Time', y='Lead'), aspect='auto', color_continuous_scale='Viridis')
    fig.update_layout(height=400, plot_bgcolor='white', paper_bgcolor='white', font_color='black')
    return fig


def save_full_multilead_plt(model_type, model, orig, recon, std, out_dir, file_name='report'):
    # Use plotting utilities from original project (visualization_cae / visualization_vae) to save an image
    import os
    os.makedirs(out_dir, exist_ok=True)
    T = orig.shape[1]
    # call the right function: they expect [T, 12]
    x_orig = orig.T
    x_mean = recon.T
    x_std = std.T
    try:
        if 'CAE' in model_type:
            plot_full_cae(x_orig, x_mean, x_std, 0, out_dir, out_dir)
        else:
            plot_full_vae(x_orig, x_mean, x_std, 0, out_dir, out_dir)
    except Exception:
        # fallback to simple matplotlib
        fig, ax = plt.subplots(12, 1, sharex=True, figsize=(14, 12))
        # ensure white background and dark axes for visibility
        fig.patch.set_facecolor('white')
        try:
            for a in ax:
                a.set_facecolor('white')
                for spine in a.spines.values():
                    spine.set_color('black')
                a.tick_params(colors='black', which='both')
        except Exception:
            pass
        x = np.arange(T)
        for i in range(12):
            ax[i].plot(x, orig[i], color='black')
            ax[i].plot(x, recon[i], color='C1', linestyle='--')
        fig_file = os.path.join(out_dir, file_name + '.png')
        fig.savefig(fig_file, dpi=150, facecolor='white')
        plt.close(fig)
        return fig_file
    return os.path.join(out_dir, f'epoch_000_full_multilead.png')
