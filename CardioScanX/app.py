import streamlit as st
from pathlib import Path
import pandas as pd
import plotly.express as px
import numpy as np
import zipfile
import tempfile
import os
import io
from model_loader import MODEL_REGISTRY, load_model_by_name
from data_io import load_ecg_file
from preprocessing import apply_preprocessing
from legacy_visualization import compute_reconstruction_and_anomaly
from compute_st_vae_anomaly import compute_st_vae_reconstruction_and_anomaly, normalize_score
from visualization import plot_12_lead_plotly, plot_anomaly_timeline, plot_heatmap, save_full_multilead_plt
from report import generate_pdf_report
from scipy.signal import find_peaks

st.set_page_config(page_title='CardioScanX – Mass ECG Pre-Screening Tool', layout='wide')

# Material design-like color accent
ACCENT = '#1A73E8'
TAB_OPTIONS = ['Dashboard','Detailed Analysis','Reports']

st.title('CardioScanX – Mass ECG Pre-Screening Tool')

# Fixed model selection
SELECTED_MODEL = 'ST-VAE (Spatiotemporal VAE)'

# ------- Sidebar
with st.sidebar:
    st.markdown('### Upload ECG files')
    uploaded = st.file_uploader('Drop files here or select', accept_multiple_files=True, type=['npy','dat','hea','zip'])
    st.markdown('---')

    st.markdown('### Preprocessing options')
    bp = st.checkbox('Bandpass filter', value=True)
    nf = st.checkbox('Notch filter (50Hz)', value=True)
    norm = st.checkbox('Normalize per-lead', value=True)

    st.markdown('---')
    st.info('Using ST-VAE model with normalized scoring')
    st.caption('Normalized Score Range: Normal (0.06-0.29, mean: 0.23), Abnormal (0.80-0.94, mean: 0.84), Threshold: 0.69 → 100% Accuracy')
    run = st.button('Run Screening')

# Initialize app state
if 'results' not in st.session_state:
    st.session_state['results'] = []
if 'selected_idx' not in st.session_state:
    st.session_state['selected_idx'] = 0
if 'active_tab' not in st.session_state:
    st.session_state['active_tab'] = TAB_OPTIONS[0]

# ------- App logic
if run:
    if uploaded is None or len(uploaded) == 0:
        st.warning('Please upload at least one ECG (.npy, .dat/.hea, or zip).')
    else:
        # Use the fixed model and threshold
        selected_model = SELECTED_MODEL
        # Threshold based on normalized score analysis:
        # Normal ECGs: 0.3-0.5 (mean: 0.47)
        # Abnormal ECGs: 0.7-1.0 (mean: 0.92)
        # Optimal threshold: 0.69 (100% accuracy on demo samples)
        thr = 0.69
        
        if 'model' not in st.session_state or st.session_state.get('model_name') != selected_model:
            with st.spinner(f"Loading {selected_model}..."):
                model, md = load_model_by_name(selected_model, map_location='cpu')
                st.session_state['model'] = model
                st.session_state['model_name'] = selected_model

        model = st.session_state['model']

        all_files = []
        file_mapping = {}  # Map temp paths to original names
        # Handle ZIPs and regular files
        for f in uploaded:
            fname = f.name
            if fname.endswith('.zip'):
                with tempfile.TemporaryDirectory() as td:
                    zfile = zipfile.ZipFile(f)
                    for fn in zfile.namelist():
                        if Path(fn).suffix.lower() in ['.npy','.dat','.hea']:
                            zfile.extract(fn, td)
                            extracted_path = Path(td) / fn
                            all_files.append(extracted_path)
                            file_mapping[str(extracted_path)] = Path(fn).name
            else:
                # Save the uploaded file to tmp
                temp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(fname).suffix)
                temp.write(f.read())
                temp.flush()
                temp_path = Path(temp.name)
                all_files.append(temp_path)
                file_mapping[str(temp_path)] = fname

        progress = st.progress(0)
        results = []
        for i, fp in enumerate(all_files):
            try:
                sig = load_ecg_file(str(fp))
            except Exception as e:
                original_name = file_mapping.get(str(fp), fp.name)
                st.warning(f'Could not load {original_name}: {e}')
                continue
            sigp = apply_preprocessing(sig, bandpass=bp, notch=nf, normalize_sig=norm)
            # Ensure length 5000 like v/caes
            if sigp.shape[1] < 5000:
                sigp = np.pad(sigp, ((0,0),(0, 5000 - sigp.shape[1])), mode='constant')
            else:
                sigp = sigp[:, :5000]

            # Use ST-VAE anomaly detection
            recon, std, mse, raw_score = compute_st_vae_reconstruction_and_anomaly(model, sigp, window_size=1000, stride=500)
            
            # Normalize score to [0, 1] range
            score = normalize_score(raw_score)

            # compute BPM using lead II
            try:
                sig_lead = sigp[1]
                peaks, _ = find_peaks(sig_lead, distance=0.4*500)
                if len(peaks) >= 2:
                    bpm = 60.0 / np.mean(np.diff(peaks) / 500.0)
                else:
                    bpm = float('nan')
            except Exception:
                bpm = float('nan')

            decision = 'Abnormal' if score > thr else 'Normal'
            qual = 'Good' if np.nanmean(mse) < 0.2 else 'Low'
            
            # Use original filename from mapping
            original_name = file_mapping.get(str(fp), fp.name)
            
            results.append({
                'file': original_name,
                'path': str(fp),
                'bpm': float(bpm),
                'score': float(score),
                'quality': qual,
                'decision': decision,
                'mse': mse,
                'recon': recon,
                'std': std,
                'orig': sigp
            })
            progress.progress(int((i+1)/len(all_files)*100))

        st.session_state['results'] = results
        st.success('Batch screening finished')

# ------- Navigation & views
st.markdown('---')
active_tab = st.radio('Navigate', TAB_OPTIONS, horizontal=True, key='active_tab')
results = st.session_state['results']

if active_tab == 'Dashboard':
    st.header('Batch Summary')
    n_files = len(results)
    if n_files == 0:
        st.info('No results yet. Upload files and click Run Screening.')
    else:
        abn = sum(1 for r in results if r['decision']=='Abnormal')
        norm_c = n_files - abn
        mean_score = np.nanmean([r['score'] for r in results])
        mean_bpm = np.nanmean([r['bpm'] for r in results])

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric('Files', n_files)
        c2.metric('Abnormal', abn)
        c3.metric('Normal', norm_c)
        c4.metric('Mean Anom Score', round(mean_score, 4))
        c5.metric('Mean BPM', round(mean_bpm, 1))

        df = pd.DataFrame([{ 'file': r['file'], 'bpm': r['bpm'], 'score': r['score'], 'quality': r['quality'], 'decision': r['decision']} for r in results])
        st.plotly_chart(px.bar(df, x='file', y='score', color='decision', color_discrete_map={'Normal': ACCENT, 'Abnormal': 'red'}), use_container_width=True)
        st.plotly_chart(px.pie(df, names='decision', color='decision', color_discrete_map={'Normal': ACCENT, 'Abnormal': 'red'}), use_container_width=True)

        st.write('### Results table')
        st.dataframe(df[['file','bpm','score','quality','decision']])

elif active_tab == 'Detailed Analysis':
    st.header('Detailed Analysis')
    if len(results) == 0:
        st.info('No results yet. Upload files and click Run Screening.')
    else:
        if st.session_state['selected_idx'] >= len(results):
            st.session_state['selected_idx'] = 0
        options = [(i, r['file']) for i, r in enumerate(results)]
        sel = st.selectbox(
            'Inspect file',
            options=options,
            format_func=lambda x: x[1],
            key='inspect_details',
            index=st.session_state['selected_idx']
        )
        st.session_state['selected_idx'] = sel[0]
        idx = st.session_state['selected_idx']
        r = results[idx]
        st.subheader(r['file'])
        col1, col2, col3, col4 = st.columns(4)
        col1.metric('BPM', round(r['bpm'],1))
        col2.metric('Anomaly Score', round(r['score'],3))
        col3.metric('Decision', r['decision'])
        col4.metric('Quality', r['quality'])

        st.write('#### 12-lead waveform')
        fig = plot_12_lead_plotly(r['orig'], r['recon'])
        st.plotly_chart(fig, use_container_width=True)

else:  # Reports
    st.header('Reports')
    if len(results) == 0:
        st.info('No results yet. Upload files and click Run Screening.')
    else:
        if st.session_state['selected_idx'] >= len(results):
            st.session_state['selected_idx'] = 0
        options = [(i, r['file']) for i, r in enumerate(results)]
        sel = st.selectbox(
            'Inspect file',
            options=options,
            format_func=lambda x: x[1],
            key='inspect_reports',
            index=st.session_state['selected_idx']
        )
        st.session_state['selected_idx'] = sel[0]
        idx = st.session_state['selected_idx']
        r = results[idx]
        st.write('Preview metrics:')
        st.json({'file': r['file'], 'bpm': r['bpm'], 'score': r['score'], 'decision': r['decision']})

        if st.button('Generate PDF report'):
            with st.spinner('Generating PDF...'):
                tempdir = tempfile.mkdtemp()
                img_path = save_full_multilead_plt(SELECTED_MODEL, st.session_state['model'], r['orig'], r['recon'], r['std'], tempdir)
                pdf_path = str(Path(tempdir) / (r['file'] + '.pdf'))
                metrics = {'BPM': r['bpm'], 'Anomaly Score': r['score'], 'Decision': r['decision']}
                pdf_file = generate_pdf_report(pdf_path, r['file'], metrics, [img_path])
                with open(pdf_file, 'rb') as f:
                    st.download_button('Download PDF', data=f, file_name=f"{r['file']}.pdf")
                st.success('Report generated')


st.caption('CardioScanX — pre-screening visualization only. NO DIAGNOSTIC INTERPRETATION IS PROVIDED.')
