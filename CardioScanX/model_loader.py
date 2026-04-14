import os
import sys
from pathlib import Path
import torch
import pickle
import importlib.util

# Path to local weights and models folders (copied from 12-lead-ECG-AD)
ROOT = Path(__file__).resolve().parent
SRC_WEIGHTS = ROOT / 'weights'
SRC_MODELS = ROOT / 'models'

# Legacy class aliases saved inside some checkpoints (e.g., training scripts defined MA_VAE inside __main__)
CLASS_ALIASES = {
    'ST_VAE': ['ST_VAE', 'STVAE', 'SpatiotemporalVAE'],
    'VAE_BILSTM_MHA': ['MA_VAE', 'VAE_MHA'],
    'VAE': ['MA_VAE', 'VAE'],
    'CAE': ['CAE'],
    'VAEEncoder': ['VAEEncoder'],
    'VAEDecoder': ['VAEDecoder'],
    'MHA': ['MHA'],
    'GaussianNoise': ['GaussianNoise'],
    'BetaScheduler': ['BetaScheduler']
}

_CLASSES_REGISTERED = False


def _register_all_model_classes():
    global _CLASSES_REGISTERED
    if _CLASSES_REGISTERED:
        return
    main_module = sys.modules.get('__main__')
    if main_module is None:
        return
    # Use local models directory
    models_dir = SRC_MODELS
    if not models_dir.exists():
        return
    for py_file in models_dir.glob('*.py'):
        if py_file.name.startswith('__'):
            continue
        try:
            spec = importlib.util.spec_from_file_location(py_file.stem, str(py_file))
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)  # type: ignore
                for attr_name, attr_value in module.__dict__.items():
                    if isinstance(attr_value, type):
                        setattr(main_module, attr_name, attr_value)
                        for alias in CLASS_ALIASES.get(attr_name, []):
                            setattr(main_module, alias, attr_value)
        except Exception:
            continue
    _CLASSES_REGISTERED = True

# Map descriptive names to weight files and model classes
MODEL_REGISTRY = {
    'ST-VAE (Spatiotemporal VAE)': {
        'weight': SRC_WEIGHTS / 'best_st_vae_model.pt',
        'class_file': SRC_MODELS / 'st_vae.py',
        'class_name': 'ST_VAE'
    },
    'CAE (Reconstruction Autoencoder)': {
        'weight': SRC_WEIGHTS / 'best_cae_model.pth',
        'class_file': SRC_MODELS / 'cae.py',
        'class_name': 'CAE'
    },
    'VAE (BiLSTM Attention)': {
        'weight': SRC_WEIGHTS / 'best_vae_attn_model.pt',
        'class_file': SRC_MODELS / 'vae_bilstm_attention.py',
        'class_name': 'VAE'
    },
    'VAE-MHA (MA-VAE variant)': {
        'weight': SRC_WEIGHTS / 'best_ma_vae.pth',
        'class_file': SRC_MODELS / 'vae_bilstm_mha.py',
        'class_name': 'VAE_BILSTM_MHA'
    },
    'VAE-GRU (GRU-based VAE)': {
        'weight': SRC_WEIGHTS / 'best_ma_vae_full.pt',
        'class_file': SRC_MODELS / 'vae_gru.py',
        'class_name': 'VAE'
    }
}


def load_model_by_name(name: str, map_location='cpu'):
    """
    Load a model as-is from the source `src/weights` -- DO NOT alter or change weights.
    Returns: (model, metadata)
    The model is put into eval() mode and all parameters require_grad=False.

    The weights file is loaded EXACTLY from the path in the original project. This enforces
    the "never modify models" requirement.
    """
    assert name in MODEL_REGISTRY, f"Unknown model name: {name}"
    info = MODEL_REGISTRY[name]
    _register_all_model_classes()
    weight_path = info['weight']
    class_file = info['class_file']
    class_name = info['class_name']

    # Validate the choice
    if not weight_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weight_path}")

    # Dynamic import: first try the original src, otherwise fall back to CardioScanX/models
    if not class_file.exists():
        # fallback local copy inside CardioScanX/models
        local = ROOT / 'models' / class_file.name
        if local.exists():
            class_file = local
    spec = importlib.util.spec_from_file_location('model_module', str(class_file))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore

    ModelClass = getattr(module, class_name)

    # Make all classes defined in the original module available under __main__ so
    # torch.load fallbacks can resolve pickled references like '__main__.VAEEncoder'.
    main_module = sys.modules.get('__main__')
    if main_module is not None:
        for attr_name, attr_value in module.__dict__.items():
            if isinstance(attr_value, type):
                setattr(main_module, attr_name, attr_value)
                for alias in CLASS_ALIASES.get(attr_name, []):
                    setattr(main_module, alias, attr_value)

    # Use a safe factory: do not override model code, only instantiate and map weights
    model = None
    # try common constructors, try instantiation defensively
    try:
        # for ST_VAE and most VAE models: ModelClass(n_leads=12)
        model = ModelClass(n_leads=12)
    except TypeError:
        try:
            # for CAE: CAE(in_channels=12) or CAE(12)
            model = ModelClass(12)
        except TypeError:
            try:
                model = ModelClass(seq_len=500, n_leads=12, latent_dim=64)
            except Exception as e:
                raise RuntimeError(f"Cannot instantiate model class {class_name}: {e}")

    # Load weights (no modifications).
    # By default in recent PyTorch versions `torch.load` may use `weights_only=True`
    # which prevents unpickling non-weight objects. If that fails (Pickle UnpicklingError)
    # we try again with `weights_only=False` for local trusted checkpoints.
    try:
        state = torch.load(str(weight_path), map_location=map_location)
    except pickle.UnpicklingError as e:
        # Warning: this second call will run arbitrary code inside the checkpoint.
        # Only fall back if the file is trusted (local model weights in this repo).
        try:
            state = torch.load(str(weight_path), map_location=map_location, weights_only=False)
        except Exception as e2:
            raise RuntimeError(f"Weights only load failed and fallback also failed: {e2}")

    # Best practice: if state is a dict with model keys, try loading
    try:
        if isinstance(state, dict) and 'state_dict' in state:
            state_dict = state['state_dict']
        else:
            state_dict = state
        try:
            model.load_state_dict(state_dict)
        except Exception as e_load:
            # Some checkpoints may differ in layer names (renaming in new code).
            # We attempt a safe key remapping that DOES NOT change values - only renames keys
            def remap_keys(sd: dict):
                new_sd = {}
                for k, v in sd.items():
                    # Map legacy prefix 'ma.' to 'mha.' (common rename in MHA modules)
                    if k.startswith('ma.') and not k.startswith('mha.'):
                        new_sd['mha.' + k[len('ma.'):]] = v
                    else:
                        new_sd[k] = v
                return new_sd

            new_state = remap_keys(state_dict)
            try:
                model.load_state_dict(new_state)
            except Exception:
                # Could not remap in a straightforward way — bubble up original error
                raise RuntimeError(e_load)
    except Exception:
        # try passing the whole object directly (some checkpoints contain the whole model)
        try:
            model.load_state_dict(state)
        except Exception as e:
            # Last resort: assume this weight file is a state dict produced elsewhere, warn
            raise RuntimeError(f"Could not load weights from {weight_path}: {e}")

    # Freeze the weights and set evaluation mode
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return model, info
