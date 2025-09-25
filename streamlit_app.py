import streamlit as st
import torch
from pathlib import Path
import sentencepiece as spm
from nig import Seq2SeqModel, device as script_device, translate_text as script_translate

st.set_page_config(page_title="Urdu → Roman Transliteration", layout="centered")
st.title("Urdu → Roman Urdu Transliteration")
st.caption("Deploy and interact with your trained Seq2Seq model")

# --------- Utility functions ---------
@st.cache_resource
def load_tokenizers():
    # Try tokenizers/ folder first, then root
    urdu_paths = [
        Path("tokenizers/urdu_tokenizer.model"),
        Path("urdu_tokenizer.model"),
    ]
    roman_paths = [
        Path("tokenizers/roman_tokenizer.model"),
        Path("roman_tokenizer.model"),
    ]
    urdu_model_path = next((p for p in urdu_paths if p.exists()), None)
    roman_model_path = next((p for p in roman_paths if p.exists()), None)

    if urdu_model_path is None or roman_model_path is None:
        raise FileNotFoundError("Tokenizer models not found. Please run training first to generate 'urdu_tokenizer.model' and 'roman_tokenizer.model'.")

    urdu_tok = spm.SentencePieceProcessor(model_file=str(urdu_model_path))
    roman_tok = spm.SentencePieceProcessor(model_file=str(roman_model_path))
    return urdu_tok, roman_tok

@st.cache_resource
def build_model(urdu_vocab_size: int, roman_vocab_size: int, embed_dim: int = 128, hidden_dim: int = 256, dropout: float = 0.1, device: str = "cpu"):
    model = Seq2SeqModel(
        urdu_vocab_size=urdu_vocab_size,
        roman_vocab_size=roman_vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )
    return model.to(device)

def load_weights(model: torch.nn.Module, weights_path: Path, device: str = "cpu"):
    sd = torch.load(str(weights_path), map_location=device)
    model.load_state_dict(sd)
    model.eval()
    return model

# --------- Sidebar: configuration ---------
st.sidebar.header("Configuration")

device_choice = st.sidebar.selectbox("Device", ["Auto (CUDA if available)", "CPU", "CUDA"], index=0)
if device_choice == "CUDA" and torch.cuda.is_available():
    device = "cuda"
elif device_choice == "Auto (CUDA if available)" and torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
st.sidebar.write(f"Using device: {device}")

# Tokenizers
try:
    urdu_tokenizer, roman_tokenizer = load_tokenizers()
    st.sidebar.success(f"Tokenizers loaded (Urdu: {urdu_tokenizer.get_piece_size()}, Roman: {roman_tokenizer.get_piece_size()})")
except Exception as e:
    st.sidebar.error(str(e))
    st.stop()

# Model weights selection
default_weights = Path("best_model_baseline.pth")
use_default = default_weights.exists()
st.sidebar.write("Model Weights")
if use_default:
    st.sidebar.success("Found best_model_baseline.pth")
else:
    st.sidebar.warning("best_model_baseline.pth not found. Upload weights below.")

uploaded = st.sidebar.file_uploader("Upload model weights (.pth)", type=["pth"], accept_multiple_files=False)

# Load or build model
if "model" not in st.session_state:
    st.session_state.model = None

if st.sidebar.button("Load Model"):
    try:
        model = build_model(
            urdu_vocab_size=urdu_tokenizer.get_piece_size(),
            roman_vocab_size=roman_tokenizer.get_piece_size(),
            device=device,
        )
        if use_default and uploaded is None:
            weights_path = default_weights
        elif uploaded is not None:
            tmp_path = Path("uploaded_model.pth")
            with open(tmp_path, "wb") as f:
                f.write(uploaded.read())
            weights_path = tmp_path
        else:
            st.sidebar.error("No weights available. Provide best_model_baseline.pth or upload a file.")
            st.stop()
        st.session_state.model = load_weights(model, weights_path, device=device)
        st.sidebar.success("Model loaded successfully.")
    except Exception as e:
        st.sidebar.error(f"Failed to load model: {e}")

# --------- Main UI ---------
st.subheader("Try it out")
urdu_input = st.text_area("Enter Urdu text", height=150, help="Paste a verse or sentence in Urdu script")

col1, col2 = st.columns([1, 3])
with col1:
    translate_clicked = st.button("Translate")
with col2:
    clear_clicked = st.button("Clear")

if clear_clicked:
    st.experimental_rerun()

if translate_clicked:
    if st.session_state.model is None:
        st.error("Please load the model first from the sidebar.")
    elif not urdu_input.strip():
        st.error("Please enter some Urdu text.")
    else:
        try:
            # Prefer using the project's translate_text for consistency (handles EOS & special tokens)
            result = script_translate(st.session_state.model, urdu_input.strip(), urdu_tokenizer, roman_tokenizer)
            st.success("Translation complete")
            st.markdown("### Roman Urdu")
            st.write(result if result else "(No output)")
        except Exception as e:
            st.error(f"Translation failed: {e}")

st.markdown("---")
st.caption("Tip: If you don't see outputs, ensure your weights match the architecture (embed_dim=128, hidden_dim=256) and tokenizers used during training.")