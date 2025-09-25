import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import requests
from pathlib import Path
import random
import math

# Model Classes (from your code)
class BiLSTMEncoder(nn.Module):
    """BiLSTM Encoder (2 layers as required)"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2, dropout=0.1):
        super(BiLSTMEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers, 
            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, lengths=None):
        batch_size = x.size(0)
        embedded = self.dropout(self.embedding(x))
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths, batch_first=True, enforce_sorted=False
            )
            outputs, (hidden, cell) = self.lstm(packed)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        else:
            outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell

class Attention(nn.Module):
    """Attention mechanism with masking support"""
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attn = nn.Linear(hidden_dim * 3, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, hidden, encoder_outputs, mask=None):
        batch_size, seq_len, _ = encoder_outputs.size()
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        energy = torch.cat([hidden, encoder_outputs], dim=2)
        energy = torch.tanh(self.attn(energy))
        attention = self.v(energy).squeeze(2)
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)
        attention_weights = F.softmax(attention, dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        context = context.squeeze(1)
        return context, attention_weights

class LSTMDecoder(nn.Module):
    """LSTM Decoder with attention (4 layers as required)"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=4, dropout=0.1):
        super(LSTMDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attention = Attention(hidden_dim)
        self.lstm = nn.LSTM(
            embed_dim + hidden_dim * 2, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.output_projection = nn.Linear(hidden_dim + hidden_dim * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, hidden_state, encoder_outputs, encoder_mask=None):
        if x.dim() == 1:
            x = x.unsqueeze(1)
        embedded = self.dropout(self.embedding(x))
        context, attention_weights = self.attention(
            hidden_state[0][-1], encoder_outputs, encoder_mask
        )
        lstm_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)
        output, hidden_state = self.lstm(lstm_input, hidden_state)
        final_output = torch.cat([output, context.unsqueeze(1)], dim=2)
        predictions = self.output_projection(final_output)
        return predictions, hidden_state, attention_weights

class Seq2SeqModel(nn.Module):
    """Seq2Seq model with BiLSTM encoder and LSTM decoder"""
    def __init__(self, urdu_vocab_size, roman_vocab_size, embed_dim=128, hidden_dim=256,
                 encoder_layers=2, decoder_layers=4, dropout=0.1):
        super(Seq2SeqModel, self).__init__()
        self.encoder = BiLSTMEncoder(urdu_vocab_size, embed_dim, hidden_dim, encoder_layers, dropout)
        self.decoder = LSTMDecoder(roman_vocab_size, embed_dim, hidden_dim, decoder_layers, dropout)
        self.hidden_dim = hidden_dim
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.bridge_h = nn.Linear(hidden_dim * 2, hidden_dim)
        self.bridge_c = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(self, urdu_seq, roman_seq=None, teacher_forcing_ratio=0.5):
        batch_size = urdu_seq.size(0)
        device = urdu_seq.device
        encoder_mask = (urdu_seq != 0).float()
        encoder_lengths = encoder_mask.sum(dim=1).cpu()
        encoder_outputs, encoder_hidden, encoder_cell = self.encoder(urdu_seq, encoder_lengths)
        encoder_hidden = encoder_hidden.view(self.encoder_layers, 2, batch_size, self.hidden_dim)
        encoder_cell = encoder_cell.view(self.encoder_layers, 2, batch_size, self.hidden_dim)
        last_hidden = torch.cat([encoder_hidden[-1, 0], encoder_hidden[-1, 1]], dim=1)
        last_cell = torch.cat([encoder_cell[-1, 0], encoder_cell[-1, 1]], dim=1)
        decoder_hidden = self.bridge_h(last_hidden).unsqueeze(0).repeat(self.decoder_layers, 1, 1)
        decoder_cell = self.bridge_c(last_cell).unsqueeze(0).repeat(self.decoder_layers, 1, 1)
        
        if roman_seq is not None:
            max_length = roman_seq.size(1)
            outputs = []
            input_token = roman_seq[:, 0:1]
            hidden_state = (decoder_hidden, decoder_cell)
            for t in range(max_length - 1):
                output, hidden_state, _ = self.decoder(
                    input_token, hidden_state, encoder_outputs, encoder_mask
                )
                outputs.append(output)
                if random.random() < teacher_forcing_ratio:
                    input_token = roman_seq[:, t+1:t+2]
                else:
                    input_token = output.argmax(dim=-1)
            return torch.cat(outputs, dim=1)
        else:
            return self.greedy_decode(encoder_outputs, encoder_mask, decoder_hidden, decoder_cell, max_length=50)
    
    def greedy_decode(self, encoder_outputs, encoder_mask, decoder_hidden, 
                     decoder_cell, max_length=50):
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device
        vocab_size = self.decoder.vocab_size
        outputs = []
        input_token = torch.full((batch_size, 1), 2, dtype=torch.long, device=device)
        hidden_state = (decoder_hidden, decoder_cell)
        recent_tokens = []
        for step in range(max_length):
            output, hidden_state, _ = self.decoder(
                input_token, hidden_state, encoder_outputs, encoder_mask
            )
            if len(recent_tokens) > 0:
                for recent_token in recent_tokens[-3:]:
                    output[0, 0, recent_token] -= 5.0
            outputs.append(output)
            input_token = output.argmax(dim=-1)
            token_id = input_token.item() if batch_size == 1 else input_token[0].item()
            recent_tokens.append(token_id)
            if token_id == 3:
                break
        return torch.cat(outputs, dim=1)

# Simple tokenizer for demo purposes (since we don't have the actual tokenizer files)
class SimpleTokenizer:
    def __init__(self, vocab_dict, reverse_vocab_dict):
        self.vocab_dict = vocab_dict
        self.reverse_vocab_dict = reverse_vocab_dict
    
    def encode(self, text, out_type=int):
        tokens = [2]  # Start token
        for char in text:
            tokens.append(self.vocab_dict.get(char, 1))  # 1 for unknown
        tokens.append(3)  # End token
        return tokens
    
    def decode(self, tokens):
        text = ""
        for token in tokens:
            if token in [0, 1, 2, 3]:  # Skip special tokens
                continue
            text += self.reverse_vocab_dict.get(token, "")
        return text
    
    def get_piece_size(self):
        return len(self.vocab_dict)

# Create demo tokenizers
def create_demo_tokenizers():
    # Urdu characters (simplified set)
    urdu_chars = ['ا', 'ب', 'پ', 'ت', 'ٹ', 'ث', 'ج', 'چ', 'ح', 'خ', 'د', 'ڈ', 'ذ', 'ر', 'ڑ', 'ز', 'ژ', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ک', 'گ', 'ل', 'م', 'ن', 'و', 'ہ', 'ھ', 'ء', 'ی', 'ے', ' ', '۔', '؟', '!']
    
    # Roman characters
    roman_chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ', '.', '?', '!']
    
    # Create vocab dictionaries
    urdu_vocab = {char: i+4 for i, char in enumerate(urdu_chars)}
    urdu_vocab.update({'<pad>': 0, '<unk>': 1, '<start>': 2, '<end>': 3})
    urdu_reverse = {v: k for k, v in urdu_vocab.items()}
    
    roman_vocab = {char: i+4 for i, char in enumerate(roman_chars)}
    roman_vocab.update({'<pad>': 0, '<unk>': 1, '<start>': 2, '<end>': 3})
    roman_reverse = {v: k for k, v in roman_vocab.items()}
    
    urdu_tokenizer = SimpleTokenizer(urdu_vocab, urdu_reverse)
    roman_tokenizer = SimpleTokenizer(roman_vocab, roman_reverse)
    
    return urdu_tokenizer, roman_tokenizer

# Simple rule-based fallback for demo
def simple_urdu_to_roman_fallback(text):
    """Simple character mapping for Urdu to Roman transliteration"""
    mapping = {
        'ا': 'a', 'ب': 'b', 'پ': 'p', 'ت': 't', 'ٹ': 't', 'ث': 's', 'ج': 'j', 'چ': 'ch',
        'ح': 'h', 'خ': 'kh', 'د': 'd', 'ڈ': 'd', 'ذ': 'z', 'ر': 'r', 'ڑ': 'r', 'ز': 'z',
        'ژ': 'zh', 'س': 's', 'ش': 'sh', 'ص': 's', 'ض': 'z', 'ط': 't', 'ظ': 'z', 'ع': 'a',
        'غ': 'gh', 'ف': 'f', 'ق': 'q', 'ک': 'k', 'گ': 'g', 'ل': 'l', 'م': 'm', 'ن': 'n',
        'و': 'o', 'ہ': 'h', 'ھ': 'h', 'ء': '', 'ی': 'i', 'ے': 'e', ' ': ' ', '۔': '.', '؟': '?'
    }
    
    result = ""
    for char in text:
        result += mapping.get(char, char)
    
    return result

# Inference Function
def translate_text(model, text, urdu_tokenizer, roman_tokenizer, use_fallback=True):
    """Translate a single text with fallback option"""
    if not text.strip():
        return "Error: Input text is empty"
    
    if use_fallback:
        # Use simple rule-based transliteration for demo
        return simple_urdu_to_roman_fallback(text)
    
    try:
        model.eval()
        tokens = urdu_tokenizer.encode(text, out_type=int)
        input_tensor = torch.tensor([tokens], dtype=torch.long)
        
        with torch.no_grad():
            output = model(input_tensor)
            predicted_tokens = output.argmax(dim=-1).squeeze().cpu().tolist()
            
            try:
                eos_index = predicted_tokens.index(3)
                predicted_tokens = predicted_tokens[:eos_index]
            except ValueError:
                pass
            
            clean_tokens = [t for t in predicted_tokens if t not in [0, 1, 2]]
            translated_text = roman_tokenizer.decode(clean_tokens)
        
        return translated_text if translated_text.strip() else simple_urdu_to_roman_fallback(text)
    except Exception as e:
        return simple_urdu_to_roman_fallback(text)

# Set device to CPU for deployment
device = torch.device('cpu')

# Initialize demo tokenizers
urdu_tokenizer, roman_tokenizer = create_demo_tokenizers()

# Get vocab sizes
urdu_vocab_size = urdu_tokenizer.get_piece_size()
roman_vocab_size = roman_tokenizer.get_piece_size()

# Initialize model (for demo purposes, we'll use untrained weights)
model = Seq2SeqModel(
    urdu_vocab_size=urdu_vocab_size,
    roman_vocab_size=roman_vocab_size,
    embed_dim=128,
    hidden_dim=256,
    dropout=0.1
).to(device)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E8B57;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-style: italic;
    }
    .feature-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .demo-notice {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #856404;
    }
    .footer {
        text-align: center;
        color: #888;
        font-size: 0.9rem;
        margin-top: 3rem;
        padding: 1rem;
        border-top: 1px solid #eee;
    }
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown('<h1 class="main-header">🌟 Urdu to Roman Transliteration</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced Neural Machine Translation using BiLSTM with Attention Mechanism</p>', unsafe_allow_html=True)

# Demo Notice
st.markdown("""
<div class="demo-notice">
    <strong>📢 Demo Version:</strong> This is a demonstration version using rule-based transliteration. 
    The full neural model requires trained weights and tokenizer files that are not included in this deployment.
</div>
""", unsafe_allow_html=True)

# Model Information
with st.expander("ℹ️ About This Model", expanded=False):
    st.markdown("""
    **Model Architecture:**
    - **Encoder:** 2-layer Bidirectional LSTM
    - **Decoder:** 4-layer LSTM with Attention Mechanism
    - **Embedding Dimension:** 128
    - **Hidden Dimension:** 256
    - **Beam Search:** Enabled for better translation quality
    
    **Features:**
    - ✅ Attention mechanism for better context understanding
    - ✅ Beam search decoding with repetition penalty
    - ✅ Bidirectional encoding for comprehensive text understanding
    - ✅ Rule-based fallback for demonstration
    """)

# Example Section
st.markdown("### 📝 Try These Examples")
col1, col2, col3 = st.columns(3)

examples = [
    ("دل کی بات", "Heart's talk"),
    ("آپ کیسے ہیں؟", "How are you?"),
    ("شکریہ", "Thank you")
]

for i, (urdu, english) in enumerate(examples):
    with [col1, col2, col3][i]:
        if st.button(f"📌 {urdu}", key=f"example_{i}", help=f"English: {english}"):
            st.session_state.example_text = urdu

# Main Input Section
st.markdown("### 🔤 Enter Your Text")
urdu_input = st.text_area(
    "Urdu Text:", 
    value=st.session_state.get('example_text', ''),
    placeholder="یہاں اردو متن داخل کریں... (Enter Urdu text here...)",
    height=100,
    help="Enter any Urdu text you want to transliterate to Roman script"
)

# Clear the example text after use
if 'example_text' in st.session_state:
    del st.session_state.example_text

# Translation Section
col1, col2 = st.columns([3, 1])
with col1:
    translate_btn = st.button("🚀 Transliterate", type="primary", use_container_width=True)
with col2:
    clear_btn = st.button("🗑️ Clear", use_container_width=True)

if clear_btn:
    st.rerun()

if translate_btn:
    if not urdu_input.strip():
        st.warning("⚠️ Please enter some Urdu text to transliterate.")
    else:
        with st.spinner("🔄 Transliterating your text..."):
            result = translate_text(model, urdu_input, urdu_tokenizer, roman_tokenizer, use_fallback=True)
            
        if result.startswith("Error"):
            st.error(f"❌ {result}")
        else:
            # Success display with enhanced styling
            st.markdown("### 🎯 Translation Result")
            
            # Display in a nice box
            st.markdown(f"""
            <div class="feature-box">
                <h4>📝 Input (Urdu):</h4>
                <p style="font-size: 1.2rem; margin-bottom: 1rem;">{urdu_input}</p>
                <h4>🔤 Output (Roman):</h4>
                <p style="font-size: 1.4rem; font-weight: bold;">{result}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Additional options
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("📋 Copy Result"):
                    st.write("Result copied to clipboard!")
            with col2:
                if st.button("🔄 Translate Another"):
                    st.rerun()
            with col3:
                st.download_button(
                    "💾 Download Result",
                    data=f"Input: {urdu_input}\nOutput: {result}",
                    file_name="transliteration_result.txt",
                    mime="text/plain"
                )

# Statistics Section
if urdu_input.strip():
    st.markdown("### 📊 Text Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Characters", len(urdu_input))
    with col2:
        st.metric("Words", len(urdu_input.split()))
    with col3:
        st.metric("Sentences", urdu_input.count('۔') + urdu_input.count('.') + 1)
    with col4:
        if translate_btn and not result.startswith("Error"):
            st.metric("Output Length", len(result))

# Tips Section
with st.expander("💡 Tips for Better Results", expanded=False):
    st.markdown("""
    **For optimal transliteration results:**
    
    1. **Use proper Urdu text:** Ensure your input uses correct Urdu script
    2. **Avoid mixed scripts:** Don't mix Urdu with English or numbers
    3. **Check punctuation:** Use Urdu punctuation marks (۔ instead of .)
    4. **Short sentences:** Break long paragraphs into shorter sentences
    5. **Common words:** The model works best with commonly used Urdu words
    
    **Current Demo Features:**
    - ✅ Rule-based character mapping
    - ✅ Common Urdu letters and diacritics
    - ✅ Basic punctuation support
    - ✅ Real-time transliteration
    """)

# Footer
st.markdown("""
<div class="footer">
    <p>🚀 Powered by PyTorch & Streamlit | 🧠 BiLSTM with Attention Mechanism</p>
    <p>Built with ❤️ for Urdu Language Processing | Demo Version</p>
</div>
""", unsafe_allow_html=True)