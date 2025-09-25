
"""
Fixed Urdu to Roman Urdu Neural Machine Translation
Complete Implementation with all corrections

Key fixes:
1. Proper vocabulary size handling
2. Fixed repetition issue in inference
3. Added accuracy metrics
4. Fixed evaluation functions
5. Improved beam search decoding
"""

import os
import re
import json
import pickle
import random
import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import unicodedata
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Core libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

# Install required packages (run in Kaggle)
# !pip install sentencepiece

import sentencepiece as spm

# Set random seeds
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ===================== DATA PREPROCESSING =====================

class UrduRomanDataProcessor:
    """Data processor for Urdu-Roman translation pairs from urdu_ghazals_rekhta dataset"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.urdu_texts = []
        self.roman_texts = []
    
    def load_data(self):
        """Load data from the urdu_ghazals_rekhta dataset structure"""
        print("Loading data from urdu_ghazals_rekhta dataset...")
        
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path {self.dataset_path} not found")
        
        # Dataset structure: poets -> [ur, en, hi] -> files (no extensions)
        for poet_dir in self.dataset_path.iterdir():
            if not poet_dir.is_dir() or poet_dir.name.startswith('.'):
                continue
                
            urdu_dir = poet_dir / 'ur'
            english_dir = poet_dir / 'en'  # This contains Roman Urdu transliteration
            
            if not (urdu_dir.exists() and english_dir.exists()):
                continue
            
            # Get all Urdu files (no extension filter needed)
            urdu_files = [f for f in urdu_dir.iterdir() if f.is_file() and not f.name.startswith('.')]
            
            for urdu_file in urdu_files:
                english_file = english_dir / urdu_file.name
                
                if english_file.exists() and english_file.is_file():
                    try:
                        # Read Urdu text
                        with open(urdu_file, 'r', encoding='utf-8') as f:
                            urdu_content = f.read().strip()
                        
                        # Read Roman Urdu text
                        with open(english_file, 'r', encoding='utf-8') as f:
                            roman_content = f.read().strip()
                        
                        # Split by lines to get verse pairs
                        urdu_lines = [line.strip() for line in urdu_content.split('\n') if line.strip()]
                        roman_lines = [line.strip() for line in roman_content.split('\n') if line.strip()]
                        
                        # Pair up lines (verses)
                        for urdu_line, roman_line in zip(urdu_lines, roman_lines):
                            if urdu_line and roman_line:
                                self.urdu_texts.append(urdu_line)
                                self.roman_texts.append(roman_line)
                                
                    except Exception as e:
                        print(f"Error reading {urdu_file.name}: {e}")
                        continue
        
        print(f"Loaded {len(self.urdu_texts)} text pairs")
        
        if len(self.urdu_texts) == 0:
            raise ValueError("No data loaded. Check dataset structure and paths.")
    
    def clean_text(self, text: str, is_urdu: bool = True) -> str:
        """Clean and normalize text"""
        # Unicode normalization
        text = unicodedata.normalize('NFKC', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        if is_urdu:
            # Keep Urdu characters and basic punctuation
            text = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\s\.\,\?\!\:\;\-\(\)\"\']+', '', text)
        else:
            # Convert to lowercase and keep Roman characters
            text = text.lower()
            text = re.sub(r'[^a-z0-9\s\.\,\?\!\:\;\-\(\)\"\']+', '', text)
        
        return text.strip()
    
    def preprocess_data(self, min_words=3, max_words=50):
        """Clean and filter the data"""
        print("Preprocessing and filtering data...")
        
        cleaned_urdu = []
        cleaned_roman = []
        
        for urdu, roman in zip(self.urdu_texts, self.roman_texts):
            # Clean texts
            clean_urdu = self.clean_text(urdu, is_urdu=True)
            clean_roman = self.clean_text(roman, is_urdu=False)
            
            # Filter by length
            urdu_words = len(clean_urdu.split())
            roman_words = len(clean_roman.split())
            
            if (min_words <= urdu_words <= max_words and 
                min_words <= roman_words <= max_words and 
                clean_urdu and clean_roman):
                cleaned_urdu.append(clean_urdu)
                cleaned_roman.append(clean_roman)
        
        self.urdu_texts = cleaned_urdu
        self.roman_texts = cleaned_roman
        
        print(f"After preprocessing: {len(self.urdu_texts)} pairs")
        
        if len(self.urdu_texts) < 100:
            print("Warning: Very few text pairs available. Consider relaxing filtering criteria.")
    
    def split_data(self, test_size=0.25, val_size=0.25, random_state=42):
        """Split data into train/val/test sets (50/25/25 as required)"""
        # First split: separate test set (25%)
        X_temp, X_test, y_temp, y_test = train_test_split(
            self.urdu_texts, self.roman_texts, 
            test_size=test_size, random_state=random_state
        )
        
        # Second split: divide remaining into train/val
        val_adjusted = val_size / (1 - test_size)  # 0.25 / 0.75 = 0.333
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, 
            test_size=val_adjusted, random_state=random_state
        )
        
        print(f"Data split - Train: {len(X_train)} ({len(X_train)/len(self.urdu_texts)*100:.1f}%), "
              f"Val: {len(X_val)} ({len(X_val)/len(self.urdu_texts)*100:.1f}%), "
              f"Test: {len(X_test)} ({len(X_test)/len(self.urdu_texts)*100:.1f}%)")
        
        return {
            'train': {'urdu': X_train, 'roman': y_train},
            'val': {'urdu': X_val, 'roman': y_val},
            'test': {'urdu': X_test, 'roman': y_test}
        }

# ===================== TOKENIZATION =====================

def create_tokenizers(train_urdu, train_roman, vocab_size=4000):
    """
    Create and train SentencePiece tokenizers
    
    Why 4000-5000 vocab size?
    - Covers most frequent subwords and words
    - Balances between representation quality and model size
    - Small vocab (300-500) leads to excessive word splitting
    - Large vocab (10000+) may overfit on limited data
    """
    print(f"Training SentencePiece tokenizers with vocab_size={vocab_size}...")
    
    # Create tokenizers directory
    os.makedirs('tokenizers', exist_ok=True)
    
    # Save training data
    with open('tokenizers/urdu_train.txt', 'w', encoding='utf-8') as f:
        for text in train_urdu:
            f.write(text + '\n')
    
    with open('tokenizers/roman_train.txt', 'w', encoding='utf-8') as f:
        for text in train_roman:
            f.write(text + '\n')
    
    # Estimate reasonable vocab size based on data
    def estimate_vocab_size(texts, target_size):
        all_text = ' '.join(texts)
        unique_chars = len(set(all_text))
        unique_words = len(set(all_text.split()))
        # Use target size but cap at reasonable limits
        return min(target_size, max(unique_chars * 10, 1000), unique_words)
    
    urdu_vocab_size = estimate_vocab_size(train_urdu, vocab_size)
    roman_vocab_size = estimate_vocab_size(train_roman, vocab_size)
    
    print(f"Adjusted vocab sizes - Urdu: {urdu_vocab_size}, Roman: {roman_vocab_size}")
    
    # Train Urdu tokenizer
    spm.SentencePieceTrainer.train(
        input='tokenizers/urdu_train.txt',
        model_prefix='tokenizers/urdu_tokenizer',
        vocab_size=urdu_vocab_size,
        model_type='unigram',
        character_coverage=1.0,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3
    )
    
    # Train Roman tokenizer
    spm.SentencePieceTrainer.train(
        input='tokenizers/roman_train.txt',
        model_prefix='tokenizers/roman_tokenizer',
        vocab_size=roman_vocab_size,
        model_type='unigram',
        character_coverage=1.0,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3
    )
    
    # Load trained models
    urdu_tokenizer = spm.SentencePieceProcessor(model_file='tokenizers/urdu_tokenizer.model')
    roman_tokenizer = spm.SentencePieceProcessor(model_file='tokenizers/roman_tokenizer.model')
    
    print(f"Final vocab sizes - Urdu: {urdu_tokenizer.get_piece_size()}, "
          f"Roman: {roman_tokenizer.get_piece_size()}")
    
    return urdu_tokenizer, roman_tokenizer

# ===================== DATASET AND DATALOADER =====================

class TranslationDataset(Dataset):
    """Dataset for translation pairs"""
    
    def __init__(self, urdu_texts, roman_texts, urdu_tokenizer, roman_tokenizer, max_length=50):
        self.urdu_texts = urdu_texts
        self.roman_texts = roman_texts
        self.urdu_tokenizer = urdu_tokenizer
        self.roman_tokenizer = roman_tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.urdu_texts)
    
    def __getitem__(self, idx):
        urdu_text = self.urdu_texts[idx]
        roman_text = self.roman_texts[idx]
        
        # Tokenize
        urdu_tokens = self.urdu_tokenizer.encode(urdu_text, out_type=int)
        roman_tokens = self.roman_tokenizer.encode(roman_text, out_type=int)
        
        # Ensure BOS and EOS in target and respect max_length
        roman_tokens = roman_tokens[:max(0, self.max_length - 2)]
        roman_tokens = [2] + roman_tokens + [3]
        
        # Truncate if necessary
        urdu_tokens = urdu_tokens[:self.max_length]
        
        return {
            'urdu': torch.tensor(urdu_tokens, dtype=torch.long),
            'roman': torch.tensor(roman_tokens, dtype=torch.long),
            'urdu_text': urdu_text,
            'roman_text': roman_text
        }

def collate_fn(batch):
    """Collate function with padding"""
    urdu_seqs = [item['urdu'] for item in batch]
    roman_seqs = [item['roman'] for item in batch]
    
    # Pad sequences
    urdu_padded = nn.utils.rnn.pad_sequence(urdu_seqs, batch_first=True, padding_value=0)
    roman_padded = nn.utils.rnn.pad_sequence(roman_seqs, batch_first=True, padding_value=0)
    
    return {
        'urdu': urdu_padded,
        'roman': roman_padded,
        'urdu_texts': [item['urdu_text'] for item in batch],
        'roman_texts': [item['roman_text'] for item in batch]
    }

# ===================== MODEL ARCHITECTURE =====================

class xLSTMEncoder(nn.Module):
    """xLSTM-style Encoder: BiLSTM with LayerNorm on outputs for stability"""
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2, dropout=0.1):
        super(xLSTMEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(hidden_dim * 2)
        
    def forward(self, x, lengths=None):
        # x: (batch_size, seq_len)
        
        # Embedding with dropout
        embedded = self.dropout(self.embedding(x))
        
        # Pack padded sequence for efficiency if lengths provided
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths, batch_first=True, enforce_sorted=False
            )
            outputs, (hidden, cell) = self.lstm(packed)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        else:
            outputs, (hidden, cell) = self.lstm(embedded)
        
        # LayerNorm over the feature dimension
        outputs = self.out_norm(outputs)
        
        return outputs, hidden, cell

# Attention class removed (unused)

class xLSTMDecoder(nn.Module):
    """xLSTM-style Decoder: LSTM with LayerNorm on hidden and output, weight tying preserved"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=4, dropout=0.1):
        super(xLSTMDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.context_proj = nn.Linear(hidden_dim * 2, embed_dim)
        self.lstm = nn.LSTM(embed_dim * 2, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.h2e = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.h_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x, hidden_state, encoder_outputs, encoder_mask=None):
        if x.dim() == 1:
            x = x.unsqueeze(1)
        embedded = self.dropout(self.embedding(x))
        if encoder_mask is not None:
            lengths = encoder_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            context = (encoder_outputs * encoder_mask.unsqueeze(-1)).sum(dim=1) / lengths
        else:
            context = encoder_outputs.mean(dim=1)
        context = self.context_proj(context).unsqueeze(1)
        dec_input = torch.cat([embedded, context], dim=-1)
        output, (h, c) = self.lstm(dec_input, hidden_state)
        # LayerNorm on hidden/output
        output = self.out_norm(output)
        h = self.h_norm(h)
        logits = F.linear(self.h2e(output), self.embedding.weight)
        return logits, (h, c), None

class LSTMDecoder(nn.Module):
    """LSTM Decoder without attention (4 layers as required)"""
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=4, dropout=0.1):
        super(LSTMDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Fixed encoder context: project mean-pooled encoder outputs to embedding dim
        self.context_proj = nn.Linear(hidden_dim * 2, embed_dim)
        
        # Decoder operates on [embedding ; fixed_context]
        self.lstm = nn.LSTM(
            embed_dim * 2, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # Weight tying: map hidden_dim -> embed_dim, then tie to embedding weight for logits
        self.h2e = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, hidden_state, encoder_outputs, encoder_mask=None):
        # x: (batch_size, 1) - single time step
        if x.dim() == 1:
            x = x.unsqueeze(1)
        
        embedded = self.dropout(self.embedding(x))
        
        # Compute fixed context (masked mean over encoder sequence)
        if encoder_mask is not None:
            lengths = encoder_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            context = (encoder_outputs * encoder_mask.unsqueeze(-1)).sum(dim=1) / lengths
        else:
            context = encoder_outputs.mean(dim=1)
        context = self.context_proj(context).unsqueeze(1)  # (batch, 1, embed_dim)
        
        # Concatenate embedding with fixed context
        dec_input = torch.cat([embedded, context], dim=-1)
        
        output, hidden_state = self.lstm(dec_input, hidden_state)
        
        # Project to embedding space and tie with embedding weights to produce logits
        logits = F.linear(self.h2e(output), self.embedding.weight)
        
        return logits, hidden_state, None

class Seq2SeqModel(nn.Module):
    """Seq2Seq model with BiLSTM encoder and LSTM decoder"""
    
    def __init__(self, urdu_vocab_size, roman_vocab_size, embed_dim=128, hidden_dim=256,
                 encoder_layers=2, decoder_layers=4, dropout=0.1, decoder_word_dropout=0.0):
        super(Seq2SeqModel, self).__init__()
        
        self.encoder = xLSTMEncoder(urdu_vocab_size, embed_dim, hidden_dim, encoder_layers, dropout)
        self.decoder = xLSTMDecoder(roman_vocab_size, embed_dim, hidden_dim, decoder_layers, dropout)
        
        self.hidden_dim = hidden_dim
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        
        self.bridge_h = nn.Linear(hidden_dim * 2, hidden_dim)
        self.bridge_c = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Decoder word dropout probability used during teacher forcing
        self.decoder_word_dropout = decoder_word_dropout

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
        
        # Use tanh bridge for stability
        decoder_hidden = torch.tanh(self.bridge_h(last_hidden)).unsqueeze(0).repeat(self.decoder_layers, 1, 1)
        decoder_cell = torch.tanh(self.bridge_c(last_cell)).unsqueeze(0).repeat(self.decoder_layers, 1, 1)
        
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
                
                # Decide next input
                if random.random() < teacher_forcing_ratio:
                    next_token = roman_seq[:, t+1:t+2]
                    # Apply decoder word dropout on teacher-forced inputs
                    if self.training and self.decoder_word_dropout > 0.0:
                        # Replace with UNK (id=1) with probability p
                        dropout_mask = (torch.rand_like(next_token.float()) < self.decoder_word_dropout)
                        next_token = next_token.masked_fill(dropout_mask, 1)
                    input_token = next_token
                else:
                    input_token = output.argmax(dim=-1)
            
            return torch.cat(outputs, dim=1)
        else:
            # Dynamic decode length based on input length
            dyn_max_len = int(encoder_lengths.max().item() * 1.5) + 10
            dyn_max_len = max(dyn_max_len, 20)
            return self.beam_search_decode(
                encoder_outputs, encoder_mask, decoder_hidden, decoder_cell,
                beam_size=5, max_length=dyn_max_len
            )
    
    def beam_search_decode(self, encoder_outputs, encoder_mask, decoder_hidden, 
                          decoder_cell, beam_size=5, max_length=50):
        """Beam search decoding to avoid repetition with length penalty"""
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device
        
        if batch_size > 1:
            return self.greedy_decode(encoder_outputs, encoder_mask, decoder_hidden, 
                                    decoder_cell, max_length)
        
        beams = [([], 0.0, (decoder_hidden, decoder_cell))]
        completed = []
        start_token = torch.tensor([[2]], dtype=torch.long, device=device)
        
        alpha = 0.6  # length penalty factor
        unk_id = 1   # UNK token id
        
        for step in range(max_length):
            candidates = []
            
            for seq, score, hidden_state in beams:
                if len(seq) > 0 and seq[-1] == 3:
                    completed.append((seq, score))
                    continue
                
                input_token = start_token if len(seq) == 0 else torch.tensor([[seq[-1]]], dtype=torch.long, device=device)
                
                output, new_hidden, _ = self.decoder(
                    input_token, hidden_state, encoder_outputs, encoder_mask
                )
                
                log_probs = F.log_softmax(output.squeeze(1), dim=-1)
                # Penalize UNK
                if log_probs.size(-1) > unk_id:
                    log_probs[0, unk_id] -= 2.0
                
                top_k_scores, top_k_tokens = log_probs.topk(beam_size)
                
                for k in range(beam_size):
                    token = top_k_tokens[0, k].item()
                    token_score = top_k_scores[0, k].item()
                    
                    if len(seq) > 0 and token == seq[-1]:
                        token_score -= 2.0
                    if len(seq) > 3:
                        recent = seq[-3:]
                        if recent.count(token) > 1:
                            token_score -= 3.0
                    
                    new_seq = seq + [token]
                    new_score = score + token_score
                    candidates.append((new_seq, new_score, new_hidden))
            
            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:beam_size]
            
            if len(beams) == 0:
                break
        
        completed.extend(beams)
        
        if completed:
            def lp(s):
                L = max(1, len(s))
                return L ** alpha
            best_seq = max(completed, key=lambda x: x[1] / lp(x[0]))[0]
        else:
            best_seq = beams[0][0] if beams else []
        
        if best_seq:
            vocab_size = self.decoder.vocab_size
            output_probs = torch.zeros(1, len(best_seq), vocab_size, device=device)
            for i, token in enumerate(best_seq):
                output_probs[0, i, token] = 1.0
        else:
            # Emit a single EOS step to avoid zero-length outputs
            vocab_size = self.decoder.vocab_size
            output_probs = torch.zeros(1, 1, vocab_size, device=device)
            output_probs[0, 0, 3] = 1.0  # EOS
        
        return output_probs
    
    def greedy_decode(self, encoder_outputs, encoder_mask, decoder_hidden, 
                     decoder_cell, max_length=50):
        """Greedy decoding with repetition penalty"""
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device
        vocab_size = self.decoder.vocab_size
        
        outputs = []
        input_token = torch.full((batch_size, 1), 2, dtype=torch.long, device=device)  # SOS
        hidden_state = (decoder_hidden, decoder_cell)
        
        # Track recent tokens for repetition penalty
        recent_tokens = []
        unk_id = 1
        
        for step in range(max_length):
            output, hidden_state, _ = self.decoder(
                input_token, hidden_state, encoder_outputs, encoder_mask
            )
            
            # Penalize UNK
            if output.size(-1) > unk_id:
                output[:, :, unk_id] -= 2.0
            
            # Apply repetition penalty
            if len(recent_tokens) > 0:
                for recent_token in recent_tokens[-3:]:  # Penalize last 3 tokens
                    if recent_token < output.size(-1):
                        output[0, 0, recent_token] -= 5.0
            
            outputs.append(output)
            
            # Get next token
            input_token = output.argmax(dim=-1)
            token_id = input_token.item() if batch_size == 1 else input_token[0].item()
            
            recent_tokens.append(token_id)
            
            # Stop if EOS token
            if token_id == 3:
                break
        
        return torch.cat(outputs, dim=1)

# ===================== ENHANCED TOKENIZER CREATION =====================

def create_tokenizers(urdu_texts, roman_texts, vocab_size=8000):
    """Create SentencePiece tokenizers for Urdu and Roman text"""
    import tempfile
    import os
    
    # Create temporary files for training data
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
        urdu_temp_file = f.name
        for text in urdu_texts:
            f.write(text + '\n')
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
        roman_temp_file = f.name
        for text in roman_texts:
            f.write(text + '\n')
    
    try:
        # Train Urdu tokenizer
        spm.SentencePieceTrainer.train(
            input=urdu_temp_file,
            model_prefix='urdu_tokenizer',
            vocab_size=vocab_size,
            character_coverage=1.0,
            model_type='bpe',
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3
        )
        
        # Train Roman tokenizer
        spm.SentencePieceTrainer.train(
            input=roman_temp_file,
            model_prefix='roman_tokenizer',
            vocab_size=vocab_size,
            character_coverage=1.0,
            model_type='bpe',
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3
        )
        
        # Load tokenizers
        urdu_tokenizer = spm.SentencePieceProcessor()
        urdu_tokenizer.load('urdu_tokenizer.model')
        
        roman_tokenizer = spm.SentencePieceProcessor()
        roman_tokenizer.load('roman_tokenizer.model')
        
        print(f"Urdu tokenizer vocabulary size: {urdu_tokenizer.get_piece_size()}")
        print(f"Roman tokenizer vocabulary size: {roman_tokenizer.get_piece_size()}")
        
        return urdu_tokenizer, roman_tokenizer
        
    finally:
        # Clean up temporary files
        try:
            os.unlink(urdu_temp_file)
            os.unlink(roman_temp_file)
        except:
            pass

# ===================== EVALUATION METRICS =====================

def calculate_perplexity(model, data_loader, criterion, device):
    """Calculate perplexity"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in data_loader:
            urdu_seq = batch['urdu'].to(device)
            roman_seq = batch['roman'].to(device)
            
            decoder_target = roman_seq[:, 1:]
            # Use full teacher forcing to compute true NLL
            outputs = model(urdu_seq, roman_seq, teacher_forcing_ratio=1.0)
            
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), decoder_target.reshape(-1))
            non_pad_tokens = (decoder_target != 0).sum().item()
            
            total_loss += loss.item() * non_pad_tokens
            total_tokens += non_pad_tokens
    
    if total_tokens == 0:
        return float('inf')
    
    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)

def calculate_cer(predictions, targets, tokenizer):
    """Calculate Character Error Rate"""
    def edit_distance_cer(s1, s2):
        if len(s1) < len(s2):
            return edit_distance_cer(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    total_chars = 0
    total_errors = 0
    
    for pred, target in zip(predictions, targets):
        # Convert to lists and remove special tokens
        if hasattr(pred, 'tolist'):
            pred_tokens = pred.tolist()
        else:
            pred_tokens = list(pred)
        
        if hasattr(target, 'tolist'):
            target_tokens = target.tolist()
        else:
            target_tokens = list(target)
        
        pred_clean = [t for t in pred_tokens if t not in [0, 1, 2, 3]]
        target_clean = [t for t in target_tokens if t not in [0, 1, 2, 3]]
        
        if len(pred_clean) > 0 and len(target_clean) > 0:
            pred_text = tokenizer.decode(pred_clean)
            target_text = tokenizer.decode(target_clean)
            
            errors = edit_distance_cer(pred_text, target_text)
            total_errors += errors
            total_chars += len(target_text)
    
    return total_errors / total_chars if total_chars > 0 else 1.0

def edit_distance(s1, s2):
    """Calculate edit distance (Levenshtein distance) between two strings"""
    if len(s1) < len(s2):
        return edit_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def calculate_bleu_score(predictions, targets, tokenizer):
    """Calculate BLEU score properly"""
    from collections import Counter
    
    def get_ngrams(tokens, n):
        if len(tokens) < n:
            return []
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    
    def calculate_bleu(pred_tokens, target_tokens, max_n=4):
        if len(pred_tokens) == 0 or len(target_tokens) == 0:
            return 0.0
        
        precisions = []
        for n in range(1, min(max_n + 1, len(pred_tokens) + 1)):
            pred_ngrams = Counter(get_ngrams(pred_tokens, n))
            target_ngrams = Counter(get_ngrams(target_tokens, n))
            
            if len(pred_ngrams) == 0:
                precisions.append(0.0)
                continue
            
            matches = sum((pred_ngrams & target_ngrams).values())
            total = sum(pred_ngrams.values())
            precision = matches / total if total > 0 else 0.0
            precisions.append(precision)
        
        # Brevity penalty
        if len(pred_tokens) == 0:
            bp = 0.0
        elif len(pred_tokens) < len(target_tokens):
            bp = math.exp(1 - len(target_tokens) / len(pred_tokens))
        else:
            bp = 1.0
        
        # Geometric mean of precisions
        if precisions and all(p > 0 for p in precisions):
            geo_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
            score = bp * geo_mean
        else:
            score = 0.0
        
        return score
    
    total_score = 0.0
    count = 0
    total_errors = 0
    total_chars = 0
    
    for pred, target in zip(predictions, targets):
        # Handle tensors
        if hasattr(pred, 'cpu'):
            pred = pred.cpu()
        if hasattr(target, 'cpu'):
            target = target.cpu()
        
        # Convert to lists
        if hasattr(pred, 'tolist'):
            pred_tokens = pred.tolist()
        else:
            pred_tokens = list(pred)
        
        if hasattr(target, 'tolist'):
            target_tokens = target.tolist()
        else:
            target_tokens = list(target)
        
        # Remove special tokens
        pred_clean = [t for t in pred_tokens if t not in [0, 1, 2, 3]]
        target_clean = [t for t in target_tokens if t not in [0, 1, 2, 3]]
        
        if len(pred_clean) > 0 and len(target_clean) > 0:
            # Calculate BLEU score
            bleu = calculate_bleu(pred_clean, target_clean)
            total_score += bleu
            count += 1
            
            # Calculate CER (Character Error Rate)
            pred_text = tokenizer.decode(pred_clean)
            target_text = tokenizer.decode(target_clean)
            
            if len(pred_text) > 0 and len(target_text) > 0:
                errors = edit_distance(pred_text, target_text)
                total_errors += errors
                total_chars += len(target_text)
    
    # Return BLEU score (not CER)
    return total_score / count if count > 0 else 0.0

def calculate_accuracy(predictions, targets, tokenizer):
    """Calculate token-level and sequence-level accuracy"""
    total_tokens = 0
    correct_tokens = 0
    total_sequences = 0
    correct_sequences = 0
    
    for pred, target in zip(predictions, targets):
        # Convert to lists and remove special tokens
        if hasattr(pred, 'tolist'):
            pred_tokens = pred.tolist()
        else:
            pred_tokens = list(pred)
        
        if hasattr(target, 'tolist'):
            target_tokens = target.tolist()
        else:
            target_tokens = list(target)
        
        # Remove special tokens (padding, start, end, unknown)
        pred_clean = [t for t in pred_tokens if t not in [0, 1, 2, 3]]
        target_clean = [t for t in target_tokens if t not in [0, 1, 2, 3]]
        
        if len(pred_clean) > 0 and len(target_clean) > 0:
            # Token-level accuracy
            min_len = min(len(pred_clean), len(target_clean))
            max_len = max(len(pred_clean), len(target_clean))
            
            # Count matching tokens up to the minimum length
            matches = sum(1 for i in range(min_len) if pred_clean[i] == target_clean[i])
            correct_tokens += matches
            total_tokens += max_len  # Use max length to penalize length differences
            
            # Sequence-level accuracy (exact match)
            if pred_clean == target_clean:
                correct_sequences += 1
            total_sequences += 1
    
    token_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
    sequence_accuracy = correct_sequences / total_sequences if total_sequences > 0 else 0.0
    
    return {
        'token_accuracy': token_accuracy,
        'sequence_accuracy': sequence_accuracy
    }

def train_epoch(model, train_loader, optimizer, criterion, teacher_forcing_ratio=0.5, scaler=None):
    """Train one epoch with optional AMP"""
    model.train()
    total_loss = 0
    use_amp = scaler is not None and torch.cuda.is_available()
    
    for batch in train_loader:
        urdu_seq = batch['urdu'].to(device)
        roman_seq = batch['roman'].to(device)
        
        decoder_target = roman_seq[:, 1:]
        
        optimizer.zero_grad()
        
        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(urdu_seq, roman_seq, teacher_forcing_ratio)
                loss = criterion(outputs.reshape(-1, outputs.size(-1)), decoder_target.reshape(-1))
            scaler.scale(loss).backward()
            # Unscale before clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(urdu_seq, roman_seq, teacher_forcing_ratio)
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), decoder_target.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def evaluate_model(model, data_loader, criterion, roman_tokenizer):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in data_loader:
            urdu_seq = batch['urdu'].to(device)
            roman_seq = batch['roman'].to(device)
            
            decoder_target = roman_seq[:, 1:]
            
            # Loss calculation with teacher forcing
            outputs_tf = model(urdu_seq, roman_seq, teacher_forcing_ratio=1.0)
            loss = criterion(outputs_tf.reshape(-1, outputs_tf.size(-1)), decoder_target.reshape(-1))
            non_pad_tokens = (decoder_target != 0).sum().item()
            
            total_loss += loss.item() * non_pad_tokens
            total_tokens += non_pad_tokens
            
            # Inference predictions using decoding (beam/greedy)
            infer_outputs = model(urdu_seq)
            pred_tokens = infer_outputs.argmax(dim=-1)
            # Convert to list of tensors per batch item
            if pred_tokens.dim() == 1:
                pred_list = [pred_tokens]
            else:
                pred_list = [pred_tokens[i] for i in range(pred_tokens.size(0))]
            for i in range(len(pred_list)):
                predictions.append(pred_list[i].cpu())
                targets.append(decoder_target[i].cpu())

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    bleu = calculate_bleu_score(predictions, targets, roman_tokenizer)
    perplexity = math.exp(avg_loss) if avg_loss != float('inf') else float('inf')
    cer = calculate_cer(predictions, targets, roman_tokenizer)
    accuracy_metrics = calculate_accuracy(predictions, targets, roman_tokenizer)
    
    return {
        'loss': avg_loss,
        'bleu': bleu,
        'perplexity': perplexity,
        'cer': cer,
        'token_accuracy': accuracy_metrics['token_accuracy'],
        'sequence_accuracy': accuracy_metrics['sequence_accuracy']
    }

def translate_text(model, text, urdu_tokenizer, roman_tokenizer):
    """Translate a single text"""
    model.eval()
    
    tokens = urdu_tokenizer.encode(text, out_type=int)
    input_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        predicted_tokens = output.argmax(dim=-1).squeeze().cpu().tolist()
        
        # Truncate prediction at first EOS token if present (EOS id = 3)
        try:
            eos_index = predicted_tokens.index(3)
            predicted_tokens = predicted_tokens[:eos_index]
        except ValueError:
            pass
        
        # Remove special tokens except EOS (PAD=0, UNK=1, BOS=2)
        clean_tokens = [t for t in predicted_tokens if t not in [0, 1, 2]]
        
        translated_text = roman_tokenizer.decode(clean_tokens)
    
    return translated_text

def run_experiment(config, splits, urdu_tokenizer, roman_tokenizer):
    """Run a single experiment with given configuration"""
    print(f"\n{'='*50}")
    print(f"Running experiment: {config['name']}")
    print(f"Config: {config}")
    print(f"{'='*50}")
    
    test_results = None  # Initialize test_results to avoid UnboundLocalError
    
    try:
        # Optional augmentation on training split
        aug_ratio = config.get('augment_ratio', 0.2)
        train_urdu_aug, train_roman_aug = augment_pairs(
            splits['train']['urdu'], splits['train']['roman'], urdu_tokenizer, roman_tokenizer, ratio=aug_ratio
        )
        
        # Create datasets
        train_dataset = TranslationDataset(
            train_urdu_aug, train_roman_aug,
            urdu_tokenizer, roman_tokenizer
        )
        val_dataset = TranslationDataset(
            splits['val']['urdu'], splits['val']['roman'], 
            urdu_tokenizer, roman_tokenizer
        )
        test_dataset = TranslationDataset(
            splits['test']['urdu'], splits['test']['roman'], 
            urdu_tokenizer, roman_tokenizer
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                                 shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                               shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], 
                                shuffle=False, collate_fn=collate_fn)
        
        # Initialize model
        model = Seq2SeqModel(
            urdu_vocab_size=urdu_tokenizer.get_piece_size(),
            roman_vocab_size=roman_tokenizer.get_piece_size(),
            embed_dim=config['embed_dim'],
            hidden_dim=config['hidden_dim'],
            dropout=config['dropout'],
            decoder_word_dropout=config.get('decoder_word_dropout', 0.1)
        ).to(device)
        
        # Initialize optimizer and criterion
        base_lr = config['learning_rate']
        optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)
        try:
            criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
        except TypeError:
            # Fallback if torch version doesn't support label_smoothing
            criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        # AMP scaler
        scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
        
        # LR scheduler on validation loss
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
        warmup_epochs = config.get('warmup_epochs', 2)
        
        # Training loop
        best_val_bleu = 0
        patience = 5
        patience_counter = 0
        
        train_losses = []
        val_metrics = []
        
        for epoch in range(config['epochs']):
            # Warmup learning rate
            if epoch < warmup_epochs:
                warmup_factor = float(epoch + 1) / float(max(1, warmup_epochs))
                for pg in optimizer.param_groups:
                    pg['lr'] = base_lr * warmup_factor
            current_lr = optimizer.param_groups[0]['lr']
            
            # Scheduled teacher forcing: decay towards 0.1
            base_tf = config.get('teacher_forcing_ratio', 0.5)
            tf_ratio_epoch = max(0.1, base_tf * (1.0 - (epoch / max(1, config['epochs']))))
            
            train_loss = train_epoch(model, train_loader, optimizer, criterion, tf_ratio_epoch, scaler)
            train_losses.append(train_loss)
            
            val_results = evaluate_model(model, val_loader, criterion, roman_tokenizer)
            val_metrics.append(val_results)
            
            # Step scheduler with validation loss after warmup
            if epoch + 1 > warmup_epochs:
                scheduler.step(val_results['loss'])
            
            print(f"Epoch {epoch+1}/{config['epochs']}")
            print(f"Train Loss: {train_loss:.4f} (TF={tf_ratio_epoch:.2f}, LR={current_lr:.6f})")
            print(f"Val Loss: {val_results['loss']:.4f}, BLEU: {val_results['bleu']:.4f}, "
                  f"Perplexity: {val_results['perplexity']:.2f}, CER: {val_results['cer']:.4f}")
            print(f"Token Accuracy: {val_results['token_accuracy']:.4f}, "
                  f"Sequence Accuracy: {val_results['sequence_accuracy']:.4f}")
            
            # Early stopping by BLEU
            if val_results['bleu'] > best_val_bleu:
                best_val_bleu = val_results['bleu']
                patience_counter = 0
                torch.save(model.state_dict(), f"best_model_{config['name']}.pth")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model for testing
        try:
            model.load_state_dict(torch.load(f"best_model_{config['name']}.pth"))
            
            # Test evaluation
            test_results = evaluate_model(model, test_loader, criterion, roman_tokenizer)
            
            print(f"\nFinal Test Results for {config['name']}:")
            print(f"Test Loss: {test_results['loss']:.4f}")
            print(f"Test BLEU: {test_results['bleu']:.4f}")
            print(f"Test Perplexity: {test_results['perplexity']:.2f}")
            print(f"Test CER: {test_results['cer']:.4f}")
            print(f"Test Token Accuracy: {test_results['token_accuracy']:.4f}")
            print(f"Test Sequence Accuracy: {test_results['sequence_accuracy']:.4f}")
            
            # Sample translations
            print(f"\nSample Translations for {config['name']}:")
            sample_texts = splits['test']['urdu'][:5]
            for i, urdu_text in enumerate(sample_texts):
                translation = translate_text(model, urdu_text, urdu_tokenizer, roman_tokenizer)
                actual = splits['test']['roman'][i]
                print(f"Urdu: {urdu_text}")
                print(f"Predicted: {translation}")
                print(f"Actual: {actual}")
                print("-" * 50)
                
        except Exception as e:
            print(f"Error during model loading or testing: {e}")
            # Create default test results if testing fails
            test_results = {
                'loss': float('inf'),
                'bleu': 0.0,
                'perplexity': float('inf'),
                'cer': 1.0,
                'token_accuracy': 0.0,
                'sequence_accuracy': 0.0
            }
    
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        
        # Return default values if experiment fails completely
        return {
            'config': config,
            'train_losses': [],
            'val_metrics': [],
            'test_results': {
                'loss': float('inf'),
                'bleu': 0.0,
                'perplexity': float('inf'),
                'cer': 1.0,
                'token_accuracy': 0.0,
                'sequence_accuracy': 0.0
            },
            'best_val_bleu': 0.0
        }
    
    # Ensure test_results is never None
    if test_results is None:
        test_results = {
            'loss': float('inf'),
            'bleu': 0.0,
            'perplexity': float('inf'),
            'cer': 1.0,
            'token_accuracy': 0.0,
            'sequence_accuracy': 0.0
        }
    
    return {
        'config': config,
        'train_losses': train_losses,
        'val_metrics': val_metrics,
        'test_results': test_results,
        'best_val_bleu': best_val_bleu
    }


# ===================== MAIN FUNCTION =====================

def main():
    """Main function to run all experiments"""
    print("Starting Urdu to Roman Transliteration Experiments")
    print("=" * 60)
    
    # Load and preprocess data
    print("Loading data...")
    processor = UrduRomanDataProcessor("dataset")
    processor.load_data()
    processor.preprocess_data()
    
    # Create train/val/test splits
    print("Creating data splits...")
    splits = processor.split_data()
    
    print(f"Train samples: {len(splits['train']['urdu'])}")
    print(f"Validation samples: {len(splits['val']['urdu'])}")
    print(f"Test samples: {len(splits['test']['urdu'])}")
    
    # Create tokenizers
    print("Creating tokenizers...")
    urdu_tokenizer, roman_tokenizer = create_tokenizers(
        splits['train']['urdu'] + splits['val']['urdu'], 
        splits['train']['roman'] + splits['val']['roman']
    )
    
    # Interactive teacher forcing prompt
    print("\nTeacher Forcing Setup")
    use_tf_input = input("Use teacher forcing during training? (y/n, default y): ").strip().lower()
    if use_tf_input == 'n':
        tf_ratio = 0.0
        print("Teacher forcing disabled (ratio = 0.0)")
    else:
        tf_ratio_input = input("Enter teacher forcing ratio [0.0-1.0] (default 0.5): ").strip()
        try:
            tf_ratio = float(tf_ratio_input) if tf_ratio_input else 0.5
        except ValueError:
            print("Invalid input. Defaulting teacher forcing ratio to 0.5")
            tf_ratio = 0.5
        tf_ratio = max(0.0, min(1.0, tf_ratio))
        print(f"Using teacher forcing ratio: {tf_ratio}")
    
    # Define experiment configurations
    configs = [
        {
            'name': 'baseline',
            'embed_dim': 128,
            'hidden_dim': 256,
            'dropout': 0.1,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 15,
            'teacher_forcing_ratio': tf_ratio,
            'decoder_word_dropout': 0.1,
            'max_length': 60,
            'warmup_epochs': 2
        }
    ]
    
    # Run experiments
    results = []
    for config in configs:
        try:
            result = run_experiment(config, splits, urdu_tokenizer, roman_tokenizer)
            # Only add results that have both test_results and config
            if result and 'test_results' in result and 'config' in result:
                results.append(result)
            else:
                print(f"Warning: Experiment {config['name']} returned incomplete results")
        except Exception as e:
            print(f"Error running experiment {config['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    
    if not results:
        print("No experiments completed successfully.")
        return
    
    print(f"{'Experiment':<20} {'BLEU':<8} {'CER':<8} {'Perplexity':<12} {'Token Acc':<10} {'Seq Acc':<10}")
    print("-" * 80)
    
    for result in results:
        config = result.get('config', {})
        test_results = result.get('test_results', {})
        
        name = config.get('name', 'Unknown')
        bleu = test_results.get('bleu', 0.0)
        cer = test_results.get('cer', 1.0)
        perplexity = test_results.get('perplexity', float('inf'))
        token_acc = test_results.get('token_accuracy', 0.0)
        seq_acc = test_results.get('sequence_accuracy', 0.0)
        
        # Handle infinite perplexity for display
        perp_str = f"{perplexity:.2f}" if perplexity != float('inf') else "inf"
        
        print(f"{name:<20} {bleu:<8.4f} {cer:<8.4f} {perp_str:<12} {token_acc:<10.4f} {seq_acc:<10.4f}")
    
    # Find best model
    if results:
        best_result = max(results, key=lambda x: x.get('test_results', {}).get('bleu', 0))
        best_config = best_result.get('config', {})
        best_test = best_result.get('test_results', {})
        
        print(f"\nBest Model: {best_config.get('name', 'Unknown')}")
        print(f"Best BLEU Score: {best_test.get('bleu', 0.0):.4f}")
        print(f"Best CER: {best_test.get('cer', 1.0):.4f}")
        print(f"Best Token Accuracy: {best_test.get('token_accuracy', 0.0):.4f}")
        print(f"Best Sequence Accuracy: {best_test.get('sequence_accuracy', 0.0):.4f}")
    
    print("\nExperiments completed!")

if __name__ == "__main__":
    main()

# ===================== DATA AUGMENTATION =====================
class NoiseInjector:
    """Injects random noise into Roman Urdu text for augmentation"""
    def __init__(self, swap_prob=0.05, delete_prob=0.03, insert_prob=0.03):
        self.swap_prob = swap_prob
        self.delete_prob = delete_prob
        self.insert_prob = insert_prob
        self.alphabet = list("abcdefghijklmnopqrstuvwxyz ',-?!")
    
    def augment(self, text: str) -> str:
        chars = list(text)
        # Random swaps
        i = 0
        while i < len(chars) - 1:
            if random.random() < self.swap_prob:
                chars[i], chars[i+1] = chars[i+1], chars[i]
                i += 2
            else:
                i += 1
        # Random deletions
        chars = [c for c in chars if not (random.random() < self.delete_prob and c != ' ')]
        # Random insertions
        i = 0
        out = []
        while i < len(chars):
            out.append(chars[i])
            if random.random() < self.insert_prob:
                out.append(random.choice(self.alphabet))
            i += 1
        return ''.join(out)

class BackTransliterator:
    """Generate synthetic Urdu->Roman pairs by back-transliterating Roman to pseudo-Urdu and then back."""
    def __init__(self, urdu_tokenizer: spm.SentencePieceProcessor, roman_tokenizer: spm.SentencePieceProcessor):
        self.urdu_tokenizer = urdu_tokenizer
        self.roman_tokenizer = roman_tokenizer
        # Simple mapping for common sequences (placeholder heuristic)
        self.map = {
            'kh': 'خ', 'gh': 'غ', 'sh': 'ش', 'ch': 'چ', 'th': 'تھ', 'ph': 'پھ',
            'aa': 'ا', 'ii': 'ی', 'oo': 'و', 'ye': 'ے', 'ka': 'ک', 'ki': 'کی', 'ka ': 'کا ',
        }
    
    def roman_to_pseudo_urdu(self, text: str) -> str:
        out = text
        for k, v in self.map.items():
            out = out.replace(k, v)
        return out
    
    def back_transliterate_pair(self, urdu_text: str, roman_text: str) -> Tuple[str, str]:
        # Noise on roman, then pseudo urdu from roman, pair with original roman
        noisy_roman = roman_text
        pseudo_urdu = self.roman_to_pseudo_urdu(noisy_roman)
        return pseudo_urdu, roman_text

# Hook augmentation into preprocessing pipeline
def augment_pairs(urdu_texts: List[str], roman_texts: List[str], urdu_tokenizer, roman_tokenizer, ratio: float = 0.2):
    injector = NoiseInjector()
    backtrans = BackTransliterator(urdu_tokenizer, roman_tokenizer)
    n = len(urdu_texts)
    k = max(1, int(n * ratio))
    indices = random.sample(range(n), k)
    aug_urdu = []
    aug_roman = []
    for i in indices:
        u = urdu_texts[i]
        r = roman_texts[i]
        # two augmented variants: noise on roman and back-transliteration
        aug_roman.append(injector.augment(r))
        pu, rr = backtrans.back_transliterate_pair(u, r)
        aug_urdu.append(pu)
        aug_roman.append(rr)
        aug_urdu.append(u)
    return urdu_texts + aug_urdu, roman_texts + aug_roman
