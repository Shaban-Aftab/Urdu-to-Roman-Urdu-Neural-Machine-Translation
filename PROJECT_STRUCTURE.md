# Project Structure Documentation

This document provides a detailed overview of the project structure, file descriptions, and code organization for the Urdu to Roman Urdu Neural Machine Translation project.



## 📓 Jupyter Notebooks

### 1. `22F-3275_BCS-7A_lstm-with-attention.ipynb`
**Purpose**: Complete implementation of BiLSTM encoder-decoder with attention mechanism

**Key Components**:
- Data preprocessing and tokenization
- BiLSTM encoder (2 layers, bidirectional)
- LSTM decoder (4 layers) with attention
- Training loop with teacher forcing
- Evaluation metrics (BLEU, Perplexity, CER)
- Beam search decoding
- Visualization and analysis

**Architecture**:
```python
Encoder: BiLSTM(2 layers) → hidden_states
Attention: Attention(hidden_states, decoder_hidden) → context
Decoder: LSTM(4 layers) + context → output_logits
```

**Usage**:
- Educational purposes and experimentation
- Complete end-to-end pipeline demonstration
- Baseline model with attention mechanism

### 2. `22F-3275_BCS-7A_lstm-and-xlstm-without-attention.ipynb`
**Purpose**: Implementation of BiLSTM without attention and xLSTM variants

**Key Components**:
- BiLSTM encoder with fixed context projection
- LSTM decoder without attention mechanism
- xLSTM implementation (modern LSTM variant)
- Subword regularization techniques
- Advanced decoding strategies

**Architecture**:
```python
Encoder: BiLSTM(2 layers) → mean_pooled_context
Decoder: LSTM(4 layers) + fixed_context → output_logits
Alternative: xLSTM components for improved performance
```

**Usage**:
- Comparison with attention-based models
- Exploration of modern LSTM variants
- Performance optimization studies

## 🐍 Python Scripts

### 1. `nig3.py` - Optimized BiLSTM Implementation
**Purpose**: Production-ready BiLSTM model with advanced features

**Key Features**:
- Subword regularization during training
- Per-sample beam search evaluation
- Gradient clipping and mixed precision training
- Comprehensive evaluation metrics
- Modular and extensible design

**Main Classes**:
```python
class BiLSTMEncoder(nn.Module)      # 2-layer bidirectional encoder
class LSTMDecoder(nn.Module)        # 4-layer decoder with fixed context
class Seq2SeqModel(nn.Module)       # Complete model wrapper
class TranslationDataset(Dataset)   # PyTorch dataset with augmentation
```

**Configuration Options**:
- `vocab_size`: Tokenizer vocabulary size
- `embed_dim`: Embedding dimensions
- `hidden_dim`: LSTM hidden dimensions
- `sp_nbest`: Subword regularization parameter
- `sp_alpha`: Sampling temperature
- `teacher_forcing_ratio`: Training strategy

### 2. `nig4.py` - xLSTM with Data Augmentation
**Purpose**: Advanced model with xLSTM architecture and comprehensive augmentation

**Key Features**:
- xLSTM encoder and decoder components
- Multiple data augmentation strategies
- Noise injection and back-transliteration
- Advanced beam search with penalties
- Configurable augmentation ratios

**Main Classes**:
```python
class xLSTMEncoder(nn.Module)       # Modern LSTM encoder
class xLSTMDecoder(nn.Module)       # Modern LSTM decoder
class NoiseInjector                 # Character-level noise augmentation
class BackTransliterator           # Synthetic data generation
class Seq2SeqModel(nn.Module)       # Complete xLSTM model
```

**Augmentation Features**:
- Subword regularization
- Character swapping/deletion/insertion
- Synthetic pair generation
- Configurable augmentation ratios

### 3. `streamlit_app.py` - Web Interface
**Purpose**: Interactive web application for real-time translation

**Features**:
- Real-time Urdu to Roman translation
- Model selection interface
- Translation confidence scores
- Batch translation support
- Export functionality

**Components**:
- Model loading and caching
- Text preprocessing pipeline
- Interactive UI components
- Result visualization

## 📊 Data Organization

### Dataset Structure
```
dataset/
├── poet-name/
│   ├── ur/          # Urdu text files
│   ├── en/          # Roman Urdu files
│   └── hi/          # Hindi files (unused)
```

### Processed Data
```
processed_data/
├── train.tsv        # Training split (50%)
├── val.tsv          # Validation split (25%)
└── test.tsv         # Test split (25%)
```

### Tokenizers
```
tokenizers/
├── urdu_tokenizer.model     # SentencePiece Urdu model
├── urdu_tokenizer.vocab     # Urdu vocabulary
├── roman_tokenizer.model    # SentencePiece Roman model
├── roman_tokenizer.vocab    # Roman vocabulary
├── urdu_train.txt          # Training text for Urdu tokenizer
└── roman_train.txt         # Training text for Roman tokenizer
```

## 🔧 Configuration Files

### `requirements.txt`
Lists all Python dependencies with version specifications:
- Core ML libraries (PyTorch, NumPy, SciPy)
- NLP tools (SentencePiece, NLTK)
- Data processing (Pandas, Scikit-learn)
- Visualization (Matplotlib, Seaborn)
- Web interface (Streamlit)
- Development tools (Jupyter, pytest)

### `.gitignore`
Excludes from version control:
- Python cache files (`__pycache__/`)
- Trained models (`*.pth`, `*.model`)
- Data files (`*.tsv`, `*.csv`)
- IDE files (`.vscode/`, `.idea/`)
- OS files (`.DS_Store`, `Thumbs.db`)
- Virtual environments (`venv/`, `libs/`)

## 🏗️ Code Architecture

### Model Architecture Hierarchy
```
Seq2SeqModel
├── Encoder (BiLSTM/xLSTM)
│   ├── Embedding Layer
│   ├── LSTM/xLSTM Layers
│   └── Dropout
├── Decoder (LSTM/xLSTM)
│   ├── Embedding Layer
│   ├── Context Projection
│   ├── LSTM/xLSTM Layers
│   └── Output Projection
└── Decoding Strategies
    ├── Greedy Decoding
    ├── Beam Search
    └── Length Normalization
```

### Data Pipeline
```
Raw Text → Preprocessing → Tokenization → Dataset → DataLoader → Model
    ↓
Augmentation ← SentencePiece ← Cleaning ← Loading ← File System
```

### Training Pipeline
```
Model → Forward Pass → Loss Calculation → Backpropagation → Optimization
  ↑                                                              ↓
Evaluation ← Validation ← Metrics ← Decoding ← Gradient Clipping
```

## 📈 Performance Monitoring

### Metrics Tracked
- **BLEU Score**: Translation quality
- **Perplexity**: Language model confidence
- **Character Error Rate**: Character-level accuracy
- **Training Loss**: Model convergence
- **Validation Loss**: Overfitting detection

### Logging and Visualization
- Training progress tracking
- Loss curve visualization
- Sample translation outputs
- Model performance comparisons

## 🔄 Development Workflow

### 1. Data Preparation
```bash
# Load and preprocess data
python -c "from nig4 import UrduRomanDataProcessor; processor = UrduRomanDataProcessor('dataset'); processor.load_data(); processor.preprocess_data()"
```

### 2. Model Training
```bash
# Train BiLSTM model
python nig3.py

# Train xLSTM model with augmentation
python nig4.py
```

### 3. Evaluation
```bash
# Run evaluation on test set
python -c "from nig4 import evaluate_model; evaluate_model(model, test_loader, tokenizers)"
```

### 4. Interactive Testing
```bash
# Launch web interface
streamlit run streamlit_app.py
```

## 🧪 Testing and Validation

### Unit Tests
- Model component testing
- Data preprocessing validation
- Tokenization consistency checks
- Evaluation metric verification

### Integration Tests
- End-to-end pipeline testing
- Model training validation
- Web interface functionality
- Performance benchmarking

## 📝 Code Style and Standards

### Python Standards
- PEP 8 compliance
- Type hints for function signatures
- Comprehensive docstrings
- Modular design principles

### Documentation Standards
- Clear function and class documentation
- Usage examples in docstrings
- Configuration parameter descriptions
- Error handling documentation

## 🚀 Deployment Considerations

### Model Serving
- Model serialization and loading
- Inference optimization
- Batch processing support
- API endpoint design

### Scalability
- GPU memory management
- Batch size optimization
- Model quantization options
- Distributed training support

---

This project structure is designed for:
- **Modularity**: Easy to extend and modify
- **Reproducibility**: Clear dependencies and configurations
- **Maintainability**: Well-documented and organized code
- **Scalability**: Ready for production deployment

For questions about the project structure or to suggest improvements, please open an issue in the repository.