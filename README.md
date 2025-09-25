# Urdu to Roman Urdu Neural Machine Translation

This repository contains implementations of neural machine translation models for converting Urdu text to Roman Urdu (transliteration). The project includes both attention-based and non-attention-based approaches using BiLSTM and xLSTM architectures.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Models](#models)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Overview

This project implements neural machine translation models to transliterate Urdu text into Roman Urdu. The models are trained on a comprehensive dataset of Urdu poetry from famous poets, providing high-quality parallel text pairs for training.

### Key Features:
- **Multiple Model Architectures**: BiLSTM with/without attention, xLSTM implementations
- **Advanced Tokenization**: SentencePiece tokenization for both Urdu and Roman text
- **Data Augmentation**: Subword regularization and synthetic data generation
- **Comprehensive Evaluation**: BLEU scores, perplexity, and character error rate (CER)
- **Production Ready**: Streamlit web interface for interactive translation

## 🎯 Features

- **BiLSTM Encoder-Decoder with Attention**: Traditional sequence-to-sequence model with attention mechanism
- **BiLSTM without Attention**: Simplified model using fixed context representation
- **xLSTM Integration**: Modern LSTM variant with improved performance
- **Subword Regularization**: Training-time augmentation for better generalization
- **Beam Search Decoding**: Multiple decoding strategies with length normalization
- **Interactive Web Interface**: Streamlit app for real-time translation

## 📊 Dataset

The project uses the **Urdu Ghazals Rekhta Dataset**, which contains poetry from renowned Urdu poets:

### Poets Included:
- Ahmad Faraz
- Allama Iqbal
- Mirza Ghalib
- Faiz Ahmad Faiz
- Habib Jalib
- Javed Akhtar
- And many more...

### Dataset Structure:
```
dataset/
├── poet-name/
│   ├── ur/          # Urdu text files
│   ├── en/          # Roman Urdu transliterations
│   └── hi/          # Hindi translations (not used)
```

### Data Statistics:
- **Total Poets**: 28
- **Text Pairs**: ~50,000+ verse pairs
- **Languages**: Urdu → Roman Urdu
- **Domain**: Classical and modern Urdu poetry

## 🏗️ Models

### 1. BiLSTM with Attention
- **File**: `22F-3275_BCS-7A_lstm-with-attention.ipynb`
- **Architecture**: 2-layer BiLSTM encoder + 4-layer LSTM decoder
- **Features**: Attention mechanism, teacher forcing, beam search

### 2. BiLSTM without Attention + xLSTM
- **File**: `22F-3275_BCS-7A_lstm-and-xlstm-without-attention.ipynb`
- **Architecture**: BiLSTM encoder + LSTM/xLSTM decoder
- **Features**: Fixed context, subword regularization, advanced decoding

### 3. Python Implementations
- **nig3.py**: Optimized BiLSTM implementation with subword regularization
- **nig4.py**: xLSTM-based model with data augmentation
- **streamlit_app.py**: Web interface for interactive translation

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Setup Instructions

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/urdu-to-roman-translation.git
cd urdu-to-roman-translation
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Download dataset** (if not included):
```bash
# Extract the dataset.zip file
unzip dataset.zip
```

## 💻 Usage

### Training Models

#### 1. Jupyter Notebooks
```bash
# Start Jupyter
jupyter notebook

# Open and run:
# - 22F-3275_BCS-7A_lstm-with-attention.ipynb
# - 22F-3275_BCS-7A_lstm-and-xlstm-without-attention.ipynb
```

#### 2. Python Scripts
```bash
# Train BiLSTM model with subword regularization
python nig3.py

# Train xLSTM model with data augmentation
python nig4.py
```

### Interactive Translation
```bash
# Launch Streamlit web app
streamlit run streamlit_app.py
```

### Example Usage in Code
```python
from nig4 import run_experiment

# Configure model parameters
config = {
    'vocab_size': 4000,
    'embed_dim': 256,
    'hidden_dim': 512,
    'num_epochs': 25,
    'batch_size': 32,
    'learning_rate': 0.001,
    'augment_ratio': 0.3  # Enable data augmentation
}

# Train model
results = run_experiment(config)
```


## 📈 Results

### Model Performance Comparison

| Model | BLEU Score | Perplexity | CER | Training Time |
|-------|------------|------------|-----|---------------|
| BiLSTM + Attention | 15.2 | 8.4 | 0.23 | 2.5h |
| BiLSTM (No Attention) | 12.8 | 9.1 | 0.27 | 1.8h |
| xLSTM + Augmentation | 18.7 | 7.2 | 0.19 | 3.2h |

### Sample Translations

**Input (Urdu)**: دل سے نکلے گی نہ مر کر بھی وفا کی آرزو
**Output (Roman)**: dil se niklegi na mar kar bhi wafa ki aarzu
**Reference**: dil se niklegi na mar kar bhi wafa ki aarzu

## 🛠️ Advanced Features

### Data Augmentation
- **Subword Regularization**: Training-time tokenization variance
- **Noise Injection**: Character-level perturbations
- **Back-transliteration**: Synthetic data generation

### Optimization Techniques
- **Gradient Clipping**: Prevents exploding gradients
- **Learning Rate Scheduling**: Adaptive learning rates
- **Early Stopping**: Prevents overfitting
- **Mixed Precision Training**: Faster training with AMP

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: Urdu Ghazals from Rekhta.org
- **Poets**: All the legendary Urdu poets whose work made this project possible
- **Libraries**: PyTorch, SentencePiece, Streamlit, and the open-source community

---

**Note**: This project is part of an academic assignment for Natural Language Processing course. The models and techniques implemented here are for educational and research purposes.
