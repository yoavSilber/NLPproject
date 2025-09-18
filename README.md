# Hebrew Biblical Text Analysis & Classification 🔬📜

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Machine Learning](https://img.shields.io/badge/ML-scikit--learn-orange.svg)](https://scikit-learn.org/)
[![NLP](https://img.shields.io/badge/NLP-transformers-green.svg)](https://huggingface.co/transformers/)
[![Hebrew](https://img.shields.io/badge/Language-Hebrew-purple.svg)]()

A sophisticated Natural Language Processing and Machine Learning project for analyzing and classifying Hebrew biblical texts (Torah/Tanach) using multiple data sources, advanced linguistic features, and state-of-the-art transformer models.

## 🎯 Project Overview

This project implements a comprehensive system for computational biblical studies, combining:

- **Multi-source XML parsing** from authoritative biblical databases
- **Advanced Hebrew NLP** using DictaBERT transformer models
- **Syntactic analysis** with dependency and constituency parsing
- **Traditional Jewish text analysis** through teamim (cantillation marks) processing
- **Machine learning classification** for biblical authorship attribution

The system can classify biblical verses by:

1. **Source book** (Genesis, Exodus, Leviticus, Numbers, Deuteronomy)
2. **Documentary Hypothesis sources** (J, E, P, R, D - traditional biblical scholarship)

## ✨ Key Features

### 🔤 **Advanced Hebrew Text Processing**

- Integration with **DictaBERT** (Hebrew transformer model)
- Removal of diacritics and cantillation marks for analysis
- Tokenization and embedding generation for Hebrew texts

### 🌳 **Multi-Level Linguistic Analysis**

- **Dependency tree parsing** with syntactic relationships
- **Constituency tree analysis** for phrase structure
- **Teamim tree construction** based on traditional cantillation hierarchy
- **Lexical diversity** and morphological feature extraction

### 📊 **Comprehensive Feature Engineering**

- **TF-IDF vectorization** for word and lexical features
- **Syntactic features** from dependency structures
- **Prosodic features** from teamim (cantillation) patterns
- **Statistical text features** (word count, unique words, etc.)
- **Transformer embeddings** from Hebrew language models

### 🤖 **Machine Learning Pipeline**

- **Feature selection** using SelectKBest with f_classif
- **Complement Naive Bayes** classifier for robust performance
- **10-fold cross-validation** for reliable evaluation
- **Multiple evaluation metrics** (accuracy, precision, recall, F1)

### 📈 **Statistical Analysis**

- Comprehensive text statistics by book and DH source
- Word frequency analysis and lexical diversity metrics
- Syntactic complexity measurements
- Comparative analysis across different biblical sources

## 🗂️ Data Sources

The project integrates multiple authoritative biblical text databases:

### 📚 **Primary Text Sources**

- **TANACH.US**: Unicode/XML Leningrad Codex with morphological data
- **SHEBANQ**: ETCBC database with syntactic annotations
- **DH_Markings**: Documentary Hypothesis source attributions (Friedman)

### 📋 **Auxiliary Data**

- **Teamim.xlsx**: Cantillation mark hierarchy and prosodic data
- **5,847 biblical verses** from the Torah (Five Books of Moses)

## 🛠️ Technical Architecture

```
Project Structure:
├── ProjectMain.py          # Main orchestration script
├── PasukClassifier.py      # ML classifier implementation
├── ProjectStatistics.py    # Statistical analysis module
├── TeamimTree.py          # Cantillation tree processor
├── requirements.txt       # Python dependencies
├── Teamim.xlsx           # Cantillation mark data
├── TANACH.US/            # Primary Hebrew text files
├── SHEBANQ/              # Syntactic annotation files
└── DH_Markings/          # Documentary Hypothesis data
```

### 🧠 **Core Components**

**PasukClassifier**: Machine learning pipeline with:

- Feature extraction from multiple linguistic levels
- Hebrew transformer model integration (DictaBERT)
- Cross-validation and performance evaluation

**TeamimTree**: Hierarchical prosodic structure analysis:

- Traditional Jewish cantillation mark processing
- Tree construction based on prosodic hierarchy
- Feature extraction for ML classification

**ProjectStatistics**: Comprehensive text analysis:

- Multi-dimensional statistical features
- Comparative analysis across sources
- Data visualization and reporting

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, for faster transformer processing)

### Installation Steps

1. **Clone the repository**

```bash
git clone <repository-url>
cd NLPproject-main
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Verify data files**
   Ensure all required data files are present:

- `Teamim.xlsx`
- XML files in `TANACH.US/`, `SHEBANQ/`, and `DH_Markings/` directories

## 🎮 Usage

### Basic Execution

```bash
python ProjectMain.py
```

### What the system does:

1. **Parses XML data** from multiple biblical databases
2. **Builds dependency trees** using Hebrew NLP models
3. **Constructs teamim trees** from cantillation data
4. **Extracts comprehensive features** for machine learning
5. **Trains classifiers** for book and source attribution
6. **Generates statistical reports** and analysis

### Output Files

- `analyzed_pasuks.json`: Detailed analysis of sample verses
- `pasuk_statistics.csv`: Comprehensive statistics dataset
- `classifier_pickle.pkl`: Trained model data
- `score.pkl`: Classification performance metrics

## 📊 Key Results & Performance

The system achieves robust classification performance using sophisticated feature engineering:

### 🎯 **Classification Targets**

- **Biblical Book Classification**: Genesis, Exodus, Leviticus, Numbers, Deuteronomy
- **Documentary Hypothesis Classification**: J, E, P, R, D sources

### 📈 **Feature Set**

- **25,000 selected features** using statistical feature selection
- **Multi-modal features**: textual, syntactic, prosodic, semantic
- **Transformer embeddings**: 768-dimensional Hebrew language representations

### 🔬 **Evaluation Methodology**

- **10-fold cross-validation** for robust performance estimation
- **Multiple metrics**: Accuracy, Precision, Recall, F1-score
- **Weighted averaging** for imbalanced classes

## 🔧 Advanced Features

### 🤖 **Machine Learning Techniques**

- **ComplementNB**: Robust classifier for imbalanced text data
- **Feature scaling**: MinMaxScaler for normalized inputs
- **Variance filtering**: Removes low-variance features
- **Cross-validation**: K-fold validation for reliable estimates

### 🌐 **Hebrew NLP Integration**

- **DictaBERT-tiny-joint**: Specialized Hebrew transformer model
- **Morphological analysis**: Leveraging Hebrew linguistic structure
- **Dependency parsing**: Syntactic relationship extraction

### 📊 **Statistical Analysis**

- **Lexical diversity**: Type-token ratios and vocabulary richness
- **Syntactic complexity**: Tree depth and phrase transitions
- **Prosodic analysis**: Cantillation pattern complexity
- **Comparative metrics**: Cross-source statistical comparison

## 🎓 Academic & Professional Value

This project demonstrates expertise in:

### 💻 **Technical Skills**

- **Advanced Python programming** with scientific libraries
- **Machine Learning** pipeline development and evaluation
- **Natural Language Processing** for non-Latin scripts
- **XML parsing** and data integration from multiple sources
- **Statistical analysis** and data visualization

### 🔬 **Domain Expertise**

- **Computational linguistics** for Hebrew text processing
- **Digital humanities** and biblical studies methodology
- **Traditional Jewish text analysis** integration with modern NLP
- **Cross-disciplinary research** combining technology and humanities

### 🏗️ **Software Engineering**

- **Modular architecture** with clean separation of concerns
- **Data pipeline** design for complex multi-source integration
- **Performance optimization** for large-scale text processing
- **Reproducible research** with proper documentation and structure

## 📚 Technologies Used

- **Python 3.8+**: Core programming language
- **scikit-learn**: Machine learning algorithms and evaluation
- **transformers (Hugging Face)**: Hebrew transformer models
- **pandas**: Data manipulation and analysis
- **numpy & scipy**: Numerical computing and statistics
- **torch**: Deep learning framework for transformers
- **lxml**: XML parsing and processing

## 🤝 Contributing

This project represents a sophisticated intersection of traditional biblical scholarship and modern computational methods. It showcases advanced capabilities in Hebrew NLP, multi-source data integration, and machine learning for humanities research.

## 📄 License

This project uses data from multiple sources with appropriate attribution:

- TANACH.US: Unicode/XML Leningrad Codex
- SHEBANQ: ETCBC database
- Documentary Hypothesis markings: R. E. Friedman research

## 🎯 For Recruiters

This project demonstrates:

- **Advanced NLP skills** with non-English languages
- **Machine learning expertise** with real-world data
- **Data engineering** capabilities with complex XML sources
- **Statistical analysis** and performance evaluation
- **Academic research** integration with practical implementation
- **Clean, maintainable code** with proper architecture

The combination of traditional scholarship with cutting-edge technology showcases both technical depth and interdisciplinary problem-solving abilities.
