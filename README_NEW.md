# DNA Binding Protein Classification

A production-ready web application for classifying DNA binding proteins using machine learning and deep learning models.

## 🚀 Features

- **Multiple Model Types**: PseAAC, Physicochemical Properties, and CNN models
- **Web Interface**: User-friendly Streamlit application
- **Batch Processing**: Support for single sequences and FASTA files
- **Production Ready**: Optimized with warning suppression and error handling
- **Export Results**: Download predictions as CSV

## 📁 Project Structure

```
DNA Binding Protein Classification/
├── main_app.py                 # Main Streamlit application
├── app.py                      # Original application (legacy)
├── requirements.txt            # Python dependencies
├── README.md                  # This file
├── src/                       # Source code
│   └── utils/                 # Utility modules
│       ├── deployment_utils.py    # Production environment setup
│       ├── prediction_utils.py    # Model prediction functions
│       ├── feature_utils.py       # Feature extraction functions
│       └── data_utils.py          # Data processing functions
├── scripts/                   # Utility scripts
│   ├── fasta_to_csv.py           # Convert FASTA to CSV
│   └── create_sample.py          # Create balanced samples
├── models/                    # Trained models
│   ├── Traditional ML - PseAAC/
│   ├── Traditional ML - Physicochemical Properties/
│   └── CNN/
├── data/                      # Data files
│   ├── samples/               # Sample data
│   │   ├── sample.fasta       # 100 balanced sequences
│   │   └── sample.csv         # CSV version
│   ├── DNA_Train.csv          # Training data
│   ├── DNA_Test.csv           # Test data
│   ├── Train.fasta            # Training sequences
│   └── Test.fasta             # Test sequences
├── docs/                      # Documentation
│   ├── Project_Instruction.pdf
│   ├── Project_Report.pdf
│   └── *.md                   # Various documentation files
└── ML Models & Feature Extraction/  # Jupyter notebooks
    ├── Exploratory Data Analysis.ipynb
    ├── TF-IDF.ipynb
    ├── PSEAAC.ipynb
    ├── Physicochemical Properties.ipynb
    ├── CNN.ipynb
    └── Comparison Plot between Models.ipynb
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd "DNA Binding Protein Classification"
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run main_app.py
   ```

## 🔧 Requirements

- Python 3.8+
- Streamlit
- TensorFlow
- scikit-learn
- pandas
- numpy
- joblib
- biopython

## 📊 Available Models

### Traditional Machine Learning

#### PseAAC Models (Recommended)
- **Random Forest**: Best overall performance
- **SVM**: High accuracy for complex patterns
- **Logistic Regression**: Fast and interpretable
- **Decision Tree**: Good for feature understanding
- **Naive Bayes**: Probabilistic approach
- **KNN**: Instance-based learning

#### Physicochemical Properties
- Same algorithms as PseAAC
- Based on amino acid physicochemical properties
- Includes hydrophobic, hydrophilic, aromatic, aliphatic, charged, and polar features

### Deep Learning

#### CNN Models
- **CNN1**: Basic convolutional architecture
- **CNN2**: Enhanced CNN with deeper layers
- **ProtCNN1**: Protein-specific CNN design
- **ProtCNN2**: Advanced protein CNN architecture

## 🎯 Usage

### Web Interface

1. **Start the application**:
   ```bash
   streamlit run main_app.py
   ```

2. **Select a model** from the sidebar

3. **Input protein sequences**:
   - Text input: Enter sequences directly
   - File upload: Upload FASTA files

4. **Get predictions** with confidence scores

5. **Download results** as CSV for further analysis

### Command Line Scripts

#### Convert FASTA to CSV:
```bash
python scripts/fasta_to_csv.py input.fasta output.csv
```

#### Create balanced samples:
```bash
python scripts/create_sample.py input.fasta output.fasta --samples 100
```

## 🧬 Input Format

### Text Input
- Enter protein sequences one per line
- Only amino acid letters (A-Z) are accepted
- Invalid characters are automatically removed

### FASTA File
```
>Protein_1
MKVSQILPLAGAISVASGFWIPDFSNKQNSNSYPGQYKGKGGYQ
>Protein_2
MTQYHDEIVCTNGATRFCSKEMDFGQVRSLTQRQALQSAQTRSKT
```

## 📈 Model Performance

All models have been trained and validated on a comprehensive dataset of DNA binding proteins. The PseAAC models generally provide the best balance of accuracy and interpretability.

## 🔧 Production Features

- **Warning Suppression**: Clean output without TensorFlow/sklearn warnings
- **Error Handling**: Robust error handling for production deployment
- **Memory Optimization**: Efficient model loading and prediction
- **Containerization Ready**: Optimized for Docker deployment
- **Scalable Architecture**: Modular design for easy maintenance

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- BioPython for sequence processing
- Streamlit for the web interface
- TensorFlow and scikit-learn for machine learning capabilities
- Research papers on DNA binding protein classification

---

**Note**: This is a production-ready application with comprehensive error handling and optimization for deployment environments.
