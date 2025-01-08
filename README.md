# CyberAttackDetection-Python

![🔧 Build Status](https://github.com/canstralian/CyberAttackDetection-Python/actions/workflows/ci.yml/badge.svg)  
![📊 Coverage](https://codecov.io/gh/canstralian/CyberAttackDetection-Python/branch/main/graph/badge.svg)  
![📦 Dependencies](https://img.shields.io/librariesio/release/github/canstralian/CyberAttackDetection-Python)  
![📜 License](https://img.shields.io/github/license/canstralian/CyberAttackDetection-Python)  
![🕒 Last Commit](https://img.shields.io/github/last-commit/canstralian/CyberAttackDetection-Python)  
![🚀 Release](https://img.shields.io/github/v/release/canstralian/CyberAttackDetection-Python)  
![🐞 Issues](https://img.shields.io/github/issues/canstralian/CyberAttackDetection-Python)

## Overview
CyberAttackDetection-Python is a project aimed at detecting cyber attacks using machine learning models. This repository contains code for data preprocessing, model training, evaluation, and utilities.

## Directory Structure
```
CyberAttackDetection-Python/
├── .github/
│   └── workflows/
│       └── ci.yml
├── models/
│   └── random_forest_model.pkl
│   └── simple_nn_model.pth
├── data/
│   └── raw_data.csv
│   └── processed_data.csv
├── notebooks/
│   └── exploratory_data_analysis.ipynb
│   └── model_training.ipynb
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── utilities.py
├── tests/
│   └── test_data_preprocessing.py
│   └── test_model_training.py
│   └── test_model_evaluation.py
├── .gitignore
├── .replit
├── README.md
├── main.py
├── poetry.lock
├── pyproject.toml
└── requirements.txt
```

## Getting Started
To get started with this project, follow these steps:

1. Clone the repository:
   ```sh
   git clone https://github.com/canstralian/CyberAttackDetection-Python.git
   ```
2. Install the dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
### Data Preprocessing
Run the data preprocessing script:
```sh
python src/data_preprocessing.py
```

### Model Training
Train the models:
```sh
python src/model_training.py
```

### Model Evaluation
Evaluate the models:
```sh
python src/model_evaluation.py
```

## Contributing
Contributions are welcome! Please read the [contributing guidelines](CONTRIBUTING.md) for more information.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- [PyTorch](https://pytorch.org/)
- [scikit-learn](https://scikit-learn.org/)
