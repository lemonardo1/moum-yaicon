# YAICON Dashboard

A web dashboard for visualizing and interacting with the YAICON drug sensitivity prediction model that uses triple-omics data (GEP, CNV, and MUT).

## Features

- **Model Architecture Visualization**: View the PASO_GEP_CNV_MUT model architecture and parameters
- **Data Explorer**: Browse and visualize the triple-omics datasets and drug sensitivity data
- **Results Visualization**: View model performance metrics and predictions
- **Interactive Visualizations**: Explore attention weights and prediction accuracy

## Setup and Installation

### Basic Installation

1. Install the core dependencies first:

```bash
pip install numpy
pip install -r requirements.txt
```

2. For machine learning functionality (optional, install only if you need prediction features):

```bash
pip install torch
pip install rdkit-pypi
pip install deepchem
```

### Alternative Installation Method

If you encounter issues with the dependencies, you can install them individually:

```bash
pip install numpy
pip install flask==2.0.1 werkzeug==2.0.1 jinja2==3.0.1
pip install pandas scikit-learn
pip install plotly
```

### Running the Dashboard

1. Run the Flask application:

```bash
python run.py
```

2. Open your browser and navigate to http://localhost:5000

## Dashboard Structure

- **Home**: Overview of the model and available folds
- **Model Architecture**: Detailed view of the model architecture and parameters
- **Data Explorer**: Browse and visualize the datasets
- **Results**: View model performance metrics and predictions

## Data Requirements

The dashboard expects the following data structure:

- `data/`: Contains the omics and drug sensitivity data
- `result/model/train_MixedSet_10Fold_GEP_CNV_MUT/`: Contains the model results for each fold

## About the Model

PASO_GEP_CNV_MUT is a deep learning model for drug sensitivity prediction that integrates:

- Drug molecular information (SMILES)
- Gene Expression Profiles (GEP)
- Copy Number Variations (CNV)
- Mutation Data (MUT)

The model uses attention mechanisms to learn interactions between drug molecules and omics data to predict IC50 values.
