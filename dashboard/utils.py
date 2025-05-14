import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import torch
from collections import defaultdict

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import project-specific modules
from models.model import PASO_GEP_CNV_MUT
from utils.utils import get_device, get_log_molar
from data.TripleOmics_Drug_Dataset import TripleOmics_Drug_dataset, custom_collate_fn

def load_model_params(fold_num=1):
    """Load model parameters from JSON file."""
    model_dir = f'../result/model/train_MixedSet_10Fold_GEP_CNV_MUT/Fold{fold_num}'
    params_file = os.path.join(model_dir, "TCGA_classifier_best_aucpr_GEP.json")
    
    if not os.path.exists(params_file):
        # Return default parameters if file doesn't exist
        return {
            'smiles_padding_length': 256,
            'smiles_embedding_size': 512,
            'number_of_pathways': 619,
            'smiles_attention_size': 64,
            'gene_attention_size': 1,
            'molecule_temperature': 1.0,
            'gene_temperature': 1.0,
            'molecule_gep_heads': [2],
            'molecule_cnv_heads': [2],
            'molecule_mut_heads': [2],
            'gene_heads': [1],
            'cnv_heads': [1],
            'mut_heads': [1],
            'n_heads': 2,
            'num_layers': 4,
            'omics_dense_size': 256,
            'stacked_dense_hidden_sizes': [1024, 512],
            'dropout': 0.5,
            'temperature': 1.0,
            'activation_fn': 'relu',
            'batch_norm': True,
            'drug_sensitivity_processing_parameters': {
                'parameters': {'max': 100, 'min': 0}
            },
            'loss_fn': 'mse'
        }
    
    with open(params_file, 'r') as f:
        params = json.load(f)
    
    return params

def get_available_folds():
    """Get list of available fold directories."""
    base_dir = '../result/model/train_MixedSet_10Fold_GEP_CNV_MUT'
    folds = []
    
    if os.path.exists(base_dir):
        for item in os.listdir(base_dir):
            if item.startswith('Fold') and os.path.isdir(os.path.join(base_dir, item)):
                try:
                    fold_num = int(item.replace('Fold', ''))
                    folds.append(fold_num)
                except ValueError:
                    continue
    
    # If no folds found, return default range
    if not folds:
        folds = list(range(1, 11))
    
    return sorted(folds)

def load_model(fold_num=1):
    """Load model with weights if available."""
    params = load_model_params(fold_num)
    model = PASO_GEP_CNV_MUT(params).to(get_device())
    
    # Try to load weights if available
    weights_path = f'../result/model/train_MixedSet_10Fold_GEP_CNV_MUT/Fold{fold_num}/weights'
    if os.path.exists(weights_path) and len(os.listdir(weights_path)) > 0:
        weight_files = [f for f in os.listdir(weights_path) if f.endswith('.pt')]
        if weight_files:
            latest_weight = sorted(weight_files)[-1]
            model.load(os.path.join(weights_path, latest_weight))
    
    return model

def get_data_files():
    """Get list of available data files."""
    data_dir = '../data'
    data_files = []
    
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.endswith('.csv') or file.endswith('.pkl'):
                data_files.append(file)
    
    return sorted(data_files)

def load_data_file(filename):
    """Load data from file."""
    file_path = os.path.join('../data', filename)
    
    if filename.endswith('.csv'):
        return pd.read_csv(file_path)
    elif filename.endswith('.pkl'):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    else:
        return None

def get_fold_data(fold_num=1):
    """Get training and testing data for a specific fold."""
    drug_sensitivity_filepath = '../data/10_fold_data/mixed/MixedSet_'
    smiles_filepath = '../data/CCLE-GDSC-SMILES.csv'
    gep_filepath = '../data/GEP_Wilcoxon_Test_Analysis_Log10_P_value_C2_KEGG_MEDICUS.csv'
    cnv_filepath = '../data/CNV_Cardinality_Analysis_of_Variance_C2_KEGG_MEDICUS.csv'
    mut_filepath = '../data/MUT_Cardinality_Analysis_of_Variance_C2_KEGG_MEDICUS.csv'
    
    # Check if files exist
    train_file = f"{drug_sensitivity_filepath}train_Fold{fold_num-1}.csv"
    test_file = f"{drug_sensitivity_filepath}test_Fold{fold_num-1}.csv"
    
    if not (os.path.exists(train_file) and os.path.exists(test_file)):
        return None, None
    
    # Create datasets
    train_dataset = TripleOmics_Drug_dataset(
        drug_sensitivity_filepath=train_file,
        smiles_filepath=smiles_filepath,
        gep_filepath=gep_filepath,
        cnv_filepath=cnv_filepath,
        mut_filepath=mut_filepath,
        gep_standardize=True,
        cnv_standardize=True,
        mut_standardize=True,
        drug_sensitivity_min_max=True,
        column_names=('drug', 'cell_line', 'IC50')
    )
    
    test_dataset = TripleOmics_Drug_dataset(
        drug_sensitivity_filepath=test_file,
        smiles_filepath=smiles_filepath,
        gep_filepath=gep_filepath,
        cnv_filepath=cnv_filepath,
        mut_filepath=mut_filepath,
        gep_standardize=True,
        cnv_standardize=True,
        mut_standardize=True,
        drug_sensitivity_min_max=True,
        column_names=('drug', 'cell_line', 'IC50')
    )
    
    return train_dataset, test_dataset

def get_model_summary(fold_num=1):
    """Generate a summary of the model architecture."""
    params = load_model_params(fold_num)
    
    # Extract key architecture details
    summary = {
        'input_features': {
            'drug': 'SMILES representation (padded to length ' + str(params.get('smiles_padding_length', 256)) + ')',
            'gep': str(params.get('number_of_pathways', 619)) + ' pathways',
            'cnv': str(params.get('number_of_pathways', 619)) + ' pathways',
            'mut': str(params.get('number_of_pathways', 619)) + ' pathways'
        },
        'embedding': {
            'drug': 'Graph Neural Network with embedding size ' + str(params.get('smiles_embedding_size', 512)),
            'omics': 'Dense layer with size ' + str(params.get('omics_dense_size', 256))
        },
        'attention': {
            'molecule_gep': str(params.get('molecule_gep_heads', [2])),
            'molecule_cnv': str(params.get('molecule_cnv_heads', [2])),
            'molecule_mut': str(params.get('molecule_mut_heads', [2])),
            'gene': str(params.get('gene_heads', [1])),
            'cnv': str(params.get('cnv_heads', [1])),
            'mut': str(params.get('mut_heads', [1]))
        },
        'dense_layers': str(params.get('stacked_dense_hidden_sizes', [1024, 512])),
        'dropout': str(params.get('dropout', 0.5)),
        'activation': params.get('activation_fn', 'relu'),
        'loss': params.get('loss_fn', 'mse')
    }
    
    return summary

def generate_sample_results():
    """Generate sample results for demonstration."""
    # Training progress
    epochs = list(range(1, 201))
    train_loss = [1.0 * np.exp(-0.01 * e) + 0.1 + 0.05 * np.random.random() for e in epochs]
    test_loss = [1.2 * np.exp(-0.01 * e) + 0.15 + 0.1 * np.random.random() for e in epochs]
    pearson = [0.5 + 0.3 * (1 - np.exp(-0.02 * e)) + 0.05 * np.random.random() for e in epochs]
    rmse = [1.0 * np.exp(-0.01 * e) + 0.2 + 0.05 * np.random.random() for e in epochs]
    r2 = [0.3 + 0.4 * (1 - np.exp(-0.015 * e)) + 0.05 * np.random.random() for e in epochs]
    
    # Best metrics
    best_epoch = np.argmax(pearson)
    best_metrics = {
        'epoch': epochs[best_epoch],
        'train_loss': train_loss[best_epoch],
        'test_loss': test_loss[best_epoch],
        'pearson': pearson[best_epoch],
        'rmse': rmse[best_epoch],
        'r2': r2[best_epoch]
    }
    
    # Predictions vs actual
    actual = np.random.random(100) * 10
    predicted = actual + (np.random.random(100) - 0.5) * 2
    
    # Attention weights
    attention_weights = defaultdict(dict)
    for sample_id in range(1, 4):
        pathways = [f"Pathway {i+1}" for i in range(20)]
        drug_features = [f"Drug Feature {i+1}" for i in range(15)]
        weights = np.random.random((20, 15))
        
        # Create some pattern in the attention weights
        for i in range(20):
            for j in range(15):
                if i == j or i == j + 1 or i == j - 1:
                    weights[i, j] = 0.7 + 0.3 * np.random.random()
        
        attention_weights[sample_id] = {
            'pathways': pathways,
            'drug_features': drug_features,
            'weights': weights.tolist()
        }
    
    results = {
        'training_progress': {
            'epochs': epochs,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'pearson': pearson,
            'rmse': rmse,
            'r2': r2
        },
        'best_metrics': best_metrics,
        'predictions': {
            'actual': actual.tolist(),
            'predicted': predicted.tolist()
        },
        'attention_weights': attention_weights
    }
    
    return results
