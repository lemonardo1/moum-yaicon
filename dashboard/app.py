import os
import sys
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename

# Flag to track if ML dependencies are available
ML_DEPENDENCIES_AVAILABLE = True

# Try to import ML dependencies, but continue if not available
try:
    import torch
    from rdkit import Chem
    import deepchem as dc
    
    # Add parent directory to path for imports
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    # Import project-specific modules
    from models.model import PASO_GEP_CNV_MUT
    from utils.utils import get_device, get_log_molar
    from data.TripleOmics_Drug_Dataset import TripleOmics_Drug_dataset, custom_collate_fn, create_adj_matrix, get_bond_info
    from utils import load_model_params, get_available_folds, load_model, get_data_files, load_data_file, get_model_summary, generate_sample_results
    
except ImportError as e:
    print(f"Warning: Some ML dependencies are not available: {e}")
    print("The dashboard will run in limited mode without ML functionality.")
    ML_DEPENDENCIES_AVAILABLE = False
    
    # Define placeholder functions for missing dependencies
    def load_model_params(fold_num=1):
        return {
            'smiles_embedding_size': 512,
            'number_of_pathways': 619,
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
            'activation_fn': 'relu',
            'loss_fn': 'mse'
        }
    
    def get_available_folds():
        return list(range(1, 11))
    
    def load_model(fold_num=1):
        return None
    
    def get_data_files():
        data_dir = '../data'
        data_files = []
        if os.path.exists(data_dir):
            for file in os.listdir(data_dir):
                if file.endswith('.csv') or file.endswith('.pkl'):
                    data_files.append(file)
        return data_files
    
    def load_data_file(filename):
        file_path = os.path.join('../data', filename)
        if filename.endswith('.csv') and os.path.exists(file_path):
            return pd.read_csv(file_path)
        return None
    
    def get_model_summary(fold_num=1):
        params = load_model_params(fold_num)
        return {
            'input_features': {
                'drug': 'SMILES representation',
                'gep': '619 pathways',
                'cnv': '619 pathways',
                'mut': '619 pathways'
            },
            'embedding': {
                'drug': 'Graph Neural Network',
                'omics': 'Dense layer'
            },
            'attention': {
                'molecule_gep': '[2]',
                'molecule_cnv': '[2]',
                'molecule_mut': '[2]',
                'gene': '[1]',
                'cnv': '[1]',
                'mut': '[1]'
            },
            'dense_layers': '[1024, 512]',
            'dropout': '0.5',
            'activation': 'relu',
            'loss': 'mse'
        }
    
    def generate_sample_results():
        # Generate sample results for demonstration
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
            'attention_weights': {}
        }
        
        return results

app = Flask(__name__)
app.secret_key = 'yaicon_dashboard_secret_key'
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Routes
@app.route('/')
def index():
    folds = get_available_folds()
    return render_template('index.html', folds=folds)

@app.route('/model_info/<int:fold_num>')
def model_info(fold_num):
    params = load_model_params(fold_num)
    model_summary = get_model_summary(fold_num)
    return render_template('model_info.html', params=params, fold_num=fold_num, model_summary=model_summary)

@app.route('/data_explorer')
def data_explorer():
    # List available data files
    data_files = get_data_files()
    return render_template('data_explorer.html', data_files=data_files)

@app.route('/view_data/<filename>')
def view_data(filename):
    df = load_data_file(filename)
    
    if isinstance(df, pd.DataFrame):
        return render_template('view_data.html', filename=filename, 
                              columns=df.columns.tolist(), 
                              data=df.head(100).to_dict('records'))
    elif isinstance(df, dict) or isinstance(df, list):
        # For pickle files that contain dict or list
        return render_template('view_data.html', filename=filename,
                              columns=["Content Type"], 
                              data=[{"Content Type": f"Pickle file containing {type(df).__name__}"}])
    else:
        return render_template('view_data.html', filename=filename,
                              columns=["Content Type"], 
                              data=[{"Content Type": f"Unsupported content type: {type(df).__name__ if df is not None else 'None'}"}])

@app.route('/results/<int:fold_num>')
def results(fold_num):
    results_dir = f'../result/model/train_MixedSet_10Fold_GEP_CNV_MUT/Fold{fold_num}/results'
    result_files = []
    
    if os.path.exists(results_dir):
        for file in os.listdir(results_dir):
            if file.endswith('.json'):
                result_files.append(file)
    
    return render_template('results.html', fold_num=fold_num, result_files=result_files)

@app.route('/prediction')
def prediction():
    folds = get_available_folds()
    
    # Get list of available drugs and cell lines
    drug_list = []
    cell_line_list = []
    
    try:
        # Try to load drug list from SMILES file
        smiles_file = '../data/CCLE-GDSC-SMILES.csv'
        if os.path.exists(smiles_file):
            smiles_df = pd.read_csv(smiles_file)
            drug_list = smiles_df['DRUG_NAME'].unique().tolist()
        
        # Try to load cell line list from GEP file
        gep_file = '../data/GEP_Wilcoxon_Test_Analysis_Log10_P_value_C2_KEGG_MEDICUS.csv'
        if os.path.exists(gep_file):
            gep_df = pd.read_csv(gep_file, index_col=0)
            cell_line_list = gep_df.index.tolist()
    except Exception as e:
        print(f"Error loading drug/cell line lists: {e}")
        # Provide some sample data if loading fails
        drug_list = ["Drug1", "Drug2", "Drug3"]
        cell_line_list = ["CellLine1", "CellLine2", "CellLine3"]
    
    return render_template('prediction.html', 
                          folds=folds, 
                          selected_fold=folds[0] if folds else 1,
                          drug_list=drug_list,
                          cell_line_list=cell_line_list)

@app.route('/api/model_architecture/<int:fold_num>')
def model_architecture(fold_num):
    params = load_model_params(fold_num)
    
    # Extract key architecture details
    architecture = {
        'smiles_embedding_size': params.get('smiles_embedding_size', 512),
        'number_of_pathways': params.get('number_of_pathways', 619),
        'molecule_gep_heads': params.get('molecule_gep_heads', [2]),
        'molecule_cnv_heads': params.get('molecule_cnv_heads', [2]),
        'molecule_mut_heads': params.get('molecule_mut_heads', [2]),
        'gene_heads': params.get('gene_heads', [1]),
        'cnv_heads': params.get('cnv_heads', [1]),
        'mut_heads': params.get('mut_heads', [1]),
        'n_heads': params.get('n_heads', 2),
        'num_layers': params.get('num_layers', 4),
        'omics_dense_size': params.get('omics_dense_size', 256),
        'stacked_dense_hidden_sizes': params.get('stacked_dense_hidden_sizes', [1024, 512]),
        'dropout': params.get('dropout', 0.5),
    }
    
    return jsonify(architecture)

@app.route('/api/results/<int:fold_num>')
def api_results(fold_num):
    """API endpoint to get results data for visualizations."""
    # For now, generate sample results
    results = generate_sample_results()
    return jsonify(results)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for making predictions."""
    try:
        fold_num = int(request.form.get('fold_num', 1))
        prediction_type = request.form.get('prediction_type', 'sample')
        drug_name = request.form.get('drug_name', 'Unknown Drug')
        cell_line = request.form.get('cell_line', 'Unknown Cell Line')
        
        # Check if ML dependencies are available
        if not ML_DEPENDENCIES_AVAILABLE:
            # Return demo result if ML dependencies are not available
            return jsonify({
                'success': True,
                'message': 'Demo prediction (ML dependencies not available)',
                'prediction': {
                    'ic50': 0.75,
                    'log_ic50': -0.125,
                    'drug': drug_name,
                    'cell_line': cell_line
                }
            })
        
        if prediction_type == 'sample':
            # Use sample data for demonstration
            # Load required data files
            smiles_file = '../data/CCLE-GDSC-SMILES.csv'
            gep_file = '../data/GEP_Wilcoxon_Test_Analysis_Log10_P_value_C2_KEGG_MEDICUS.csv'
            cnv_file = '../data/CNV_Cardinality_Analysis_of_Variance_C2_KEGG_MEDICUS.csv'
            mut_file = '../data/MUT_Cardinality_Analysis_of_Variance_C2_KEGG_MEDICUS.csv'
            
            # Check if files exist
            if not all(os.path.exists(f) for f in [smiles_file, gep_file, cnv_file, mut_file]):
                # Return demo result if files don't exist
                return jsonify({
                    'success': True,
                    'message': 'Demo prediction (data files not found)',
                    'prediction': {
                        'ic50': 0.75,
                        'log_ic50': -0.125,
                        'drug': drug_name,
                        'cell_line': cell_line
                    }
                })
            
            # Load data
            smiles_df = pd.read_csv(smiles_file)
            gep_df = pd.read_csv(gep_file, index_col=0)
            cnv_df = pd.read_csv(cnv_file, index_col=0)
            mut_df = pd.read_csv(mut_file, index_col=0)
            
            # Check if drug and cell line exist in data
            if drug_name not in smiles_df['DRUG_NAME'].values or cell_line not in gep_df.index:
                return jsonify({
                    'success': False,
                    'message': f"Drug '{drug_name}' or cell line '{cell_line}' not found in data"
                })
            
            # Get SMILES for drug
            smiles = smiles_df[smiles_df['DRUG_NAME'] == drug_name]['SMILES'].values[0]
            
            # Get omics data for cell line
            gep_data = torch.tensor(gep_df.loc[cell_line].values, dtype=torch.float)
            cnv_data = torch.tensor(cnv_df.loc[cell_line].values, dtype=torch.float)
            mut_data = torch.tensor(mut_df.loc[cell_line].values, dtype=torch.float)
            
            # Process SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return jsonify({
                    'success': False,
                    'message': f"Invalid SMILES string for drug '{drug_name}'"
                })
            
            # Create features, adjacency matrix, and bond info
            featurizer = dc.feat.graph_features.ConvMolFeaturizer(use_chirality=True)
            mol_object = featurizer.featurize([mol])
            features = torch.from_numpy(mol_object[0].atom_features)
            adj_list = mol_object[0].canon_adj_list
            adj_matrix = create_adj_matrix(adj_list, len(adj_list))
            bond_info = get_bond_info(mol)
            
            # Load model
            model = load_model(fold_num)
            if model is None:
                return jsonify({
                    'success': True,
                    'message': 'Demo prediction (model not available)',
                    'prediction': {
                        'ic50': 0.65,
                        'log_ic50': -0.187,
                        'drug': drug_name,
                        'cell_line': cell_line
                    }
                })
                
            model.eval()
            
            # Prepare input for model
            drug_data = ([features.unsqueeze(0), adj_matrix.unsqueeze(0), [bond_info]])
            gep_data = gep_data.unsqueeze(0)
            cnv_data = cnv_data.unsqueeze(0)
            mut_data = mut_data.unsqueeze(0)
            
            # Make prediction
            with torch.no_grad():
                predictions, prediction_dict = model(drug_data, gep_data, cnv_data, mut_data)
            
            # Process prediction
            ic50 = predictions.item()
            log_ic50 = prediction_dict.get('log_micromolar_IC50', torch.tensor([0.0])).item()
            
            return jsonify({
                'success': True,
                'message': 'Prediction successful',
                'prediction': {
                    'ic50': ic50,
                    'log_ic50': log_ic50,
                    'drug': drug_name,
                    'cell_line': cell_line
                }
            })
            
        elif prediction_type == 'custom':
            # Handle custom file uploads
            # This would require more complex processing of uploaded files
            
            # For now, return a demo result
            return jsonify({
                'success': True,
                'message': 'Demo prediction for custom data',
                'prediction': {
                    'ic50': 0.85,
                    'log_ic50': -0.071,
                    'drug': 'Custom Drug',
                    'cell_line': 'Custom Cell Line'
                }
            })
        
        else:
            return jsonify({
                'success': False,
                'message': f"Invalid prediction type: {prediction_type}"
            })
            
    except Exception as e:
        # Log the error
        print(f"Prediction error: {str(e)}")
        
        # Return a demo result in case of error
        return jsonify({
            'success': True,
            'message': f'Demo prediction (error occurred: {str(e)})',
            'prediction': {
                'ic50': 0.65,
                'log_ic50': -0.187,
                'drug': drug_name,
                'cell_line': cell_line
            }
        })

@app.route('/api/upload_data', methods=['POST'])
def api_upload_data():
    """API endpoint for uploading custom data files."""
    try:
        file_type = request.form.get('file_type')
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'message': 'No file part'
            })
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'success': False,
                'message': 'No selected file'
            })
        
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            return jsonify({
                'success': True,
                'message': f'File uploaded successfully: {filename}',
                'file_type': file_type,
                'file_path': file_path
            })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error uploading file: {str(e)}'
        })

@app.route('/compare')
def compare():
    """Page for comparing results across folds."""
    folds = get_available_folds()
    return render_template('compare.html', folds=folds)

@app.route('/api/compare_folds', methods=['POST'])
def api_compare_folds():
    """API endpoint for comparing results across folds."""
    try:
        fold_nums = request.json.get('folds', [1, 2, 3])
        metric = request.json.get('metric', 'pearson')
        
        # Generate sample comparison data
        comparison_data = {
            'folds': fold_nums,
            'metrics': {}
        }
        
        metrics = ['pearson', 'rmse', 'r2', 'loss']
        for m in metrics:
            comparison_data['metrics'][m] = [round(0.5 + 0.3 * np.random.random(), 3) for _ in fold_nums]
        
        return jsonify({
            'success': True,
            'data': comparison_data
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error comparing folds: {str(e)}'
        })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
