import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import pickle
from time import time
import torch

from data.TripleOmics_Drug_Dataset import TripleOmics_Drug_dataset, custom_collate_fn
from models.model import PASO_GEP_CNV_MUT
from utils.hyperparams import OPTIMIZER_FACTORY
from utils.loss_functions import pearsonr, r2_score
from utils.utils import get_device, get_log_molar

def main(
    drug_sensitivity_filepath,
    gep_filepath,
    cnv_filepath,
    mut_filepath,
    smiles_filepath,
    gene_filepath,
    model_path,
    params,
    training_name
):

    # Process parameter file:
    params = params
    params.update(
        {
            "batch_size": 4,
            "epochs": 200,
            "num_workers": 4,
            "stacked_dense_hidden_sizes": [
                1024,
                512
            ],
        }
    )
    print(params)

    # Prepare the dataset
    print("Start data preprocessing...")

    # Load the gene list
    with open(gene_filepath, "rb") as f:
        pathway_list = pickle.load(f)

    #===================================================
    n_folds = params.get("fold", 11)
    #===================================================

    for fold in range(n_folds):
        print(f"============== Fold [{fold+1}/{params['fold']}] ==============")
        # Create model directory and dump files
        model_dir = os.path.join(model_path, training_name, 'Fold' + str(fold+1))
        os.makedirs(os.path.join(model_dir, "weights"), exist_ok=True)
        os.makedirs(os.path.join(model_dir, "results"), exist_ok=True)
        with open(os.path.join(model_dir, "TCGA_classifier_best_aucpr_GEP.json"), "w") as fp:
            json.dump(params, fp, indent=4)

        # load the drug sensitivity data
        drug_sensitivity_train = drug_sensitivity_filepath + 'train_Fold' + str(fold) + '.csv'
        train_dataset = TripleOmics_Drug_dataset(
            drug_sensitivity_filepath=drug_sensitivity_train,
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
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=params["batch_size"],
            shuffle=True,
            drop_last=True,
            num_workers=0,
            collate_fn=custom_collate_fn
        )
        drug_sensitivity_test = drug_sensitivity_filepath + 'test_Fold' + str(fold) + '.csv'
        min_value = params["drug_sensitivity_processing_parameters"]["parameters"]["min"]
        max_value = params["drug_sensitivity_processing_parameters"]["parameters"]["max"]
        test_dataset = TripleOmics_Drug_dataset(
            drug_sensitivity_filepath=drug_sensitivity_test,
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
        test_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=params["batch_size"],
            shuffle=True,
            drop_last=True,
            num_workers=0,
            collate_fn=custom_collate_fn
        )
        print(
            f"FOLD [{fold+1}/{params['fold']}]"
            f"Training dataset has {len(train_dataset)} samples, test set has "
            f"{len(test_dataset)}."
        )
        device = get_device()
        save_top_model = os.path.join(model_dir, "weights/{}_{}_{}.pt")
        params.update(
            {  # yapf: disable
                "number_of_genes": len(pathway_list),
            }
        )
        model = PASO_GEP_CNV_MUT(params).to(get_device())
        model.train()

        min_loss, min_rmse, max_pearson, max_r2 = 100, 1000, 0, 0

        # Define optimizer
        optimizer = OPTIMIZER_FACTORY[params.get("optimizer", "Adam")](
            model.parameters(), lr=params.get("lr", 0.001)
        )

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        params.update({"number_of_parameters": num_params})
        print(f"Number of parameters {num_params}")

        # Overwrite params.json file with updated parameters.
        with open(os.path.join(model_dir, "TCGA_classifier_best_aucpr_GEP.json"), "w") as fp:
            json.dump(params, fp)

        # Start training
        print("Training about to start...\n")
        t = time()
        # start training
        for epoch in range(params["epochs"]):
            print(f"== Fold [{fold+1}/{params['fold']}] Epoch [{epoch+1}/{params['epochs']}] ==")

            training(model, device, epoch, fold, train_loader, optimizer, params, t)

            t = time()

            test_loss_a, test_pearson_a, test_rmse_a, test_r2_a, predictions, labels = (
                evaluation(model, device, test_loader, params, epoch, fold, max_value, min_value))

            def save(path, metric, typ, val=None):
                fold_info = "Fold_" + str(fold+1)
                model.save(path.format(fold_info + typ, metric, "bgd-test"))
                with open(os.path.join(model_dir, "results", fold_info + metric + ".json"), "w") as f:
                    json.dump(info, f)
                if typ == "best":
                    print(
                        f'\t New best performance in "{metric}"'
                        f" with value : {val:.7f} in epoch: {epoch}"
                    )

            def update_info():
                return {
                    "best_rmse": str(float(min_rmse)),
                    "best_pearson": str(float(max_pearson)),
                    "test_loss": str(min_loss),
                    "best_r2": str(float(max_r2)),
                    "predictions": [float(p) for p in predictions],
                }

            if test_loss_a < min_loss:
                min_rmse = test_rmse_a
                min_loss = test_loss_a
                min_loss_pearson = test_pearson_a
                min_loss_r2 = test_r2_a
                info = update_info()
                save(save_top_model, "mse", "best", min_loss)
                ep_loss = epoch
            if test_pearson_a > max_pearson:
                max_pearson = test_pearson_a
                max_pearson_loss = test_loss_a
                max_pearson_r2 = test_r2_a
                info = update_info()
                save(save_top_model, "pearson", "best", max_pearson)
                ep_pearson = epoch
            if test_r2_a > max_r2:
                max_r2 = test_r2_a
                max_r2_loss = test_loss_a
                max_r2_pearson = test_pearson_a
                info = update_info()
                save(save_top_model, "r2", "best", max_r2)
                ep_r2 = epoch
        print(
            f"Overall Fold {fold+1} best performances are: \n \t"
            f"Loss = {min_loss:.4f} in epoch {ep_loss} "
            f"\t (Pearson was {min_loss_pearson:4f}; R2 was {min_loss_r2:4f}) \n \t"
            f"Pearson = {max_pearson:.4f} in epoch {ep_pearson} "
            f"\t (Loss was {max_pearson_loss:2f}; R2 was {max_pearson_r2:4f}) \n \t"
            f"R2 = {max_r2:.4f} in epoch {ep_r2} "
            f"\t (Loss was {max_r2_loss:4f}; Pearson was {max_r2_pearson:4f}) \n \t"
        )
        save(save_top_model, "training", "done")

    print("Done with training, models saved, shutting down.")



def training(model, device, epoch, fold, train_loader, optimizer, params, t):
    model.train()
    train_loss = 0
    for ind, (drug_data, omic_1, omic_2, omic_3, y) in enumerate(train_loader):
        y_hat, pred_dict = model(
            drug_data, omic_1.to(device), omic_2.to(device), omic_3.to(device))
        loss = model.loss(y_hat, y.to(device))
        optimizer.zero_grad()
        loss.backward()
        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2e-6)
        optimizer.step()
        train_loss += loss.item()
    print(
        "\t **** TRAINING ****  "
        f"Fold[{fold+1}] Epoch [{epoch + 1}/{params['epochs']}], "
        f"loss: {train_loss / len(train_loader):.5f}. "
        f"This took {time() - t:.1f} secs."
    )

def evaluation(model, device, test_loader, params, epoch, fold, max_value, min_value):
    # Measure validation performance
    model.eval()
    with torch.no_grad():
        test_loss = 0
        log_pres = []
        log_labels = []
        for ind, (drug_data, omic_1, omic_2, omic_3, y) in enumerate(test_loader):
            y_hat, pred_dict = model(
                drug_data, omic_1.to(device), omic_2.to(device), omic_3.to(device)
            )
            log_pre = pred_dict.get("log_micromolar_IC50")
            log_pres.append(log_pre)
            # predictions.append(y_hat)
            log_y = get_log_molar(y, ic50_max=max_value, ic50_min=min_value)
            log_labels.append(log_y)
            # labels.append(y)
            loss = model.loss(log_pre, log_y.to(device))
            test_loss += loss.item()

    # on the logIC50 scale
    predictions = torch.cat([p.cpu() for preds in log_pres for p in preds])
    labels = torch.cat([l.cpu() for label in log_labels for l in label])
    test_pearson_a = pearsonr(torch.Tensor(predictions), torch.Tensor(labels))
    test_rmse_a = torch.sqrt(torch.mean((predictions - labels) ** 2))
    test_loss_a = test_loss / len(test_loader)
    test_r2_a = r2_score(torch.Tensor(predictions), torch.Tensor(labels))
    print(
        f"\t ****   Test   ****  Fold[{fold+1}] Epoch [{epoch + 1}/{params['epochs']}], "
        f"loss: {test_loss_a:.5f}, "
        f"Pearson: {test_pearson_a:.4f}, "
        f"RMSE: {test_rmse_a:.4f}, "
        f"R2: {test_r2_a:.4f}. "
    )
    return test_loss_a, test_pearson_a, test_rmse_a, test_r2_a, predictions, labels


if __name__ == "__main__":

    # train_sensitivity_filepath = 'data/drug_sensitivity_MixedSet_test.csv'
    # test_sensitivity_filepath = 'data/drug_sensitivity_MixedSet_test.csv'
    drug_sensitivity_filepath = 'data/10_fold_data/mixed/MixedSet_'
    smiles_filepath = 'data/CCLE-GDSC-SMILES.csv'
    gep_filepath = 'data/GEP_Wilcoxon_Test_Analysis_Log10_P_value_C2_KEGG_MEDICUS.csv'
    cnv_filepath = 'data/CNV_Cardinality_Analysis_of_Variance_C2_KEGG_MEDICUS.csv'
    mut_filepath = 'data/MUT_Cardinality_Analysis_of_Variance_C2_KEGG_MEDICUS.csv'
    gene_filepath = 'data/MUDICUS_Omic_619_pathways.pkl'

    model_path = 'result/model'
    params = {
        'fold': 10, 
        'optimizer': "adam",
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
    training_name = 'train_MixedSet_10Fold_GEP_CNV_MUT'
    # run the training
    main(
        drug_sensitivity_filepath,
        gep_filepath,
        cnv_filepath,
        mut_filepath,
        smiles_filepath,
        gene_filepath,
        model_path,
        params,
        training_name
    )
