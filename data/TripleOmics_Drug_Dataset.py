import os
import deepchem as dc
from rdkit import Chem
import numpy as np

# dev date 2023/11/25 14:28
from numpy import log
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from typing import List, Tuple, Dict



# 커스텀 collate_fn 정의  -> 고정된 패딩 크기로 바꾸는 방식을 사용해야 할 듯
def custom_collate_fn(batch):
    # batch: [(drug_data, gep_data, cnv_data, mut_data, ic50), ...]
    # drug_data: (x, adj_matrix, bond_info)

    # 1. drug_data 분리
    drug_data = [item[0] for item in batch]  # [(x, adj_matrix, bond_info), ...]
    xs = [item[0] for item in drug_data]  # [x1, x2, ...]
    adj_matrices = [item[1] for item in drug_data]  # [adj_matrix1, adj_matrix2, ...]
    bond_infos = [item[2] for item in drug_data]  # [bond_info1, bond_info2, ...]

    padded_xs = torch.zeros(len(xs), 256, xs[0].shape[1])  # [batch_size, max_len, feature_dim]
    for i, x in enumerate(xs):
        padded_xs[i, :x.shape[0], :] = x.clone()

    # 3. adj_matrix 패딩
    padded_adj_matrices = torch.zeros(len(adj_matrices), 256, 256)
    for i, adj in enumerate(adj_matrices):
        num_nodes = adj.shape[-1]
        padded_adj_matrices[i, :num_nodes, :num_nodes] = adj.clone()

    # 4. 나머지 데이터 처리
    gep_data = torch.stack([item[1] for item in batch])
    cnv_data = torch.stack([item[2] for item in batch])
    mut_data = torch.stack([item[3] for item in batch])
    ic50 = torch.stack([item[4] for item in batch])

    return (padded_xs, padded_adj_matrices, bond_infos), gep_data, cnv_data, mut_data, ic50

# 정규화된 인접 행렬 생성
def create_adj_matrix(adj_list: List[List[int]], num_nodes: int) -> torch.Tensor:
    adj = torch.eye(num_nodes)
    for node, neighbors in enumerate(adj_list):
        for neighbor in neighbors:
            adj[node, neighbor] = 1
            adj[neighbor, node] = 1
    return adj

def get_bond_info(mol: Chem.Mol) -> list:
    """RDKit Mol 객체에서 결합 정보 추출."""
    bond_info = []
    for bond in mol.GetBonds():
        start_atom = bond.GetBeginAtomIdx()
        end_atom = bond.GetEndAtomIdx()
        bond_type = str(bond.GetBondType())  # 예: SINGLE, DOUBLE, TRIPLE, AROMATIC
        bond_info.append((start_atom, end_atom, bond_type))
    return bond_info

class TripleOmics_Drug_dataset(Dataset):
    def __init__(self,
                 drug_sensitivity_filepath: str,
                 smiles_filepath: str, #CCLE-GDSC-SMILES.csv
                 gep_filepath: str,
                 cnv_filepath: str,
                 mut_filepath: str,
                 gep_standardize: bool = False,
                 cnv_standardize: bool = False,
                 mut_standardize: bool = False,
                 drug_sensitivity_dtype: torch.dtype = torch.float,
                 gep_dtype: torch.dtype = torch.float,
                 cnv_dtype: torch.dtype = torch.float,
                 mut_dtype: torch.dtype = torch.float,
                #  smiles_language: SMILESLanguage = None,
                 drug_sensitivity_min_max: bool = True,
                 column_names: Tuple[str] = ['drug', 'cell_line', 'IC50'],
                 ):
        self.drug_sensitivity = pd.read_csv(drug_sensitivity_filepath, index_col=0)
        self.smiles = pd.read_csv(smiles_filepath) 
        self.gep_standardize = gep_standardize
        self.cnv_standardize = cnv_standardize
        self.mut_standardize = mut_standardize
        self.drug_sensitivity_dtype = drug_sensitivity_dtype
        self.gep_dtype = gep_dtype
        self.cnv_dtype = cnv_dtype
        self.mut_dtype = mut_dtype
        # self.smiles_language = smiles_language
        self.drug_sensitivity_min_max = drug_sensitivity_min_max
        self.drug_sensitivity_processing_parameters = {}
        self.column_names = column_names
        self.drug_name, self.cell_name, self.label_name = self.column_names
        if gep_filepath is not None:
            self.gep = pd.read_csv(gep_filepath, index_col=0)
            if gep_standardize:
                scaler = StandardScaler()
                self.gep_standardized = scaler.fit_transform(self.gep)
                self.gep = pd.DataFrame(self.gep_standardized, index=self.gep.index)
        if cnv_filepath is not None:
            self.cnv = pd.read_csv(cnv_filepath, index_col=0)
            if cnv_standardize:
                scaler = StandardScaler()
                self.cnv_standardized = scaler.fit_transform(self.cnv)
                self.cnv = pd.DataFrame(self.cnv_standardized, index=self.cnv.index)
        if mut_filepath is not None:
            self.mut = pd.read_csv(mut_filepath, index_col=0)
            if mut_standardize:
                scaler = StandardScaler()
                self.mut_standardized = scaler.fit_transform(self.mut)
                self.mut = pd.DataFrame(self.mut_standardized, index=self.mut.index)

        # NOTE: optional min-max scaling
        if self.drug_sensitivity_min_max:
            minimum = self.drug_sensitivity_processing_parameters.get(
                'min', self.drug_sensitivity[self.label_name].min()
            )
            maximum = self.drug_sensitivity_processing_parameters.get(
                'max', self.drug_sensitivity[self.label_name].max()
            )
            self.drug_sensitivity[self.label_name] = (
                self.drug_sensitivity[self.label_name] - minimum
            ) / (maximum - minimum)
            self.drug_sensitivity_processing_parameters = {
                'processing': 'min_max',
                'parameters': {'min': minimum, 'max': maximum},
            }

    def __len__(self):
        return len(self.drug_sensitivity)

    def __getitem__(self, index):
        # drug sensitivity
        molecules = []
        selected_sample = self.drug_sensitivity.iloc[index]
        selected_drug = selected_sample[self.drug_name]
        selected_smiles = self.smiles[self.smiles["DRUG_NAME"]==selected_drug].iloc[0]["SMILES"]
        ic50_tensor = torch.tensor(
            [selected_sample[self.label_name]],
            dtype=self.drug_sensitivity_dtype,
        )
        # SMILES
        mol = Chem.MolFromSmiles(selected_smiles)
        molecules.append(mol)
        featurizer = dc.feat.graph_features.ConvMolFeaturizer(use_chirality = True)
        mol_object = featurizer.featurize(molecules)

        features = mol_object[0].atom_features  # (num_atoms, feature_dim)
        features = torch.from_numpy(features)
        adj_list = mol_object[0].canon_adj_list  # 인접 리스트
        adj_matrix = create_adj_matrix(adj_list, len(adj_list))
        bond_info = get_bond_info(mol)

        # omics data
        gene_expression_tensor = torch.tensor((
            self.gep.loc[selected_sample[self.cell_name]].values),
            dtype=self.gep_dtype)
        cnv_tensor = torch.tensor((
            self.cnv.loc[selected_sample[self.cell_name]].values),
            dtype=self.cnv_dtype)
        mut_tensor = torch.tensor((
            self.mut.loc[selected_sample[self.cell_name]].values),
            dtype=self.mut_dtype)
        return ([features, adj_matrix, bond_info], gene_expression_tensor,
                cnv_tensor, mut_tensor, ic50_tensor)


if __name__ == "__main__":
    # 파일 경로 정의 (지정한 디렉토리에 파일이 있다고 가정)
    drug_sensitivity_filepath = 'C:/Users/GyeongdeokBae/Desktop/multi-omics/yaicon/data/10_fold_data/mixed/MixedSet_test_Fold0.csv'  # 예시 폴드 파일
    smiles_filepath = 'C:/Users/GyeongdeokBae/Desktop/multi-omics/yaicon/data/CCLE-GDSC-SMILES.csv'
    gep_filepath = 'C:/Users/GyeongdeokBae/Desktop/multi-omics/yaicon/data/GEP_Wilcoxon_test_Analysis_Log10_P_value_C2_KEGG_MEDICUS.csv'
    cnv_filepath = 'C:/Users/GyeongdeokBae/Desktop/multi-omics/yaicon/data/CNV_Cardinality_Analysis_of_Variance_C2_KEGG_MEDICUS.csv'
    mut_filepath = 'C:/Users/GyeongdeokBae/Desktop/multi-omics/yaicon/data/MUT_Cardinality_Analysis_of_Variance_C2_KEGG_MEDICUS.csv'

    # 데이터셋 초기화
    dataset = TripleOmics_Drug_dataset(
        drug_sensitivity_filepath=drug_sensitivity_filepath,
        smiles_filepath=smiles_filepath,
        gep_filepath=gep_filepath,
        cnv_filepath=cnv_filepath,
        mut_filepath=mut_filepath,
        gep_standardize=True,  # 유전자 발현 데이터 표준화
        cnv_standardize=True,  # CNV 데이터 표준화
        mut_standardize=True,  # 돌연변이 데이터 표준화
        drug_sensitivity_min_max=True,  # IC50 값 min-max 정규화
        column_names=('drug', 'cell_line', 'IC50')  # 컬럼 이름 설정
    )

    # 데이터셋 길이 확인
    print(f"데이터셋 크기: {len(dataset)}")

    # 첫 번째 샘플 가져오기 및 확인
    sample = dataset[0]
    drug_data, gep_data, cnv_data, mut_data, ic50 = sample

    # 샘플 데이터 구조 출력
    print("\n샘플 데이터 구조:")
    print(f"약물 데이터 (features, adj_list, degree_list, bond_info):")
    print(f" - Atom features shape: {drug_data[0].shape}")
    print(f" - Adjacency list length: {drug_data[1].shape}")
    print(f" - Bond info length: {len(drug_data[2])}")
    print(f"유전자 발현 데이터 shape: {gep_data.shape}")
    print(f"CNV 데이터 shape: {cnv_data.shape}")
    print(f"돌연변이 데이터 shape: {mut_data.shape}")
    print(f"IC50 값: {ic50.item()}")

    # 데이터셋의 첫 5개 샘플 반복하며 IC50 값 확인
    print("\n첫 5개 샘플의 IC50 값:")
    for i in range(min(5, len(dataset))):
        _, _, _, _, ic50 = dataset[i]
        print(f"샘플 {i}: IC50 = {ic50.item()}")