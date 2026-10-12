import os
import torch
import numpy as np
from torch.utils.data import Dataset
# ===================== 新增：图模型依赖 =====================
from rdkit import Chem
from torch_geometric.data import Data, Batch
# ===================== 强制关闭 ESPF 路径报错 =====================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
def get_value(row, candidates, default=None):
    for name in candidates:
        if name in row.index:
            value = row[name]
            if value is not None and str(value).strip().lower() != "nan":
                return value
    return default
# ===================== 新增：SMILES → 分子图 =====================
def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None

    atoms = []
    for atom in mol.GetAtoms():
        symbol = [0] * 10
        atom_list = [6, 7, 8, 9, 15, 16, 17, 35, 53, 0]
        atomic_num = atom.GetAtomicNum()
        if atomic_num in atom_list:
            symbol[atom_list.index(atomic_num)] = 1
        else:
            symbol[-1] = 1

        degree = [0] * 6
        degree[min(atom.GetDegree(), 5)] = 1

        formal_charge = [0, 1] if atom.GetFormalCharge() >= 0 else [1, 0]

        hydrogens = [0] * 8
        hydrogens[min(atom.GetTotalNumHs(), 7)] = 1

        atoms.append(symbol + degree + formal_charge + hydrogens)

    edges = []
    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        edges.append([a, b])
        edges.append([b, a])

    x = torch.tensor(atoms, dtype=torch.float)

    if len(edges) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)
#编码函数
DRUG_CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890!@#$%^&*()[]=+-#/\\.%:'
PROT_CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

DRUG_DICT = {c: i + 1 for i, c in enumerate(DRUG_CHARS)}
PROT_DICT = {c: i + 1 for i, c in enumerate(PROT_CHARS)}

def encode_smiles_simple(smiles, max_len=128):
    x = torch.zeros(max_len, dtype=torch.long)
    s = str(smiles)
    for i, c in enumerate(s[:max_len]):
        x[i] = DRUG_DICT.get(c, 0)
    mask = (x != 0).long()
    return x, mask

def encode_protein_simple(seq, max_len=1024):
    x = torch.zeros(max_len, dtype=torch.long)
    s = str(seq)
    for i, c in enumerate(s[:max_len]):
        x[i] = PROT_DICT.get(c.upper(), 0)
    mask = (x != 0).long()
    return x, mask

# 直接用最简单版，不加载 BPE 词典
def drug2emb_encoder(x):
    max_d = 80
    xs = x[:max_d]
    idx = [ord(i) for i in xs]
    idx = idx + [0]*(max_d - len(idx))
    mask = [1]*len(xs) + [0]*(max_d - len(xs))
    return idx, mask

def protein2emb_encoder(x):
    max_p = 560
    xs = x[:max_p]
    idx = [ord(i) for i in xs]
    idx = idx + [0]*(max_p - len(idx))
    mask = [1]*len(xs) + [0]*(max_p - len(xs))
    return idx, mask

# ===================== DeepDTA 工具 =====================
#from datahelper import CHARISOSMISET, CHARPROTSET, label_smiles, label_sequence

class UniversalDTIDataset(Dataset):
    def __init__(
        self,
        df,
        dataset_type="default",
        dta_threshold=7.0,
        max_drug_len=128,
        max_protein_len=1024,
        use_jczs=True
    ):
        self.df = df.reset_index(drop=True)
        self.dataset_type = dataset_type
        self.dta_threshold = dta_threshold
        self.max_drug_len = max_drug_len
        self.max_protein_len = max_protein_len
        self.use_jczs = use_jczs

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        smiles = get_value(row, ["smiles", "SMILES", "drug", "Drug"], "")
        protein = get_value(row, ["protein_sequence", "Target Sequence", "target_sequence", "protein"], "")
        label = float(get_value(row, ["label", "Label", "affinity", "y"], 0.0))

        jc_code = get_value(row, ["jc_code", "JC_Code", "jczs", "herb_role_code"], 0)
        try:
            jc_code = int(jc_code)
        except Exception:
            jc_code = 0

        if not self.use_jczs:
            jc_code = 0

        drug_seq, drug_mask = encode_smiles_simple(smiles, self.max_drug_len)
        protein_seq, protein_mask = encode_protein_simple(protein, self.max_protein_len)

        graph = smiles_to_graph(smiles)
        if graph is None:
            graph = Data(
                x=torch.zeros((1, 26), dtype=torch.float),
                edge_index=torch.empty((2, 0), dtype=torch.long)
            )

        label_dta = torch.tensor(label, dtype=torch.float)
        label_dti = torch.tensor(1.0 if label >= self.dta_threshold else 0.0, dtype=torch.float)

        return {
            "id": get_value(row, ["id", "ID"], idx),
            "smiles": smiles,
            "protein_sequence": protein,
            "drug_seq": drug_seq,
            "drug_mask": drug_mask,
            "protein_seq": protein_seq,
            "protein_mask": protein_mask,
            "graph": graph,
            "label_dta": label_dta,
            "label_dti": label_dti,
            "jc_code": torch.tensor(jc_code, dtype=torch.long),
        }
#图数据 collate 函数
def dti_collate_fn(batch):
    return {
        "id": [item["id"] for item in batch],
        "smiles": [item["smiles"] for item in batch],
        "protein_sequence": [item["protein_sequence"] for item in batch],
        "drug_seq": torch.stack([item["drug_seq"] for item in batch]),
        "drug_mask": torch.stack([item["drug_mask"] for item in batch]),
        "protein_seq": torch.stack([item["protein_seq"] for item in batch]),
        "protein_mask": torch.stack([item["protein_mask"] for item in batch]),
        "graph": Batch.from_data_list([item["graph"] for item in batch]),
        "label_dta": torch.stack([item["label_dta"] for item in batch]),
        "label_dti": torch.stack([item["label_dti"] for item in batch]),
        "jc_code": torch.stack([item["jc_code"] for item in batch]),
    }
