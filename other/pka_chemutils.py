import torch
import numpy as np
import networkx as nx
from rdkit import Chem
from collections import defaultdict
import re

# 原子類型列表
elem_list = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']

# 原子有效電荷字典
atom_eff_charge_dict = {'H': 0.1, 'He': 0.16875, 'Li': 0.12792, 'Be': 0.1912, 'B': 0.24214000000000002, 'C': 0.31358, 'N': 0.3834, 'O': 0.44532, 'F': 0.51, 'Ne': 0.57584, 'Na': 0.2507400000000001, 'Mg': 0.3307500000000001, 'Al': 0.40656000000000003, 'Si': 0.42852, 'P': 0.48864, 'S': 0.54819, 'Cl': 0.61161, 'Ar': 0.6764100000000002, 'K': 0.34952000000000005, 'Ca': 0.43979999999999997, 'Sc': 0.4632400000000001, 'Ti': 0.4816800000000001, 'V': 0.4981200000000001, 'Cr': 0.5133200000000002, 'Mn': 0.5283200000000001, 'Fe': 0.5434000000000001, 'Co': 0.55764, 'Ni': 0.5710799999999999, 'Cu': 0.5842399999999998, 'Zn': 0.5965199999999999, 'Ga': 0.6221599999999999, 'Ge': 0.6780400000000001, 'As': 0.7449200000000001, 'Se': 0.8287199999999999, 'Br': 0.9027999999999999, 'Kr': 0.9769199999999998, 'Rb': 0.4984499999999997, 'Sr': 0.60705, 'Y': 0.6256, 'Zr': 0.6445499999999996, 'Nb': 0.5921, 'Mo': 0.6106000000000003, 'Tc': 0.7226500000000002, 'Ru': 0.6484499999999997, 'Rh': 0.6639499999999998, 'Pd': 1.3617599999999996, 'Ag': 0.6755499999999999, 'Cd': 0.8192, 'In': 0.847, 'Sn': 0.9102000000000005, 'Sb': 0.9994500000000003, 'Te': 1.0808500000000003, 'I': 1.16115, 'Xe': 1.2424500000000003, 'Cs': 0.6363, 'Ba': 0.7575000000000003, 'Pr': 0.7746600000000001, 'Nd': 0.9306600000000004, 'Pm': 0.9395400000000003, 'Sm': 0.8011800000000001, 'Eu': 0.8121600000000001, 'Tb': 0.8300399999999997, 'Dy': 0.8343600000000002, 'Ho': 0.8439000000000001, 'Er': 0.8476199999999999, 'Tm': 0.8584200000000003, 'Yb': 0.8593199999999996, 'Lu': 0.8804400000000001, 'Hf': 0.9164400000000001, 'Ta': 0.9524999999999999, 'W': 0.9854399999999999, 'Re': 1.0116, 'Os': 1.0323000000000009, 'Ir': 1.0566599999999995, 'Pt': 1.0751400000000004, 'Au': 1.0938000000000003, 'Hg': 1.1153400000000004, 'Tl': 1.22538, 'Pb': 1.2393, 'Bi': 1.3339799999999997, 'Po': 1.4220600000000005, 'At': 1.5163200000000003, 'Rn': 1.6075800000000002}

# 定義常見的可解離原子類型
DISSOCIABLE_ATOMS = ['O', 'N', 'S', 'P']

# 定義常見的可解離官能團
DISSOCIABLE_GROUPS = {
    'carboxylic_acid': '[CX3](=O)[OX2H1]',  # -COOH
    'alcohol': '[OX2H]',                     # -OH
    'phenol': '[OX2H][cX3]',                 # Ar-OH
    'amine_primary': '[NX3;H2]',             # -NH2
    'amine_secondary': '[NX3;H1]',           # -NH-
    'amine_tertiary': '[NX3;H0]',            # -N(-)-
    'thiol': '[SX2H]',                       # -SH
    'phosphate': '[P](=O)([O])[O]',          # -PO4
    'sulfonic_acid': '[SX4](=O)(=O)[OX2H]'   # -SO3H
}

# 預定義常見化合物SMILES到解離位點的映射
SPECIAL_STRUCTURES_MAP = {
    # 基本氨基酸
    'NCC(O)=O': [(0, 'amine_primary'), (2, 'carboxylic_acid')],  # 甘氨酸
    'NCC(=O)O': [(0, 'amine_primary'), (3, 'carboxylic_acid')],  # 甘氨酸另一種表示
    'C[C@H](N)C(O)=O': [(2, 'amine_primary'), (3, 'carboxylic_acid')],  # 丙氨酸
    'C[C@@H](N)C(O)=O': [(2, 'amine_primary'), (3, 'carboxylic_acid')],  # D-丙氨酸
    'NC(C)C(O)=O': [(0, 'amine_primary'), (3, 'carboxylic_acid')],  # 丙氨酸(無手性)
    'NC(CC(O)=O)C(O)=O': [(0, 'amine_primary'), (4, 'carboxylic_acid'), (8, 'carboxylic_acid')],  # 天冬氨酸
    'N[C@@H](CC(O)=O)C(O)=O': [(0, 'amine_primary'), (4, 'carboxylic_acid'), (8, 'carboxylic_acid')],  # L-天冬氨酸
    'N[C@@H](CCC(O)=O)C(O)=O': [(0, 'amine_primary'), (5, 'carboxylic_acid'), (9, 'carboxylic_acid')],  # L-谷氨酸
    'NC(CCC(O)=O)C(O)=O': [(0, 'amine_primary'), (5, 'carboxylic_acid'), (9, 'carboxylic_acid')],  # 谷氨酸(無手性)
    
    # 含芳香環的氨基酸
    'NC(Cc1ccccc1)C(O)=O': [(0, 'amine_primary'), (5, 'carboxylic_acid')],  # 苯丙氨酸(無手性)
    'N[C@@H](Cc1ccccc1)C(O)=O': [(0, 'amine_primary'), (5, 'carboxylic_acid')],  # L-苯丙氨酸
    'N[C@H](Cc1ccccc1)C(O)=O': [(0, 'amine_primary'), (5, 'carboxylic_acid')],  # D-苯丙氨酸
    'N[C@@H](Cc1ccc(O)cc1)C(O)=O': [(0, 'amine_primary'), (5, 'carboxylic_acid')],  # L-酪氨酸
    'NC(Cc1ccc(O)cc1)C(O)=O': [(0, 'amine_primary'), (5, 'carboxylic_acid')],  # 酪氨酸(無手性)
    
    # 含雜環的氨基酸
    'N[C@@H](Cc1c[nH]cn1)C(O)=O': [(0, 'amine_primary'), (5, 'carboxylic_acid')],  # L-組氨酸
    'NC(Cc1c[nH]cn1)C(O)=O': [(0, 'amine_primary'), (5, 'carboxylic_acid')],  # 組氨酸(無手性)
    'N[C@@H](Cc1c[nH]c2ccccc12)C(O)=O': [(0, 'amine_primary'), (5, 'carboxylic_acid')],  # L-色氨酸
    'NC(Cc1c[nH]c2ccccc12)C(O)=O': [(0, 'amine_primary'), (5, 'carboxylic_acid')],  # 色氨酸(無手性)
    
    # 胺基酸衍生物
    'OC(=O)CCc1c[nH]cn1': [(1, 'carboxylic_acid')],  # 組氨酸衍生物
    'OC(=O)CCc1c[nH]c2ccccc12': [(1, 'carboxylic_acid')],  # 色氨酸衍生物
    'N[C@@H](CS)C(O)=O': [(0, 'amine_primary'), (3, 'carboxylic_acid')],  # L-半胱氨酸
    'N[C@@H](CSSC[C@H](N)C(O)=O)C(O)=O': [(0, 'amine_primary'), (9, 'amine_primary'), (3, 'carboxylic_acid'), (12, 'carboxylic_acid')],  # 胱氨酸
    
    # 環狀結構
    'c1cc[nH]c1': [(3, 'amine_secondary')],  # 吡咯
    'c1cn[nH]c1': [(3, 'amine_secondary')],  # 吡唑
    'c1cc[nH]cc1': [(3, 'amine_secondary')],  # 咪唑
    'c1c[nH]c2ccccc12': [(2, 'amine_secondary')],  # 吲哚
    'c1ccc2c(c1)[nH]cc2': [(6, 'amine_secondary')],  # 吲哚(另一種表示)
    
    # 簡單有機酸
    'OC(=O)C(=O)O': [(0, 'carboxylic_acid'), (4, 'carboxylic_acid')],  # 草酸
    'OC(=O)CC(O)=O': [(0, 'carboxylic_acid'), (5, 'carboxylic_acid')],  # 丙二酸
    'OC(=O)CCC(O)=O': [(0, 'carboxylic_acid'), (6, 'carboxylic_acid')],  # 丁二酸
    'OC(=O)CCCC(O)=O': [(0, 'carboxylic_acid'), (7, 'carboxylic_acid')],  # 戊二酸
    
    # 常見藥物和天然產物
    'c1c(O)cc(O)cc1O': [(2, 'phenol'), (5, 'phenol'), (8, 'phenol')],  # 間苯三酚
    'c1cc(O)cc(O)c1O': [(3, 'phenol'), (6, 'phenol'), (8, 'phenol')],  # 間苯三酚(另一種表示)
    'OC(=O)c1ccccc1O': [(1, 'carboxylic_acid'), (8, 'phenol')],  # 水楊酸
    'OC(=O)c1ccccc1C(O)=O': [(1, 'carboxylic_acid'), (9, 'carboxylic_acid')],  # 鄰苯二甲酸
}

def onek_encoding_unk(value, allowable_set):
    """One-hot encoding with unknown value handling"""
    if value in allowable_set:
        return [1 if v == value else 0 for v in allowable_set]
    else:
        return [0] * len(allowable_set)

def atom_features(atom):
    """生成原子特徵向量"""
    atom_symbol = atom.GetSymbol()
    
    # 原子符號編碼
    atom_symbol_encoding = onek_encoding_unk(atom_symbol, elem_list)  # 118維
    
    # 原子有效電荷
    atom_eff_charge = [atom_eff_charge_dict.get(atom_symbol, 0.1)]   # 1維
    
    # 原子連接度
    atom_degree_encoding = onek_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5])  # 6維
    
    # 形式電荷
    formal_charge = atom.GetFormalCharge()
    formal_charge_encoding = onek_encoding_unk(formal_charge, [-4, -3, -2, -1, 0, 1, 2, 3, 4])  # 9維
    
    # 手性標籤
    chiral_tag_encoding = onek_encoding_unk(int(atom.GetChiralTag()), [0, 1, 2, 3])  # 4維
    
    # 氫原子數量
    num_h_encoding = onek_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])  # 5維
    
    # 雜化類型
    hybridization_encoding = onek_encoding_unk(
        int(atom.GetHybridization()), 
        [0, 1, 2, 3, 4, 5]  # [S, SP, SP2, SP3, SP3D, SP3D2]
    )  # 6維
    
    # 是否芳香性
    is_aromatic = [atom.GetIsAromatic()]  # 1維
    
    # 是否可能是酸性位點
    is_acidic = [1 if atom_symbol in ['O', 'S'] and atom.GetTotalNumHs() > 0 else 0]  # 1維
    
    # 是否可能是鹼性位點
    is_basic = [1 if atom_symbol == 'N' and atom.GetTotalNumHs() > 0 else 0]  # 1維
    
    # 是否可解離
    is_dissociable = [1 if atom_symbol in DISSOCIABLE_ATOMS and atom.GetTotalNumHs() > 0 else 0]  # 1維
    
    # 合併所有特徵
    return torch.tensor(
        atom_symbol_encoding +
        atom_eff_charge +
        atom_degree_encoding +
        formal_charge_encoding+
        chiral_tag_encoding +
        num_h_encoding +
        hybridization_encoding +
        is_aromatic +
        is_acidic +
        is_basic +
        is_dissociable,
        dtype=torch.float
    )

def bond_features(bond):
    """生成鍵特徵向量"""
    # 鍵類型
    bt = bond.GetBondType()
    bond_type = [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, 
                 bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC]
    
    # 是否在環中
    is_in_ring = [bond.IsInRing()]
    
    # 鍵立體化學
    stereo = int(bond.GetStereo())
    stereo_encoding = onek_encoding_unk(stereo, [0, 1, 2, 3, 4, 5])
    
    # 鍵共軛
    is_conjugated = [bond.GetIsConjugated()]
    
    return torch.tensor(bond_type + is_in_ring + stereo_encoding + is_conjugated, dtype=torch.float)

def get_dissociable_atoms(mol):
    """全面識別分子中可解離的原子，特別針對NIST 46數據庫的氨基酸和有機酸優化"""
    dissociable_atoms = []
    
    # ========== 1. 氨基酸專用處理 ==========
    # 識別常見氨基酸格式的SMARTS模式
    amino_acid_patterns = {
        'N-terminal': ['N[C@@H]', 'N[C@H]', 'NC[C@@H]', 'NC[C@H]', 'NCC', 'NC'],
        'C-terminal': ['C(O)=O', 'C(=O)O', '[C@@H](O)=O', '[C@H](O)=O', '[C@@H](=O)O', '[C@H](=O)O']
    }
    
    try:
        # 檢查是否是氨基酸結構
        is_amino_acid = False
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        
        # 檢查是否同時具有N末端和C末端特徵
        for n_term in amino_acid_patterns['N-terminal']:
            if n_term in smiles:
                for c_term in amino_acid_patterns['C-terminal']:
                    if c_term in smiles:
                        is_amino_acid = True
                        break
                if is_amino_acid:
                    break
        
        if is_amino_acid:
            # 找到N末端的氮原子
            n_term_idx = None
            for atom in mol.GetAtoms():
                if atom.GetSymbol() == 'N' and atom.GetTotalNumHs() > 0 and len([n for n in atom.GetNeighbors() if n.GetSymbol() == 'C']) > 0:
                    # 氨基酸中的N末端通常連接到含有COOH的碳鏈
                    for neighbor in atom.GetNeighbors():
                        if neighbor.GetSymbol() == 'C':
                            # 檢查這個碳是否最終連接到COOH
                            n_term_idx = atom.GetIdx()
                            break
                    if n_term_idx is not None:
                        break
            
            # 找到C末端的羧酸氧原子
            c_term_idxs = []
            for atom in mol.GetAtoms():
                if atom.GetSymbol() == 'O' and atom.GetTotalNumHs() > 0:
                    for neighbor in atom.GetNeighbors():
                        if neighbor.GetSymbol() == 'C':
                            # 檢查是否是羧酸基團
                            for bond in neighbor.GetBonds():
                                if (bond.GetBeginAtomIdx() != atom.GetIdx() and 
                                    bond.GetEndAtomIdx() != atom.GetIdx() and
                                    bond.GetBondType() == Chem.rdchem.BondType.DOUBLE and
                                    mol.GetAtomWithIdx(bond.GetOtherAtomIdx(neighbor.GetIdx())).GetSymbol() == 'O'):
                                    c_term_idxs.append(atom.GetIdx())
                                    break
            
            # 添加識別出的可解離原子
            if n_term_idx is not None:
                dissociable_atoms.append((n_term_idx, 'amine_primary'))
            
            for c_term_idx in c_term_idxs:
                dissociable_atoms.append((c_term_idx, 'carboxylic_acid'))
            
            if dissociable_atoms:
                return dissociable_atoms
    except:
        # 如果氨基酸特定處理失敗，繼續使用一般方法
        pass
    
    # ========== 2. 常見氨基酸手動映射 ==========
    # 針對常見氨基酸及其衍生物的SMILES到解離位點的映射
    amino_acid_map = {
        'NCC(O)=O': [(0, 'amine_primary'), (2, 'carboxylic_acid')],  # 甘氨酸
        'NCC(=O)O': [(0, 'amine_primary'), (3, 'carboxylic_acid')],  # 甘氨酸(另一種表示)
        'C[C@H](N)C(O)=O': [(2, 'amine_primary'), (3, 'carboxylic_acid')],  # 丙氨酸
        'N[C@@H](CC(O)=O)C(O)=O': [(0, 'amine_primary'), (4, 'carboxylic_acid'), (8, 'carboxylic_acid')],  # 天冬氨酸
        'N[C@@H](CCC(O)=O)C(O)=O': [(0, 'amine_primary'), (5, 'carboxylic_acid'), (9, 'carboxylic_acid')],  # 谷氨酸
        'NC(Cc1ccccc1)C(O)=O': [(0, 'amine_primary'), (5, 'carboxylic_acid')],  # 苯丙氨酸
        'N[C@@H](Cc1ccccc1)C(O)=O': [(0, 'amine_primary'), (5, 'carboxylic_acid')],  # L-苯丙氨酸
        'N[C@@H](Cc1ccc(O)cc1)C(O)=O': [(0, 'amine_primary'), (5, 'carboxylic_acid')],  # 酪氨酸
        'NCCC(O)=O': [(0, 'amine_primary'), (3, 'carboxylic_acid')],  # β-丙氨酸
        'NCCCC(O)=O': [(0, 'amine_primary'), (4, 'carboxylic_acid')],  # γ-氨基丁酸
        'OC(=O)CCc1c[nH]cn1': [(1, 'carboxylic_acid')],  # 組氨酸衍生物
        'N[C@@H](Cc1c[nH]cn1)C(O)=O': [(0, 'amine_primary'), (5, 'carboxylic_acid')],  # 組氨酸
    }
    
    # 檢查是否是已知結構
    try:
        smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
        if smiles in amino_acid_map:
            return amino_acid_map[smiles]
        
        # 檢查是否是立體異構體
        iso_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        for pattern, mapping in amino_acid_map.items():
            if iso_smiles.startswith(pattern.split('(')[0]) and any(term in iso_smiles for term in ['C(O)=O', 'C(=O)O']):
                return mapping
    except:
        pass
    
    # ========== 3. 羧酸和胺基專用SMARTS模式 ==========
    carboxylic_patterns = [
        'C(=O)[OH]',  # 標準羧酸
        'C([OH])=O',  # 另一種表示
        '[CX3](=O)[OX2H]',  # RDKit標準格式
        '[#6]C(=O)[O;H1]',  # 通用格式
        'C(=O)O[H]',  # 顯式氫
        '[CX3]([OX1])([OX2H])',  # 酸鹽形式
    ]
    
    amine_patterns = [
        '[NX3;H2]',  # 標準一級胺
        '[NH2]',  # 簡化一級胺
        '[#7;H2]',  # 通用一級胺
        'N[H][H]',  # 顯式氫
        'N[C@H]',  # 氨基酸格式 
        'N[C@@H]',  # 氨基酸格式
        'NC[C@H]',  # 甘氨酸相關
        'NC[C@@H]',  # 甘氨酸相關
        'NCC',  # 甘氨酸相關
    ]
    
    # 尋找羧酸基團
    carboxylic_found = False
    for pattern in carboxylic_patterns:
        try:
            patt = Chem.MolFromSmarts(pattern)
            if patt and mol.HasSubstructMatch(patt):
                matches = mol.GetSubstructMatches(patt)
                for match in matches:
                    for idx in match:
                        atom = mol.GetAtomWithIdx(idx)
                        if atom.GetSymbol() == 'O' and atom.GetTotalNumHs() > 0:
                            if not any(idx == x[0] for x in dissociable_atoms):
                                dissociable_atoms.append((idx, 'carboxylic_acid'))
                                carboxylic_found = True
        except:
            continue
    
    # 尋找胺基
    amine_found = False
    for pattern in amine_patterns:
        try:
            patt = Chem.MolFromSmarts(pattern)
            if patt and mol.HasSubstructMatch(patt):
                matches = mol.GetSubstructMatches(patt)
                for match in matches:
                    for idx in match:
                        atom = mol.GetAtomWithIdx(idx)
                        if atom.GetSymbol() == 'N' and atom.GetTotalNumHs() > 0:
                            if not any(idx == x[0] for x in dissociable_atoms):
                                # 判斷胺的類型
                                if atom.GetTotalNumHs() >= 2:
                                    dissociable_atoms.append((idx, 'amine_primary'))
                                elif atom.GetTotalNumHs() == 1:
                                    dissociable_atoms.append((idx, 'amine_secondary'))
                                amine_found = True
        except:
            continue
    
    # 如果已經找到了羧酸和胺基，直接返回
    if carboxylic_found and amine_found:
        return dissociable_atoms
    
    # ========== 4. 使用通用SMARTS模式和結構分析 ==========
    general_patterns = {
        'carboxylic_acid': [
            'C(=O)O',  # 簡單羧酸
            'C(O)=O',  # 另一種形式  
            '[CX3](=O)[OX2H1]',  # 標準RDKit格式
            '[#6]C(=O)[#8;H1]',  # 通用格式
        ],
        'alcohol': [
            '[OX2H]',  # 標準醇
            '[OH]C',  # 簡單醇
            '[#8;H1][#6]',  # 通用醇
        ],
        'phenol': [
            '[OX2H]c',  # 標準酚
            '[OH]c',  # 簡單酚
            '[#8;H1][#6;a]',  # 通用酚
        ],
        'amine': [
            'N[H]',  # 含氫胺
            '[NX3;H1,H2]',  # 一級或二級胺
            '[#7;H1,H2]',  # 通用胺
        ],
        'thiol': [
            '[SH]',  # 簡單硫醇
            '[#16;H1]',  # 通用硫醇
        ],
    }
    
    # 尋找所有可能的官能團
    for group, patterns in general_patterns.items():
        for pattern in patterns:
            try:
                patt = Chem.MolFromSmarts(pattern)
                if patt and mol.HasSubstructMatch(patt):
                    matches = mol.GetSubstructMatches(patt)
                    for match in matches:
                        for idx in match:
                            atom = mol.GetAtomWithIdx(idx)
                            # 確認是具有解離性的原子
                            if ((group == 'carboxylic_acid' and atom.GetSymbol() == 'O' and atom.GetTotalNumHs() > 0) or
                                (group == 'alcohol' and atom.GetSymbol() == 'O' and atom.GetTotalNumHs() > 0) or
                                (group == 'phenol' and atom.GetSymbol() == 'O' and atom.GetTotalNumHs() > 0) or
                                (group == 'amine' and atom.GetSymbol() == 'N' and atom.GetTotalNumHs() > 0) or
                                (group == 'thiol' and atom.GetSymbol() == 'S' and atom.GetTotalNumHs() > 0)):
                                
                                # 檢查是否已經添加
                                if not any(idx == x[0] for x in dissociable_atoms):
                                    # 根據組別分類
                                    if group == 'amine':
                                        if atom.GetTotalNumHs() >= 2:
                                            dissociable_atoms.append((idx, 'amine_primary'))
                                        else:
                                            dissociable_atoms.append((idx, 'amine_secondary'))
                                    else:
                                        dissociable_atoms.append((idx, group))
            except:
                continue
    
    # ========== 5. 如果還是找不到，使用原子屬性分析 ==========
    if not dissociable_atoms:
        # 遍歷所有原子，檢查解離性
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            symbol = atom.GetSymbol()
            total_hs = atom.GetTotalNumHs()
            
            # 只關注可能解離的原子類型
            if symbol not in ['O', 'N', 'S', 'P']:
                continue
            
            # 確認原子帶有氫原子(解離所必需)
            if total_hs == 0 and symbol != 'P':
                continue
            
            # 根據原子類型和化學環境判斷
            if symbol == 'O' and total_hs > 0:
                # 檢查是否是羧酸
                c_neighbors = [n for n in atom.GetNeighbors() if n.GetSymbol() == 'C']
                if c_neighbors:
                    c = c_neighbors[0]
                    # 檢查碳原子是否有其他雙鍵氧(羧酸特徵)
                    has_carbonyl = False
                    for bond in c.GetBonds():
                        if (bond.GetOtherAtomIdx(c.GetIdx()) != idx and
                            bond.GetBondType() == Chem.rdchem.BondType.DOUBLE and
                            mol.GetAtomWithIdx(bond.GetOtherAtomIdx(c.GetIdx())).GetSymbol() == 'O'):
                            has_carbonyl = True
                            break
                    
                    if has_carbonyl:
                        dissociable_atoms.append((idx, 'carboxylic_acid'))
                    elif any(n.GetIsAromatic() for n in atom.GetNeighbors()):
                        dissociable_atoms.append((idx, 'phenol'))
                    else:
                        dissociable_atoms.append((idx, 'alcohol'))
            
            # 氮原子處理
            elif symbol == 'N' and total_hs > 0:
                if total_hs >= 2:
                    dissociable_atoms.append((idx, 'amine_primary'))
                elif total_hs == 1:
                    dissociable_atoms.append((idx, 'amine_secondary'))
            
            # 硫原子處理
            elif symbol == 'S' and total_hs > 0:
                dissociable_atoms.append((idx, 'thiol'))
    
    # 如果找到多個羧酸基團，確保它們都被添加(對於多羧酸分子很重要)
    final_carboxylic_pattern = Chem.MolFromSmarts('C(=O)O')
    if final_carboxylic_pattern:
        try:
            matches = mol.GetSubstructMatches(final_carboxylic_pattern)
            for match in matches:
                o_idx = match[2]  # C(=O)O中的OH氧原子
                o_atom = mol.GetAtomWithIdx(o_idx)
                if o_atom.GetSymbol() == 'O' and o_atom.GetTotalNumHs() > 0:
                    if not any(o_idx == x[0] for x in dissociable_atoms):
                        dissociable_atoms.append((o_idx, 'carboxylic_acid'))
        except:
            pass
    
    return dissociable_atoms

def tensorize_molecule(smiles, dissociable_atom_indices=None, pka_values=None):
    """將分子轉換為圖形表示"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"無法解析SMILES: {smiles}")
        
        mol = Chem.AddHs(mol)  # 添加氫原子以獲得更準確的化學環境
        
        # 獲取原子特徵
        fatoms = []
        for i, atom in enumerate(mol.GetAtoms()):
            fatoms.append(atom_features(atom))
        fatoms = torch.stack(fatoms, 0)
        
        # 獲取邊和邊特徵
        edges = []
        edge_features = []
        
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            edge_feature = bond_features(bond)
            
            # 添加兩個方向的邊
            edges.append((i, j))
            edge_features.append(edge_feature)
            
            edges.append((j, i))
            edge_features.append(edge_feature)
        
        # 如果沒有鍵，為單原子分子添加自環
        if len(edges) == 0:
            for i in range(mol.GetNumAtoms()):
                edges.append((i, i))
                edge_features.append(torch.zeros(len(bond_features(None))))
        
        # 轉換為PyTorch張量
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.stack(edge_features)
        
        # 創建解離原子遮罩
        dissociable_masks = torch.zeros(mol.GetNumAtoms())
        if dissociable_atom_indices is not None:
            for idx in dissociable_atom_indices:
                dissociable_masks[idx] = 1
        
        # 創建pKa值張量
        pka_tensor = torch.full((mol.GetNumAtoms(),), float('nan'))
        if pka_values is not None:
            for idx, pka in pka_values:
                pka_tensor[idx] = pka
        
        return fatoms, edge_index, edge_attr, dissociable_masks, pka_tensor, mol
    
    except Exception as e:
        print(f"錯誤：{e}")
        return None, None, None, None, None, None

def predict_dissociable_atoms(smiles):
    """增強版預測分子中可能的解離原子，專門針對複雜環狀結構優化"""
    dissociable_atoms = []
    
    try:
        # 先嘗試從特殊映射庫中查找
        if smiles in SPECIAL_STRUCTURES_MAP:
            return SPECIAL_STRUCTURES_MAP[smiles]
            
        # 嘗試標準SMILES解析
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # 嘗試多種修復策略
            sanitized_smiles = sanitize_smiles(smiles)
            mol = Chem.MolFromSmiles(sanitized_smiles)
            
            if mol is None:
                print(f"無法解析SMILES: {smiles}")
                # 立即轉為直接字符串分析
                direct_analysis = analyze_smiles_directly(smiles)
                if direct_analysis:
                    return direct_analysis
                return []
        
        # 添加隱式氫原子，如果失敗繼續處理
        try:
            mol = Chem.AddHs(mol)
        except Exception as e:
            print(f"添加氫原子失敗: {smiles}, {e}")
        
        # 多層級解析策略 - 從專用到一般
        
        # 1. 使用專用環形結構分析
        dissociable_atoms = analyze_heterocyclic_rings(mol)
        if dissociable_atoms:
            return dissociable_atoms
            
        # 2. 使用標準解離原子識別
        dissociable_atoms = get_dissociable_atoms(mol)
        if dissociable_atoms:
            return dissociable_atoms
        
        # 3. 使用增強的芳香環識別
        dissociable_atoms = get_dissociable_atoms_for_aromatics(mol, smiles)
        if dissociable_atoms:
            return dissociable_atoms
            
        # 4. 全分子原子掃描，最全面但準確性較低
        dissociable_atoms = comprehensive_atom_scan(mol)
        if dissociable_atoms:
            return dissociable_atoms
            
        # 5. 最後嘗試直接SMILES分析
        dissociable_atoms = analyze_smiles_directly(smiles)
        if dissociable_atoms:
            return dissociable_atoms
            
        # 如果所有方法都失敗，嘗試最基本的原子解析
        return basic_atom_analysis(mol)
        
    except Exception as e:
        print(f"預測解離原子時出錯: {e}, SMILES: {smiles}")
        # 嘗試直接SMILES分析作為最後手段
        return analyze_smiles_directly(smiles)

def get_dissociable_atoms_for_aromatics(mol, smiles):
    """大幅增強的芳香環和雜環結構解析函數"""
    dissociable_atoms = []
    
    # 1. 擴展芳香環官能團SMARTS模式
    aromatic_patterns = {
        'phenol': [
            'c1ccccc1[OH]',            # 苯酚
            'cc[OH]',                  # 芳香環連接羥基
            'c-[OH]',                  # 芳香環連接羥基
            '[cR][OH]',                # 任何環上的羥基
            'c[OX2H]',                 # 另一種芳香環羥基
            'c-[O;H]',                 # 顯式表示，更寬鬆
        ],
        'aniline': [
            'c1ccccc1[NH2]',           # 苯胺
            'c[NH2]',                  # 芳香環氨基
            'c-[NH2]',                 # 芳香環氨基
            '[cR][NH2]',               # 任何環上的氨基
            'c[NX3H2]',                # 另一種表示
        ],
        'pyrrole_indole': [
            '[nH]1cccc1',              # 吡咯
            '[nH]1cncc1',              # 咪唑
            '[nH]1c2ccccc2cc1',        # 吲哚
            'c1cc[nH]c1',              # 另一種吡咯表示
            'c1c[nH]c2ccccc12',        # 另一種吲哚表示
            '[n;H;+0]',                # 任何帶氫的環氮
            '[n;X3;H1]',               # 任何帶一個氫的三鍵氮
        ],
        'carboxylic_acid': [
            'c-C(=O)[OH]',             # 苯甲酸型
            'c-C([OH])=O',             # 另一種表示
            'c-[CX3](=O)[OX2H]',       # 精確的羧酸
            'c-C(=O)[O;H]',            # 顯式表示
            'c-C(O)=O',                # 另一種表示
        ],
        'thiol': [
            'c-[SH]',                  # 芳香環硫醇
            'c[SX2H]',                 # 另一種表示
            '[sH]',                    # 環內硫原子
        ],
        'phosphate': [
            'c-O[P](=O)([OH])[OH]',    # 芳香環磷酸酯
            'c-O[PX4](=O)([OX2H])([OX2H])',  # 精確表示
        ]
    }
    
    # 處理每種模式
    for group, patterns in aromatic_patterns.items():
        for pattern in patterns:
            try:
                patt = Chem.MolFromSmarts(pattern)
                if patt and mol.HasSubstructMatch(patt):
                    matches = mol.GetSubstructMatches(patt)
                    for match in matches:
                        # 根據官能團類型找出正確的原子位置
                        target_idx = None
                        if group == 'phenol':
                            # 找O原子
                            for idx in match:
                                atom = mol.GetAtomWithIdx(idx)
                                if atom.GetSymbol() == 'O' and atom.GetTotalNumHs() > 0:
                                    target_idx = idx
                                    break
                        elif group == 'aniline':
                            # 找N原子
                            for idx in match:
                                atom = mol.GetAtomWithIdx(idx)
                                if atom.GetSymbol() == 'N' and atom.GetTotalNumHs() > 0:
                                    target_idx = idx
                                    break
                        elif group == 'pyrrole_indole':
                            # 找環N原子
                            for idx in match:
                                atom = mol.GetAtomWithIdx(idx)
                                if atom.GetSymbol() == 'N' and atom.GetTotalNumHs() > 0:
                                    target_idx = idx
                                    group_name = 'amine_secondary'  # 環N通常是二級胺
                                    break
                        elif group == 'carboxylic_acid':
                            # 找羧酸OH原子
                            for idx in match:
                                atom = mol.GetAtomWithIdx(idx)
                                if atom.GetSymbol() == 'O' and atom.GetTotalNumHs() > 0:
                                    target_idx = idx
                                    break
                        elif group == 'thiol':
                            # 找S原子
                            for idx in match:
                                atom = mol.GetAtomWithIdx(idx)
                                if atom.GetSymbol() == 'S' and atom.GetTotalNumHs() > 0:
                                    target_idx = idx
                                    break
                        elif group == 'phosphate':
                            # 找P-OH中的O原子
                            for idx in match:
                                atom = mol.GetAtomWithIdx(idx)
                                if atom.GetSymbol() == 'O' and atom.GetTotalNumHs() > 0:
                                    has_p_neighbor = False
                                    for neighbor in atom.GetNeighbors():
                                        if neighbor.GetSymbol() == 'P':
                                            has_p_neighbor = True
                                            break
                                    if has_p_neighbor:
                                        target_idx = idx
                                        break
                        
                        # 添加找到的原子
                        if target_idx is not None and not any(target_idx == x[0] for x in dissociable_atoms):
                            group_name = 'amine_secondary' if group == 'pyrrole_indole' else group
                            dissociable_atoms.append((target_idx, group_name))
            except Exception as e:
                print(f"處理芳香模式出錯 {pattern}: {e}")
                continue
    
    # 2. 分析所有環系統
    if not dissociable_atoms:
        try:
            rings = mol.GetSSSR()
            for ring in rings:
                # 檢查環上的每個原子
                for atom_idx in list(ring):
                    atom = mol.GetAtomWithIdx(atom_idx)
                    
                    # 找到可解離原子
                    if atom.GetSymbol() in ['O', 'N', 'S'] and atom.GetTotalNumHs() > 0:
                        # 根據原子類型和環境確定官能團
                        if atom.GetSymbol() == 'O':
                            if atom.GetIsAromatic():
                                dissociable_atoms.append((atom_idx, 'phenol'))
                            else:
                                # 檢查是否連接到羧酸
                                is_carboxylic = False
                                for neighbor in atom.GetNeighbors():
                                    if neighbor.GetSymbol() == 'C':
                                        for bond in neighbor.GetBonds():
                                            if (bond.GetOtherAtomIdx(neighbor.GetIdx()) != atom_idx and
                                                bond.GetBondType() == Chem.rdchem.BondType.DOUBLE and
                                                mol.GetAtomWithIdx(bond.GetOtherAtomIdx(neighbor.GetIdx())).GetSymbol() == 'O'):
                                                is_carboxylic = True
                                                break
                                if is_carboxylic:
                                    dissociable_atoms.append((atom_idx, 'carboxylic_acid'))
                                else:
                                    dissociable_atoms.append((atom_idx, 'alcohol'))
                        elif atom.GetSymbol() == 'N':
                            if atom.GetIsAromatic():
                                dissociable_atoms.append((atom_idx, 'amine_secondary'))
                            else:
                                if atom.GetTotalNumHs() >= 2:
                                    dissociable_atoms.append((atom_idx, 'amine_primary'))
                                else:
                                    dissociable_atoms.append((atom_idx, 'amine_secondary'))
                        elif atom.GetSymbol() == 'S':
                            dissociable_atoms.append((atom_idx, 'thiol'))
                
                # 檢查連接到環上的官能團
                for atom_idx in list(ring):
                    ring_atom = mol.GetAtomWithIdx(atom_idx)
                    
                    for neighbor in ring_atom.GetNeighbors():
                        if neighbor.GetIdx() not in list(ring):  # 不在環上的原子
                            n_idx = neighbor.GetIdx()
                            
                            # 檢查是否是可解離官能團
                            if neighbor.GetSymbol() in ['O', 'N', 'S'] and neighbor.GetTotalNumHs() > 0:
                                if neighbor.GetSymbol() == 'O':
                                    # 檢查是否是酚羥基
                                    if ring_atom.GetIsAromatic():
                                        dissociable_atoms.append((n_idx, 'phenol'))
                                    else:
                                        # 檢查是否是羧酸
                                        is_carboxylic = False
                                        for n2 in neighbor.GetNeighbors():
                                            if n2.GetSymbol() == 'C':
                                                for bond in n2.GetBonds():
                                                    if (bond.GetOtherAtomIdx(n2.GetIdx()) != n_idx and
                                                        bond.GetBondType() == Chem.rdchem.BondType.DOUBLE and
                                                        mol.GetAtomWithIdx(bond.GetOtherAtomIdx(n2.GetIdx())).GetSymbol() == 'O'):
                                                        is_carboxylic = True
                                                        break
                                        if is_carboxylic:
                                            dissociable_atoms.append((n_idx, 'carboxylic_acid'))
                                        else:
                                            dissociable_atoms.append((n_idx, 'alcohol'))
                                elif neighbor.GetSymbol() == 'N':
                                    if neighbor.GetTotalNumHs() >= 2:
                                        dissociable_atoms.append((n_idx, 'amine_primary'))
                                    else:
                                        dissociable_atoms.append((n_idx, 'amine_secondary'))
                                elif neighbor.GetSymbol() == 'S':
                                    dissociable_atoms.append((n_idx, 'thiol'))
        except Exception as e:
            print(f"分析環系統時出錯: {e}")
    
    # 3. 使用高級SMARTS分析環系統
    if not dissociable_atoms:
        advanced_patterns = [
            # 常見環系統
            ('c1[nH]cnc1', 'amine_secondary'),          # 咪唑
            ('c1cc[nH]c1', 'amine_secondary'),          # 吡咯
            ('c1c[nH]cn1', 'amine_secondary'),          # 咪唑
            ('c1c[nH]c2ccccc12', 'amine_secondary'),    # 吲哚
            ('c1ccc2[nH]ccc2c1', 'amine_secondary'),    # 吲哚
            
            # 常見氨基酸側鏈
            ('c1ccc(cc1)C[C@@H](N)C(O)=O', 'amine_primary'), # 苯丙氨酸
            ('c1cc(O)ccc1C[C@@H](N)C(O)=O', 'amine_primary'), # 酪氨酸
            ('c1c[nH]cn1C[C@@H](N)C(O)=O', 'amine_primary'),  # 組氨酸
            
            # 常見藥物結構
            ('c1ncc([nH]1)C(=O)', 'amine_secondary'),      # 嘧啶酮
            ('c1nc2c([nH]1)cccc2', 'amine_secondary'),     # 苯并咪唑
            ('c1cc2c(cc1)oc(=O)cc2', 'carboxylic_acid'),   # 香豆素
        ]
        
        for pattern_smarts, group_type in advanced_patterns:
            try:
                pattern = Chem.MolFromSmarts(pattern_smarts)
                if pattern and mol.HasSubstructMatch(pattern):
                    matches = mol.GetSubstructMatches(pattern)
                    for match in matches:
                        # 根據官能團類型找出目標原子
                        if group_type in ['amine_primary', 'amine_secondary']:
                            for idx in match:
                                atom = mol.GetAtomWithIdx(idx)
                                if atom.GetSymbol() == 'N' and atom.GetTotalNumHs() > 0:
                                    if not any(idx == x[0] for x in dissociable_atoms):
                                        dissociable_atoms.append((idx, group_type))
                                        break
                        elif group_type == 'carboxylic_acid':
                            for idx in match:
                                atom = mol.GetAtomWithIdx(idx)
                                if atom.GetSymbol() == 'O' and atom.GetTotalNumHs() > 0:
                                    # 檢查是否連接到C=O
                                    is_carboxylic = False
                                    for neighbor in atom.GetNeighbors():
                                        if neighbor.GetSymbol() == 'C':
                                            for bond in neighbor.GetBonds():
                                                if (bond.GetOtherAtomIdx(neighbor.GetIdx()) != idx and
                                                    bond.GetBondType() == Chem.rdchem.BondType.DOUBLE and
                                                    mol.GetAtomWithIdx(bond.GetOtherAtomIdx(neighbor.GetIdx())).GetSymbol() == 'O'):
                                                    is_carboxylic = True
                                                    break
                                    if is_carboxylic and not any(idx == x[0] for x in dissociable_atoms):
                                        dissociable_atoms.append((idx, 'carboxylic_acid'))
                                        break
            except Exception as e:
                print(f"高級SMARTS分析出錯 {pattern_smarts}: {e}")
                continue
    
    return dissociable_atoms

def analyze_heterocyclic_rings(mol):
    """專門針對雜環結構的解析函數，涵蓋常見生物分子中的各種雜環"""
    dissociable_atoms = []
    
    # 列出常見雜環SMARTS模式和對應官能團
    heterocyclic_patterns = [
        # 含氮雜環
        ('[nH]1cccc1', 'amine_secondary'),         # 吡咯
        ('[nH]1cncc1', 'amine_secondary'),         # 咪唑
        ('[nH]1ccccc1', 'amine_secondary'),        # 吡啶
        ('[nH]1nccn1', 'amine_secondary'),         # 三唑
        ('[nH]1c2ccccc2cc1', 'amine_secondary'),   # 吲哚
        ('[nH]1c2ccccc2c1', 'amine_secondary'),    # 異吲哚
        ('[nH]1c(=O)nc(=O)cc1', 'amine_secondary'),# 尿嘧啶
        ('[nH]1c(=O)nc(=O)c2[nH]cnc12', 'amine_secondary'), # 鳥嘌呤
        
        # 含氧雜環
        ('c1c[o]cc1', 'alcohol'),                  # 呋喃
        ('c1cc[o]c1', 'alcohol'),                  # 呋喃
        ('C1OC(=O)C=C1', 'carboxylic_acid'),       # 丁烯內酯
        
        # 含硫雜環
        ('c1c[s]cc1', 'thiol'),                    # 噻吩
        ('c1cc[s]c1', 'thiol'),                    # 噻吩
        
        # 縮合環系統
        ('c1ccc2c(c1)cnc[nH]2', 'amine_secondary'),# 苯并咪唑
        ('c1ccc2c(c1)[nH]cnc2', 'amine_secondary'),# 苯并吡咯
        ('c1ccc2c(c1)[nH]cc2', 'amine_secondary'), # 吲哚
        ('c1ccc2c(c1)oc(=O)c2', 'carboxylic_acid'),# 香豆素
    ]
    
    # 嘗試每個模式
    for pattern, group_type in heterocyclic_patterns:
        try:
            patt = Chem.MolFromSmarts(pattern)
            if patt is None:
                continue
                
            if mol.HasSubstructMatch(patt):
                matches = mol.GetSubstructMatches(patt)
                for match in matches:
                    # 根據環類型找出關鍵原子
                    for atom_idx in match:
                        atom = mol.GetAtomWithIdx(atom_idx)
                        
                        # 針對不同類型環中的解離位點
                        if (group_type == 'amine_secondary' and atom.GetSymbol() == 'N' and atom.GetTotalNumHs() > 0) or \
                           (group_type == 'alcohol' and atom.GetSymbol() == 'O' and atom.GetTotalNumHs() > 0) or \
                           (group_type == 'thiol' and atom.GetSymbol() == 'S' and atom.GetTotalNumHs() > 0) or \
                           (group_type == 'carboxylic_acid' and atom.GetSymbol() == 'O' and 
                            any(n.GetSymbol() == 'C' and any(b.GetBondType() == Chem.rdchem.BondType.DOUBLE for b in n.GetBonds()) 
                                for n in atom.GetNeighbors())):
                            
                            # 避免重複添加
                            if not any(atom_idx == x[0] for x in dissociable_atoms):
                                dissociable_atoms.append((atom_idx, group_type))
                                break
        except Exception as e:
            print(f"處理雜環模式時出錯: {pattern}, {e}")
            continue
    
    # 特殊處理含氮雜環
    if not dissociable_atoms:
        try:
            # 尋找環形結構
            rings = mol.GetSSSR()
            for ring in rings:
                ring_atoms = list(ring)
                
                # 尋找環中的N原子
                for idx in ring_atoms:
                    atom = mol.GetAtomWithIdx(idx)
                    if atom.GetSymbol() == 'N' and atom.GetTotalNumHs() > 0:
                        if atom.GetIsAromatic():
                            # 芳香N通常是吡咯型氮，是二級胺
                            dissociable_atoms.append((idx, 'amine_secondary'))
                        else:
                            # 根據氫原子數決定
                            if atom.GetTotalNumHs() >= 2:
                                dissociable_atoms.append((idx, 'amine_primary'))
                            else:
                                dissociable_atoms.append((idx, 'amine_secondary'))
        except Exception as e:
            print(f"處理環中氮原子時出錯: {e}")
    
    return dissociable_atoms

def comprehensive_atom_scan(mol):
    """最全面的原子掃描，尋找所有可能的解離位點"""
    dissociable_atoms = []
    
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        symbol = atom.GetSymbol()
        
        # 只關注可能解離的原子類型
        if symbol not in ['O', 'N', 'S', 'P']:
            continue
            
        # 確認帶有氫原子(大多數情況下解離需要)
        try:
            total_hs = atom.GetTotalNumHs()
        except:
            print(f"無法獲取原子 {idx} ({symbol}) 的氫原子數，假定為0")
            total_hs = 0
            
        if total_hs == 0 and symbol != 'P':
            # 無氫原子通常不解離，但還是檢查一些特殊情況
            if symbol == 'O' and atom.GetFormalCharge() == -1:  # 已經解離的羧酸鹽
                c_neighbors = [n for n in atom.GetNeighbors() if n.GetSymbol() == 'C']
                if c_neighbors:
                    for c in c_neighbors:
                        has_carbonyl = False
                        for bond in c.GetBonds():
                            if (bond.GetOtherAtomIdx(c.GetIdx()) != idx and
                                bond.GetBondType() == Chem.rdchem.BondType.DOUBLE and
                                mol.GetAtomWithIdx(bond.GetOtherAtomIdx(c.GetIdx())).GetSymbol() == 'O'):
                                has_carbonyl = True
                                break
                        if has_carbonyl:
                            dissociable_atoms.append((idx, 'carboxylic_acid'))
                            break
            continue
            
        # 根據原子類型和環境判斷官能團
        group = determine_functional_group_advanced(atom, mol)
        if group != 'unknown':
            dissociable_atoms.append((idx, group))
    
    return dissociable_atoms

def determine_functional_group_advanced(atom, mol):
    """進階版官能團類型判定，考慮更多化學結構"""
    symbol = atom.GetSymbol()
    idx = atom.GetIdx()
    
    # 氧原子判斷
    if symbol == 'O':
        # 檢查是否是羧酸
        c_neighbors = [n for n in atom.GetNeighbors() if n.GetSymbol() == 'C']
        if c_neighbors:
            for c in c_neighbors:
                has_carbonyl = False
                for bond in c.GetBonds():
                    if (bond.GetOtherAtomIdx(c.GetIdx()) != idx and
                        bond.GetBondType() == Chem.rdchem.BondType.DOUBLE and
                        mol.GetAtomWithIdx(bond.GetOtherAtomIdx(c.GetIdx())).GetSymbol() == 'O'):
                        has_carbonyl = True
                        break
                if has_carbonyl:
                    return 'carboxylic_acid'
                    
        # 檢查是否是酚
        if any(n.GetIsAromatic() for n in atom.GetNeighbors()):
            return 'phenol'
        
        # 檢查是否是磷酸
        p_neighbors = [n for n in atom.GetNeighbors() if n.GetSymbol() == 'P']
        if p_neighbors:
            return 'phosphate'
            
        # 否則是醇
        return 'alcohol'
        
    # 氮原子判斷
    elif symbol == 'N':
        # 檢查是否是雜環中的氮
        if atom.IsInRing() and atom.GetIsAromatic():
            return 'amine_secondary'
            
        # 根據氫原子數判斷
        total_hs = atom.GetTotalNumHs()
        if total_hs >= 2:
            return 'amine_primary'
        elif total_hs == 1:
            return 'amine_secondary'
        else:
            return 'amine_tertiary'
            
    # 硫原子判斷
    elif symbol == 'S':
        # 檢查是否是硫酸
        o_neighbors = [n for n in atom.GetNeighbors() if n.GetSymbol() == 'O']
        if len(o_neighbors) >= 2:
            double_bonded = any(atom.GetBondBetweenAtoms(atom.GetIdx(), n.GetIdx()).GetBondType() == Chem.rdchem.BondType.DOUBLE 
                               for n in o_neighbors)
            if double_bonded:
                return 'sulfonic_acid'
        
        return 'thiol'
        
    # 磷原子判斷
    elif symbol == 'P':
        return 'phosphate'
    
    # 未知類型
    return 'unknown'

def analyze_smiles_directly(smiles):
    """當所有結構分析方法失敗時，直接解析SMILES字符串來定位可能的解離位點"""
    dissociable_atoms = []
    
    # 常見結構直接映射
    direct_mappings = {
        # 基本氨基酸
        'NCC(O)=O': [(0, 'amine_primary'), (2, 'carboxylic_acid')],  # 甘氨酸
        'NCC(=O)O': [(0, 'amine_primary'), (3, 'carboxylic_acid')],  # 甘氨酸另一種表示
        'C[C@H](N)C(O)=O': [(2, 'amine_primary'), (3, 'carboxylic_acid')],  # 丙氨酸
        'C[C@@H](N)C(O)=O': [(2, 'amine_primary'), (3, 'carboxylic_acid')],  # D-丙氨酸
        'NC(C)C(O)=O': [(0, 'amine_primary'), (3, 'carboxylic_acid')],  # 丙氨酸變體
        'NC(CC(O)=O)C(O)=O': [(0, 'amine_primary'), (4, 'carboxylic_acid'), (8, 'carboxylic_acid')],  # 天冬氨酸
        'N[C@@H](CC(O)=O)C(O)=O': [(0, 'amine_primary'), (4, 'carboxylic_acid'), (8, 'carboxylic_acid')],  # L-天冬氨酸
        'N[C@@H](CCC(O)=O)C(O)=O': [(0, 'amine_primary'), (5, 'carboxylic_acid'), (9, 'carboxylic_acid')],  # L-谷氨酸
        'NC(CCC(O)=O)C(O)=O': [(0, 'amine_primary'), (5, 'carboxylic_acid'), (9, 'carboxylic_acid')],  # 谷氨酸(無手性)
        
        # 含芳香環的氨基酸
        'NC(Cc1ccccc1)C(O)=O': [(0, 'amine_primary'), (5, 'carboxylic_acid')],  # 苯丙氨酸(無手性)
        'N[C@@H](Cc1ccccc1)C(O)=O': [(0, 'amine_primary'), (5, 'carboxylic_acid')],  # L-苯丙氨酸
        'N[C@H](Cc1ccccc1)C(O)=O': [(0, 'amine_primary'), (5, 'carboxylic_acid')],  # D-苯丙氨酸
        'N[C@@H](Cc1ccc(O)cc1)C(O)=O': [(0, 'amine_primary'), (5, 'carboxylic_acid')],  # L-酪氨酸
        'NC(Cc1ccc(O)cc1)C(O)=O': [(0, 'amine_primary'), (5, 'carboxylic_acid')],  # 酪氨酸(無手性)
        
        # 含雜環的氨基酸
        'N[C@@H](Cc1c[nH]cn1)C(O)=O': [(0, 'amine_primary'), (5, 'carboxylic_acid')],  # L-組氨酸
        'NC(Cc1c[nH]cn1)C(O)=O': [(0, 'amine_primary'), (5, 'carboxylic_acid')],  # 組氨酸(無手性)
        'N[C@@H](Cc1c[nH]c2ccccc12)C(O)=O': [(0, 'amine_primary'), (5, 'carboxylic_acid')],  # L-色氨酸
        'NC(Cc1c[nH]c2ccccc12)C(O)=O': [(0, 'amine_primary'), (5, 'carboxylic_acid')],  # 色氨酸(無手性)
        
        # 胺基酸衍生物
        'OC(=O)CCc1c[nH]cn1': [(1, 'carboxylic_acid')],  # 組氨酸衍生物
        'OC(=O)CCc1c[nH]c2ccccc12': [(1, 'carboxylic_acid')],  # 色氨酸衍生物
        'N[C@@H](CS)C(O)=O': [(0, 'amine_primary'), (3, 'carboxylic_acid')],  # L-半胱氨酸
        'N[C@@H](CSSC[C@H](N)C(O)=O)C(O)=O': [(0, 'amine_primary'), (9, 'amine_primary'), (3, 'carboxylic_acid'), (12, 'carboxylic_acid')],  # 胱氨酸
        
        # 環狀結構
        'c1cc[nH]c1': [(3, 'amine_secondary')],  # 吡咯
        'c1cn[nH]c1': [(3, 'amine_secondary')],  # 吡唑
        'c1cc[nH]cc1': [(3, 'amine_secondary')],  # 咪唑
        'c1c[nH]c2ccccc12': [(2, 'amine_secondary')],  # 吲哚
        'c1ccc2c(c1)[nH]cc2': [(6, 'amine_secondary')],  # 吲哚(另一種表示)
        
        # 簡單有機酸
        'OC(=O)C(=O)O': [(0, 'carboxylic_acid'), (4, 'carboxylic_acid')],  # 草酸
        'OC(=O)CC(O)=O': [(0, 'carboxylic_acid'), (5, 'carboxylic_acid')],  # 丙二酸
        'OC(=O)CCC(O)=O': [(0, 'carboxylic_acid'), (6, 'carboxylic_acid')],  # 丁二酸
        'OC(=O)CCCC(O)=O': [(0, 'carboxylic_acid'), (7, 'carboxylic_acid')],  # 戊二酸
        
        # 常見藥物和天然產物
        'c1c(O)cc(O)cc1O': [(2, 'phenol'), (5, 'phenol'), (8, 'phenol')],  # 間苯三酚
        'c1cc(O)cc(O)c1O': [(3, 'phenol'), (6, 'phenol'), (8, 'phenol')],  # 間苯三酚(另一種表示)
        'OC(=O)c1ccccc1O': [(1, 'carboxylic_acid'), (8, 'phenol')],  # 水楊酸
        'OC(=O)c1ccccc1C(O)=O': [(1, 'carboxylic_acid'), (9, 'carboxylic_acid')],  # 鄰苯二甲酸
    }
    
    # 檢查完全匹配
    if smiles in direct_mappings:
        return direct_mappings[smiles]
    
    # 檢查部分匹配
    for pattern, mapping in direct_mappings.items():
        if pattern in smiles:
            return [(idx + smiles.find(pattern), group) for idx, group in mapping]
    
    # 查找常見官能團
    group_patterns = {
        'carboxylic_acid': [
            ('C(O)=O', 2),      # 羧酸：O的位置偏移
            ('C(=O)O', 4),      # 羧酸另一表示：O的位置偏移
            ('COOH', 3),        # 羧酸簡寫：O的位置偏移
        ],
        'amine_primary': [
            ('NH2', 0),         # 一級胺：N的位置偏移
            ('N(-[H])(-[H])', 0), # 一級胺顯式氫：N的位置偏移
        ],
        'amine_secondary': [
            ('NH1', 0),         # 二級胺：N的位置偏移
            ('N(-[H])', 0),     # 二級胺顯式氫：N的位置偏移
            ('[nH]', 1),        # 環氮：n的位置偏移
        ],
        'phenol': [
            ('cOH', 1),         # 酚羥基：O的位置偏移
            ('c-OH', 2),        # 酚羥基連字符：O的位置偏移
            ('c(OH)', 2),       # 酚羥基帶括號：O的位置偏移
        ],
        'alcohol': [
            ('COH', 1),         # 醇羥基：O的位置偏移
            ('C-OH', 2),        # 醇羥基連字符：O的位置偏移
            ('C(OH)', 2),       # 醇羥基帶括號：O的位置偏移
        ],
        'thiol': [
            ('SH', 0),          # 硫醇：S的位置偏移
            ('S-H', 0),         # 硫醇連字符：S的位置偏移
        ],
    }
    
    # 搜索所有模式
    for group, patterns in group_patterns.items():
        for pattern, offset in patterns:
            start = 0
            while True:
                pos = smiles.find(pattern, start)
                if pos == -1:
                    break
                
                idx = pos + offset
                dissociable_atoms.append((idx, group))
                start = pos + 1
    
    # 特殊處理雜環結構
    if ('1' in smiles or '2' in smiles) and any(char in smiles for char in ['n', 'o', 's']):
        # 尋找可能的雜環位置
        positions = []
        
        # 尋找環氮
        pos = smiles.find('[nH]')
        if pos != -1:
            positions.append((pos + 1, 'amine_secondary'))
        
        # 尋找吲哚或吡咯模式
        indole_patterns = ['c1c[nH]', 'c1cc[nH]', 'c1ccc2[nH]']
        for pattern in indole_patterns:
            pos = smiles.find(pattern)
            if pos != -1:
                n_pos = pos + pattern.find('[nH]') + 1
                positions.append((n_pos, 'amine_secondary'))
        
        # 添加找到的位置
        for pos, group in positions:
            if not any(pos == x[0] for x in dissociable_atoms):
                dissociable_atoms.append((pos, group))
    
    return dissociable_atoms

def basic_atom_analysis(mol):
    """最基本的原子分析，當所有方法都失敗時使用"""
    dissociable_atoms = []
    
    # 簡單規則來識別可能的解離位點
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        symbol = atom.GetSymbol()
        
        if symbol == 'O':
            try:
                if atom.GetTotalNumHs() > 0:
                    dissociable_atoms.append((idx, 'alcohol'))  # 默認為醇
            except:
                # 如果無法確定氫數，檢查是否連接到碳
                c_neighbors = [n for n in atom.GetNeighbors() if n.GetSymbol() == 'C']
                if c_neighbors:
                    dissociable_atoms.append((idx, 'alcohol'))
                    
        elif symbol == 'N':
            try:
                if atom.GetTotalNumHs() > 0:
                    if atom.GetTotalNumHs() >= 2:
                        dissociable_atoms.append((idx, 'amine_primary'))
                    else:
                        dissociable_atoms.append((idx, 'amine_secondary'))
            except:
                # 如果無法確定氫數，簡單地添加為胺
                dissociable_atoms.append((idx, 'amine_primary'))
                
        elif symbol == 'S':
            try:
                if atom.GetTotalNumHs() > 0:
                    dissociable_atoms.append((idx, 'thiol'))
            except:
                # 如果無法確定氫數，檢查連接
                c_neighbors = [n for n in atom.GetNeighbors() if n.GetSymbol() == 'C']
                if c_neighbors:
                    dissociable_atoms.append((idx, 'thiol'))
    
    return dissociable_atoms

def sanitize_smiles(smiles):
    """嘗試多種策略修復SMILES字符串"""
    # 原始SMILES
    if Chem.MolFromSmiles(smiles) is not None:
        return smiles
    
    # 嘗試修復策略
    
    # 1. 移除可能的問題字符
    sanitized = smiles.replace('[H]', '').replace('\\', '').replace('/', '')
    if Chem.MolFromSmiles(sanitized) is not None:
        return sanitized
    
    # 2. 標準化芳香表示
    sanitized = re.sub(r'c1([cC]+)c', 'c1\\1c', smiles)
    if Chem.MolFromSmiles(sanitized) is not None:
        return sanitized
    
    # 3. 修復可能的錯誤括號
    open_count = smiles.count('(')
    close_count = smiles.count(')')
    
    if open_count > close_count:
        sanitized = smiles + ')' * (open_count - close_count)
    elif close_count > open_count:
        sanitized = '(' * (close_count - open_count) + smiles
        
    if Chem.MolFromSmiles(sanitized) is not None:
        return sanitized
    
    # 4. 嘗試去除特殊字符
    sanitized = re.sub(r'[^A-Za-z0-9\(\)\[\]\.\+\-=#:]', '', smiles)
    if Chem.MolFromSmiles(sanitized) is not None:
        return sanitized
    
    # 如果所有修復都失敗，返回原始SMILES
    return smiles 