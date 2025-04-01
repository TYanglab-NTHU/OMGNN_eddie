import pandas as pd
import os
import re
from rdkit import Chem
from tqdm import tqdm
import sys
import torch
import logging
from typing import Optional

# 全局日誌控制
VERBOSE = False
LOGGER = None

def set_verbose(enable: bool = False) -> None:
    """
    設置是否啟用詳細日誌輸出
    
    參數:
        enable (bool): 是否啟用詳細日誌
    """
    global VERBOSE, LOGGER
    VERBOSE = enable
    
    # 設置日誌
    if VERBOSE and LOGGER is None:
        LOGGER = logging.getLogger("nist_converter")
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)
        LOGGER.setLevel(logging.INFO)

def log(message: str, level: str = 'info') -> None:
    """
    根據當前日誌設置輸出訊息
    
    參數:
        message (str): 要輸出的訊息
        level (str): 日誌級別，可以是 'debug'、'info'、'warning' 或 'error'
    """
    if not VERBOSE:
        return
    
    if LOGGER:
        if level == 'debug':
            LOGGER.debug(message)
        elif level == 'info':
            LOGGER.info(message)
        elif level == 'warning':
            LOGGER.warning(message)
        elif level == 'error':
            LOGGER.error(message)
    else:
        # 如果沒有正確初始化logger，但VERBOSE為True，則使用print
        print(f"[{level.upper()}] {message}")

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
    
    # 基本吡啶結構
    'c1ccccn1': [(5, 'pyridine_nitrogen')],  # 吡啶
    'c1cccnc1': [(4, 'pyridine_nitrogen')],  # 吡啶異構體
    'c1ccncc1': [(3, 'pyridine_nitrogen')],  # 吡嗪
    
    # 常見取代吡啶
    'Cc1ccccn1': [(6, 'pyridine_nitrogen')],  # 2-甲基吡啶（甲基吡啶）
    'CCc1ccccn1': [(7, 'pyridine_nitrogen')],  # 2-乙基吡啶
    'Cc1cccnc1': [(5, 'pyridine_nitrogen')],  # 3-甲基吡啶
    'Cc1ccncc1': [(4, 'pyridine_nitrogen')],  # 甲基吡嗪
    
    # 縮合環吡啶
    'c1ccc2ncccc2c1': [(5, 'pyridine_nitrogen')],  # 喹啉
    'c1ccnc2ccccc12': [(3, 'pyridine_nitrogen')],  # 異喹啉
    
    # 多雜環系統
    's1cccc1c2ccccn2': [(11, 'pyridine_nitrogen')],  # 噻吩-吡啶
    'c1cccc(c1)c2ccccn2': [(12, 'pyridine_nitrogen')],  # 苯基吡啶
    'c1ccc(nc1)c2ccccc2': [(4, 'pyridine_nitrogen')],  # 2-苯基吡啶
    
    # 咪唑類結構
    'Cn1ccnc1': [(1, 'imidazole_nitrogen')],      # N-甲基咪唑
    'CCn1ccnc1': [(2, 'imidazole_nitrogen')],     # N-乙基咪唑
    'CCCn1ccnc1': [(3, 'imidazole_nitrogen')],    # N-丙基咪唑
    'C=Cn1ccnc1': [(3, 'imidazole_nitrogen')],    # N-乙烯基咪唑
    
    # 三唑類結構
    'Cn1cncn1': [(1, 'triazole_nitrogen')],       # N-甲基三唑
    'Cn1ncnc1': [(1, 'triazole_nitrogen')],       # N-甲基異三唑
    'Cn1cncc1': [(1, 'triazole_nitrogen')],       # N-甲基吡唑
    
    # 含硫化合物
    's1ccnc1': [(0, 'thiazole_sulfur')],          # 噻唑
    's1cccc1': [(0, 'thiophene_sulfur')],         # 噻吩
    'SC([S-])=S': [(0, 'thione_sulfur')],         # 硫代硫酸
    
    # 含硒化合物
    'CC([SeH])CN': [(3, 'selenium_hydride')],     # 硒代醚
    '[SeH]C([Se-])=[Se]': [(0, 'selenium_hydride')], # 硒化物
    'NCC[SeH]': [(3, 'selenium_hydride')],        # 胺硒醇
    
    # 含砷化合物
    'CCCC[As](O)(=O)CCCC': [(5, 'arsenic_acid')], # 砷酸酯
    'CCCCC[As](O)(=O)CCCCC': [(6, 'arsenic_acid')], # 砷酸酯
    'C[As](C)(O)=O': [(1, 'arsenic_acid')],       # 甲基砷酸
    
    # N-氧化物
    'C[N+]([O-])=O': [(1, 'n_oxide')],            # 甲基硝基
    'CC[N+]([O-])=O': [(2, 'n_oxide')],           # 乙基硝基
    'CCC[N+]([O-])=O': [(3, 'n_oxide')],          # 丙基硝基
    'CC(C)[N+]([O-])=O': [(2, 'n_oxide')],        # 異丙基硝基
    
    # 氰基化合物
    'C#N': [(0, 'nitrile')],                      # 氰基
    
    # 複雜含氮雜環
    'C1CN2CCN1CC2': [(1, 'tertiary_amine'), (4, 'tertiary_amine')], # 1,4-二氮雜庚烷
    'C1CN2CCC1CC2': [(1, 'tertiary_amine'), (3, 'tertiary_amine')], # 1,4-二氮雜己烷
    
    # 縮合雜環系統
    'Cn1cnc2ccccc12': [(1, 'imidazole_nitrogen')], # N-甲基苯并咪唑
    'Cn1cnc2cncnc12': [(1, 'imidazole_nitrogen')], # 嘌呤衍生物
}

# 針對吡啶衍生物的映射
PYRIDINE_DERIVATIVES = {
    # 聯吡啶類 (Bipyridines)
    'c1ccc(nc1)c2ccncc2': [(4, 'pyridine_nitrogen'), (10, 'pyridine_nitrogen')],  # 2,4'-Bipyridyl
    'c1cncc(c1)c2cccnc2': [(2, 'pyridine_nitrogen'), (9, 'pyridine_nitrogen')],   # 3,3'-Bipyridyl
    'c1cncc(c1)c2ccncc2': [(2, 'pyridine_nitrogen'), (9, 'pyridine_nitrogen')],   # 3,4'-Bipyridyl
    'c1ccc(nc1)c2cccnc2': [(4, 'pyridine_nitrogen'), (10, 'pyridine_nitrogen')],  # 2,3'-Bipyridyl
    
    # 菲咯啉類 (Phenanthrolines)
    'c1cnc2ccc3ncccc3c2c1': [(2, 'pyridine_nitrogen'), (10, 'pyridine_nitrogen')], # 4,7-Phenanthroline
    'Cc1cc2cccnc2c3ncccc13': [(7, 'pyridine_nitrogen'), (12, 'pyridine_nitrogen')], # 5-Methyl-1,10-phenanthroline
    'Cc1cnc2c(c1)c(C)c(C)c3cc(C)cnc23': [(2, 'pyridine_nitrogen'), (19, 'pyridine_nitrogen')], # 3,5,6,8-Tetramethyl-1,10-phenanthroline
    '[O-][N+](=O)c1cc2cccnc2c3ncccc13': [(12, 'pyridine_nitrogen'), (17, 'pyridine_nitrogen')], # 5-Nitro-1,10-phenanthroline
    'Cc1c(C)c2cccnc2c3ncccc13': [(8, 'pyridine_nitrogen'), (13, 'pyridine_nitrogen')], # 5,6-Dimethyl-1,10-phenanthroline
    'Cc1ccnc2c1ccc3c(C)ccnc23': [(4, 'pyridine_nitrogen'), (15, 'pyridine_nitrogen')], # 4,7-Dimethyl-1,10-phenanthroline
    'c1cnc2c(c1)ccc3ncccc23': [(2, 'pyridine_nitrogen'), (13, 'pyridine_nitrogen')], # 1,10-Phenanthroline
    'Brc1cc2cccnc2c3ncccc13': [(8, 'pyridine_nitrogen'), (13, 'pyridine_nitrogen')], # 5-Bromo-1,10-phenanthroline
    
    # 含官能團的吡啶類 (Pyridines with functional groups)
    'CN1C=CC(=O)C(=C1C)O': [(1, 'tertiary_amine'), (8, 'hydroxyl_group')], # 1,2-Dimethyl-3,4-dihydroxypyridine
    'ON1C=CC=CC1=O': [(0, 'hydroxyl_group'), (1, 'tertiary_amine'), (7, 'carbonyl')], # 1-Hydroxy-1,2-dihydropyridin-2-one
    'CCN1C=CC(=O)C(=C1CC)O': [(2, 'tertiary_amine'), (11, 'hydroxyl_group')], # 1,2-Diethyl-3,4-dihydroxypyridine
    'O=C1NC=CC=C1': [(3, 'amine_secondary'), (1, 'carbonyl')], # 2-Hydroxypyridine
    'O=C1C=CNC=C1': [(5, 'amine_secondary'), (1, 'carbonyl')], # 4-Hydroxypyridine
    'Oc1cnccc1C=O': [(0, 'phenol'), (2, 'pyridine_nitrogen')], # 3-Hydroxypyridine-4-carboxaldehyde
    'COc1cnccc1C=O': [(3, 'pyridine_nitrogen')], # 3-Methoxypyridine-4-carboxaldehyde
    
    # 喹啉類 (Quinolines)
    'Oc1cccc2cccnc12': [(0, 'phenol'), (9, 'pyridine_nitrogen')], # 8-Hydroxyquinoline
    'Oc1c(Br)cc(Br)c2cccnc12': [(0, 'phenol'), (10, 'pyridine_nitrogen')], # 5,7-Dibromo-8-hydroxyquinoline
    'Oc1c(I)cc(I)c2cccnc12': [(0, 'phenol'), (10, 'pyridine_nitrogen')], # 5,7-Diiodo-8-hydroxyquinoline
    
    # 含羧基的雜環類 (Heterocyclic carboxylic acids)
    'OC(=O)CNc1cccc2cccnc12': [(1, 'carboxylic_acid'), (4, 'amine_secondary'), (12, 'pyridine_nitrogen')], # N-(8-Quinolyl)glycine
    'OC(=O)c1cccc2cccnc12': [(1, 'carboxylic_acid'), (10, 'pyridine_nitrogen')], # Quinoline-8-carboxylic acid
    'OC(=O)c1cncc(c1)C(O)=O': [(1, 'carboxylic_acid'), (8, 'carboxylic_acid'), (3, 'pyridine_nitrogen')], # Pyridine-3,5-dicarboxylic acid
    
    # 環狀酮類 (Cyclic ketones)
    'CC1(C)CC(=O)CC(=O)C1': [(5, 'carbonyl'), (8, 'carbonyl')], # 5,5-Dimethylcyclohexane-1,3-dione
    'CC(C)C1=CC(=O)C(=CC=C1)O': [(7, 'carbonyl'), (9, 'phenol')], # beta-Isopropyltropolone
    'OC1=CC=CC=CC1=O': [(0, 'phenol'), (8, 'carbonyl')], # Tropolone
    
    # 具有特殊取代基的酮類 (Ketones with special substituents)
    'O=C(CC(=O)c1occc1)c2occc2': [(1, 'carbonyl'), (5, 'carbonyl'), (9, 'ether'), (15, 'ether')], # Difuranoylmethane
    'CC(=O)CC(=O)c1occc1': [(1, 'carbonyl'), (5, 'carbonyl'), (9, 'ether')], # Furanoylacetone
    'FC(F)(F)C(=O)CC(=O)c1occc1': [(5, 'carbonyl'), (9, 'carbonyl'), (13, 'ether')], # Furanoyltrifluoroacetone
    'FC(F)(F)C(=O)CC(=O)c1ccc2ccccc2c1': [(5, 'carbonyl'), (9, 'carbonyl')], # 2-Naphthoyltrifluoroacetone
    
    # 其他特殊結構
    'OOc1ccc(O)cc1': [(0, 'hydroxyl_group'), (1, 'hydroxyl_group'), (5, 'phenol')], # 1,2,4-Trihydroxybenzene
    'Oc1cccc(ON=O)c1': [(0, 'phenol'), (6, 'hydroxyl_group')], # 1,3-Dihydroxy-4-nitrosobenzene
    'ON=Cc1ccccc1O': [(0, 'hydroxyl_group'), (7, 'phenol')], # 2-Hydroxybenzaldehyde oxime
    '[nH]1cnc2ncncc12': [(1, 'amine_secondary'), (5, 'heterocyclic_nitrogen'), (8, 'heterocyclic_nitrogen')], # Purine
}

# 將這些特殊映射加入到主映射表中
SPECIAL_STRUCTURES_MAP.update(PYRIDINE_DERIVATIVES)

# 添加剩餘的吡啶(azines)類分子
PYRIDINE_AZINES = {
    # 甲基取代的吡啶
    'Cc1cncc(C)c1': [(2, 'pyridine_nitrogen')],                  # 3,5-Dimethylpyridine
    'COc1cc(C)nc(C)c1': [(4, 'pyridine_nitrogen')],              # 2,6-Dimethyl-4-methoxypyridine
    'Cc1ccnc(C)c1': [(3, 'pyridine_nitrogen')],                  # 2,4-Dimethylpyridine
    'CC(=O)c1c(C)cc(C)nc1C': [(6, 'pyridine_nitrogen')],         # 3-Acetyl-2,4,6-trimethylpyridine
    'CCOC(=O)c1cc(C)nc(C)c1': [(6, 'pyridine_nitrogen')],        # 2,6-Dimethylpyridine-4-carboxylic acid ethyl ester
    'Cc1ccc(C)nc1': [(5, 'pyridine_nitrogen')],                  # 2,5-Dimethylpyridine
    'CCc1ccc(C)nc1': [(6, 'pyridine_nitrogen')],                 # 5-Ethyl-2-methylpyridine
    'CCN1C=CC(=O)C(=C1C)O': [(2, 'tertiary_amine'), (9, 'hydroxyl_group')], # 1-Ethyl-2-methyl-3,4-dihydroxypyridine
    'CCc1cnccc1C': [(3, 'pyridine_nitrogen')],                   # 3-Ethyl-4-methylpyridine
    'Cc1cc(C)nc(C)c1': [(4, 'pyridine_nitrogen')],               # 2,4,6-Trimethylpyridine
    'c1ccc(cc1)c2ccncc2': [(8, 'pyridine_nitrogen')],            # 4-Phenylpyridine
    'CCCCCCN1C=CC(=O)C(=C1C)O': [(6, 'tertiary_amine'), (13, 'hydroxyl_group')], # 1-Hexyl-2-methyl-3,4-dihydroxypyridine
    'c1ccc(cc1)c2cccnc2': [(9, 'pyridine_nitrogen')],            # 3-Phenylpyridine
    'c1ccnnc1': [(4, 'pyridine_nitrogen')],                      # 1,2-Diazine (Pyridazine)
    
    # 複雜的喹啉衍生物
    '[Na+].[Na+].[O-][S](=O)(=O)c1ccc(NN=C2C=C(c3cccnc3C2=O)[S]([O-])(=O)=O)c4ccccc14': 
        [(28, 'pyridine_nitrogen'), (38, 'sulfonic_acid'), (44, 'sulfonic_acid')], # 8-Hydroxy-7-(4-sulfo-1-naphthylazo)quinoline-5-sulfonic acid
}

# 更新主映射表
SPECIAL_STRUCTURES_MAP.update(PYRIDINE_AZINES)

# ========== 從pka_chemutils.py 導入的核心功能 ==========

# 添加 onek_encoding_unk 函數
def onek_encoding_unk(value, allowable_set):
    """
    將單一值轉換為獨熱編碼，如果值不在允許集合中則返回額外的「未知」位。
    
    參數:
        value: 要編碼的值
        allowable_set: 允許值的集合
        
    返回:
        list: 獨熱編碼列表
    """
    encoding = [0] * (len(allowable_set) + 1)
    if value in allowable_set:
        encoding[allowable_set.index(value)] = 1
    else:
        encoding[-1] = 1  # 未知值
    return encoding

# 添加 atom_features 函數
def atom_features(atom):
    """
    提取原子的特徵。
    
    參數:
        atom: RDKit原子對象
        
    返回:
        list: 原子特徵列表
    """
    # 基本原子類型
    possible_atoms = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I']
    atom_type = onek_encoding_unk(atom.GetSymbol(), possible_atoms)
    
    # 原子電荷
    formal_charge = onek_encoding_unk(atom.GetFormalCharge(), [-1, 0, 1])
    
    # 雜化狀態
    hybridization_types = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3
    ]
    hybridization = onek_encoding_unk(atom.GetHybridization(), hybridization_types)
    
    # 隱含氫原子數
    implicit_h = onek_encoding_unk(atom.GetNumImplicitHs(), [0, 1, 2, 3, 4])
    
    # 是否在環中
    is_in_ring = [int(atom.IsInRing())]
    
    # 芳香性
    is_aromatic = [int(atom.GetIsAromatic())]
    
    # 原子度數
    degree = onek_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6])
    
    # 顯式價電子數
    explicit_valence = onek_encoding_unk(atom.GetExplicitValence(), [0, 1, 2, 3, 4, 5, 6])
    
    # 隱含價電子數
    implicit_valence = onek_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6])
    
    # 組合所有特徵
    features = atom_type + formal_charge + hybridization + implicit_h + is_in_ring + is_aromatic + degree + explicit_valence + implicit_valence
    return features

# 替換或更新現有的 bond_features 函數
def bond_features(bond):
    """
    提取化學鍵的特徵。
    
    參數:
        bond: RDKit鍵對象
        
    返回:
        list: 鍵特徵列表
    """
    # 鍵類型
    bond_type_options = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ]
    bt = onek_encoding_unk(bond.GetBondType(), bond_type_options)
    
    # 是否在環中
    bond_is_in_ring = [int(bond.IsInRing())]
    
    # 是否共軛
    is_conjugated = [int(bond.GetIsConjugated())]
    
    # 是否為立體鍵
    is_stereobond = [int(bond.GetStereo() != Chem.rdchem.BondStereo.STEREONONE)]
    
    return bt + bond_is_in_ring + is_conjugated + is_stereobond

# 替換或更新現有的 tensorize_molecule 函數
def tensorize_molecule(smiles, dissociable_atom_indices=None, pka_values=None):
    """
    將分子轉換為圖形神經網絡可用的張量表示。
    
    參數:
        smiles (str): 分子的SMILES表示
        dissociable_atom_indices (list): 可解離原子的索引列表
        pka_values (list): 原子索引和pKa值的元組列表[(atom_idx, pka_value)]
    
    返回:
        tuple: (節點特徵, 邊索引, 邊特徵, 可解離原子掩碼, pKa值張量, mol對象)
    """
    try:
        # 嘗試修復SMILES語法問題
        smiles = sanitize_smiles(smiles)
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None, None, None, None, None
        
        # 添加氫原子
        mol = Chem.AddHs(mol)
        
        # 獲取節點特徵
        atoms = mol.GetAtoms()
        n_atoms = len(atoms)
        
        x = []
        for atom in atoms:
            x.append(atom_features(atom))
        x = torch.tensor(x, dtype=torch.float)
        
        # 建立邊表示
        edges = []
        edge_feats = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            edges.append([i, j])
            edges.append([j, i])  # 圖是無向的，所以添加反向邊
            
            # 兩個方向的邊特徵相同
            edge_feat = bond_features(bond)
            edge_feats.append(edge_feat)
            edge_feats.append(edge_feat)
        
        # 如果沒有鍵，創建空的邊表示
        if len(edges) == 0:
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            edge_attr = torch.tensor([[0, 0, 0, 0, 0, 0, 0]], dtype=torch.float)
        else:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_feats, dtype=torch.float)
        
        # 創建可解離原子掩碼
        dissociable_masks = torch.zeros(n_atoms, dtype=torch.float)
        if dissociable_atom_indices:
            for idx in dissociable_atom_indices:
                if 0 <= idx < n_atoms:  # 確保索引有效
                    dissociable_masks[idx] = 1.0
        
        # 創建pKa值張量
        pka_tensor = torch.zeros(n_atoms, dtype=torch.float)
        if pka_values:
            for idx, pka in pka_values:
                if 0 <= idx < n_atoms:  # 確保索引有效
                    pka_tensor[idx] = pka
        
        return x, edge_index, edge_attr, dissociable_masks, pka_tensor, mol
    
    except Exception as e:
        print(f"張量化分子時出錯: {smiles}, 錯誤: {e}")
        return None, None, None, None, None, None

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

def get_dissociable_atoms(mol):
    """識別分子中可解離的原子"""
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
    except Exception as e:
        print(f"氨基酸專用處理時出錯: {e}")
    
    # ========== 2. 常見氨基酸手動映射 ==========
    try:
        smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
        if smiles in SPECIAL_STRUCTURES_MAP:
            return SPECIAL_STRUCTURES_MAP[smiles]
        
        # 檢查是否是立體異構體
        iso_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        for pattern, mapping in SPECIAL_STRUCTURES_MAP.items():
            if iso_smiles.startswith(pattern.split('(')[0]) and any(term in iso_smiles for term in ['C(O)=O', 'C(=O)O']):
                return mapping
    except Exception as e:
        print(f"檢查特殊結構時出錯: {e}")
    
    # ========== 3. 使用SMARTS模式匹配常見官能團 ==========
    pattern_dict = {
        # 羧酸模式
        'carboxylic_acid': [
            '[CX3](=O)[OX2H1]',     # 標準格式
            'C(=O)[OH]',             # 更簡單格式
            'C([OH])=O',             # 另一種常見表示法
            'C(O)=O',                # 甘氨酸中的格式
            'C(=O)O[H]',             # 顯式氫寫法
            '[#6]-[#6](=[#8])-[#8;H1]' # 非常通用的模式
        ],
        
        # 醇/酚類
        'alcohol': [
            '[OX2H]',                # 標準格式
            '[OH]',                  # 簡化格式
            'O[H]',                  # 顯式氫寫法
            '[#8;H1]-[#6]'           # 通用模式
        ],
        'phenol': [
            '[OX2H][cX3]',           # 標準格式
            '[OH]c',                 # 簡化格式
            '[#8;H1]-[#6;a]'         # 通用模式
        ],
        
        # 胺類
        'amine_primary': [
            '[NX3;H2]',              # 標準格式
            '[NH2]',                 # 簡化格式
            'N([H])[H]',             # 顯式氫寫法
            '[#7;H2]'                # 通用模式
        ],
        'amine_secondary': [
            '[NX3;H1]',              # 標準格式
            '[NH1]',                 # 簡化格式
            '[#7;H1]'                # 通用模式
        ],
        
        # 硫醇類
        'thiol': [
            '[SX2H]',                # 標準格式
            '[SH]',                  # 簡化格式
            'S[H]',                  # 顯式氫寫法
            '[#16;H1]'               # 通用模式
        ],
        
        # 磷酸類
        'phosphate': [
            '[P](=O)([O])[O]',       # 標準格式
            'P(=O)([OH])([OH])[OH]', # 磷酸
            'P(=O)([OH])([OH])[O-]', # 部分解離的磷酸
            '[#15](=[#8])(-[#8])',   # 通用模式
        ],
    }
    
    # 尋找所有可能的官能團
    for group, patterns in pattern_dict.items():
        for pattern in patterns:
            try:
                patt = Chem.MolFromSmarts(pattern)
                if patt and mol.HasSubstructMatch(patt):
                    matches = mol.GetSubstructMatches(patt)
                    for match in matches:
                        for idx in match:
                            atom = mol.GetAtomWithIdx(idx)
                            # 檢查原子類型和氫原子數
                            if ((group.startswith('carboxylic') and atom.GetSymbol() == 'O' and atom.GetTotalNumHs() > 0) or
                                (group == 'alcohol' and atom.GetSymbol() == 'O' and atom.GetTotalNumHs() > 0) or
                                (group == 'phenol' and atom.GetSymbol() == 'O' and atom.GetTotalNumHs() > 0) or
                                (group.startswith('amine') and atom.GetSymbol() == 'N') or
                                (group == 'thiol' and atom.GetSymbol() == 'S' and atom.GetTotalNumHs() > 0) or
                                (group == 'phosphate' and atom.GetSymbol() in ['O', 'P'])):
                                
                                # 確認胺類型
                                if group.startswith('amine'):
                                    try:
                                        n_hs = atom.GetTotalNumHs()
                                        if n_hs >= 2:
                                            group = 'amine_primary'
                                        elif n_hs == 1:
                                            group = 'amine_secondary'
                                        else:
                                            continue  # 跳過無氫胺
                                    except:
                                        group = 'amine_primary'  # 默認
                                
                                # 檢查是否已經添加
                                if not any(idx == x[0] for x in dissociable_atoms):
                                    dissociable_atoms.append((idx, group))
            except Exception as e:
                print(f"處理SMARTS模式 {pattern} 時出錯: {e}")
                continue
    
    # 如果已經找到解離位點，直接返回
    if dissociable_atoms:
        return dissociable_atoms
    
    # ========== 4. 如果還沒找到，使用環和原子特性分析 ==========
    try:
        # 先嘗試特殊環分析
        ring_atoms = analyze_heterocyclic_rings(mol)
        if ring_atoms:
            return ring_atoms
            
        # 嘗試芳香環分析
        aromatic_atoms = get_dissociable_atoms_for_aromatics(mol, Chem.MolToSmiles(mol))
        if aromatic_atoms:
            return aromatic_atoms
            
        # 全面掃描所有可能的解離原子
        all_atoms = comprehensive_atom_scan(mol)
        if all_atoms:
            return all_atoms
    except Exception as e:
        print(f"進階解析時出錯: {e}")
    
    # 如果所有方法都失敗，返回最基本的原子分析
    return basic_atom_analysis(mol)

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
            
            # print(f"處理環中氮原子時出錯: {e}")
            pass
    
    return dissociable_atoms

def get_dissociable_atoms_for_aromatics(mol, smiles):
    """大幅增強的芳香環和雜環結構解析函數"""
    dissociable_atoms = []
    
    # 1. 使用更寬鬆的芳香環官能團模式
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
        except Exception as e:
            # print(f"分析環系統時出錯: {e}")
            pass
    
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
        'C[C@H](N)C(O)=O': [(2, 'amine_primary'), (3, 'carboxylic_acid')],  # 丙氨酸
        'NC(C)C(O)=O': [(0, 'amine_primary'), (3, 'carboxylic_acid')],  # 丙氨酸變體
        
        # 環狀結構
        'c1cc[nH]c1': [(3, 'amine_secondary')],  # 吡咯
        'c1cc[nH]cc1': [(3, 'amine_secondary')],  # 咪唑
        'c1c[nH]c2ccccc12': [(2, 'amine_secondary')],  # 吲哚
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

def predict_dissociable_atoms(smiles):
    """
    從SMILES字符串預測可解離原子。
    """
    try:
        # 嘗試修復SMILES語法問題
        smiles = sanitize_smiles(smiles)
        
        # 首先檢查特殊映射
        if smiles in SPECIAL_STRUCTURES_MAP:
            return SPECIAL_STRUCTURES_MAP[smiles]
        
        # 創建分子對象
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            log(f"無法從SMILES創建分子: {smiles}", 'warning')
            # 嘗試直接分析SMILES字符串
            return analyze_smiles_directly(smiles)
        
        # ... 其他處理邏輯 ...
        
        # 如果沒有找到解離原子，嘗試進行更深入的分析
        if not dissociable_atoms:
            log(f"使用綜合原子掃描分析: {smiles}", 'info')
            dissociable_atoms = comprehensive_atom_scan(mol)
            
        return dissociable_atoms
    
    except Exception as e:
        log(f"預測解離原子時出錯: {smiles}, 錯誤: {str(e)}", 'error')
        return []

def determine_functional_group(atom, mol):
    """根據原子及其環境確定官能團類型"""
    symbol = atom.GetSymbol()
    idx = atom.GetIdx()
    
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
        
        # 否則是醇
        return 'alcohol'
    
    elif symbol == 'N':
        # 根據氫原子數確定胺類型
        try:
            total_hs = atom.GetTotalNumHs()
            if total_hs >= 2:
                return 'amine_primary'
            elif total_hs == 1:
                # 如果是環狀或有芳香性質，可能是雜環中的N-H
                if atom.IsInRing() or atom.GetIsAromatic():
                    return 'amine_secondary'
                else:
                    return 'amine_secondary'
            else:
                return 'amine_tertiary'
        except:
            # 如果無法確定氫數，根據鍵接情況判斷
            if atom.GetDegree() == 1:
                return 'amine_primary'
            elif atom.GetDegree() == 2:
                return 'amine_secondary'
            else:
                return 'amine_tertiary'
    
    elif symbol == 'S':
        return 'thiol'
    
    elif symbol == 'P':
        return 'phosphate'
    
    # 默認情況
    return 'unknown'

def convert_nist_data(input_file, output_file):
    """全面增強的NIST數據轉換函數，專門針對雜環結構優化"""
    print(f"正在轉換 {input_file} 至 {output_file}...")
    
    try:
        # 讀取NIST數據
        df = pd.read_csv(input_file)
        print(f"載入了 {len(df)} 條數據記錄")
    except Exception as e:
        print(f"讀取輸入文件時出錯: {e}")
        return None
    
    # 創建結果數據框
    result_df = pd.DataFrame(columns=['SMILES', 'pKa', 'Dissociable_Atoms', 'Functional_Group', 'Name'])
    
    # 記錄處理統計
    processed = 0
    success = 0
    failed = 0
    error_log = []
    
    # 遍歷每一行
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="處理分子"):
        try:
            smiles = row['SMILES']
            pka = row['Value'] if 'Value' in row else row['pKa'] if 'pKa' in row else None
            name = row['Ligand'] if 'Ligand' in row else row['Name'] if 'Name' in row else f"Compound-{idx}"
            eq = row['Equilibrium']
            ionic = row['Ionic strength']
            temp = row['Temperature (C)']
            if pd.isna(smiles) or not isinstance(smiles, str) or len(smiles) == 0:
                error_log.append(f"跳過索引 {idx}: 無效的SMILES")
                continue
                
            if pd.isna(pka) or pka is None:
                error_log.append(f"跳過索引 {idx}: 無效的pKa值")
                continue
            
            # 查看是否有max_eq_num欄位
            expected_num = None
            if 'max_eq_num' in row and pd.notna(row['max_eq_num']):
                try:
                    expected_num = int(row['max_eq_num'])
                except:
                    pass
            
            # 檢查是否是特殊情況
            special_case = False
            dissociable_atoms = []
            
            # 特殊SMILES直接映射優先（確保特定的分子能被正確處理）
            if smiles in SPECIAL_STRUCTURES_MAP:
                dissociable_atoms = SPECIAL_STRUCTURES_MAP[smiles]
                special_case = True
            
            # 檢查是否是已知類別的化合物
            elif 'Ligand Class' in row and pd.notna(row['Ligand Class']):
                ligand_class = row['Ligand Class']
                
                # 根據類別選擇合適的處理函數
                if ligand_class == 'Pyridines (azines)':
                    special_case = True
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        # 嘗試使用專門的吡啶處理函數
                        dissociable_atoms = analyze_substituted_pyridines(mol, smiles)
                        if not dissociable_atoms:  # 如果第一個函數沒成功，嘗試另一個
                            dissociable_atoms = analyze_pyridine_structures(mol, smiles)
                
                elif ligand_class == 'Quinolines':
                    special_case = True
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        if '[Na+]' in smiles and '[S]' in smiles:
                            # 複雜的喹啉衍生物
                            dissociable_atoms = analyze_complex_quinolines(mol, smiles)
                        else:
                            # 普通喹啉
                            dissociable_atoms = analyze_quinoline_structures(mol, smiles)
                
                elif ligand_class == 'Bipyridines':
                    special_case = True
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        dissociable_atoms = analyze_bipyridine_structures(mol, smiles)
                
                elif ligand_class == 'Phenanthrolines':
                    special_case = True
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        dissociable_atoms = analyze_phenanthroline_structures(mol, smiles)
                
                elif ligand_class == 'Purines':
                    special_case = True
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        dissociable_atoms = analyze_purine_structures(mol, smiles)
                
                elif ligand_class in ['cyclic ketones', 'Ketones (oxo ligands)']:
                    special_case = True
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        dissociable_atoms = analyze_cyclic_ketones(mol, smiles)
            
            # 如果不是特殊情況或特殊處理未找到解離位點，繼續使用通常的處理方法
            if not special_case or not dissociable_atoms:
                # 使用增強版預測函數
                dissociable_atoms = predict_dissociable_atoms(smiles)
            
            processed += 1
            
            # 檢查與預期解離位點數
            if expected_num is not None and len(dissociable_atoms) != expected_num:
                print(f"警告: {smiles} 預期 {expected_num} 個解離位點，但找到 {len(dissociable_atoms)} 個")
                
                # 多層級嘗試找出預期數量的解離位點
                for attempt_func in [
                    lambda s: analyze_heterocyclic_rings(Chem.MolFromSmiles(s)),
                    lambda s: get_dissociable_atoms_for_aromatics(Chem.MolFromSmiles(s), s),
                    lambda s: comprehensive_atom_scan(Chem.MolFromSmiles(s)),
                    lambda s: analyze_smiles_directly(s)
                ]:
                    try:
                        mol = Chem.MolFromSmiles(smiles)
                        if mol is None:
                            continue
                            
                        alt_atoms = attempt_func(smiles if 's' in attempt_func.__code__.co_varnames else mol)
                        
                        if len(alt_atoms) == expected_num:
                            dissociable_atoms = alt_atoms
                            break
                        elif len(alt_atoms) > expected_num:
                            # 根據優先級選擇
                            priority_order = {
                                'carboxylic_acid': 0,
                                'amine_primary': 1,
                                'amine_secondary': 2,
                                'phenol': 3,
                                'pyridine_nitrogen': 4,
                                'imidazole_nitrogen': 5,
                                'tertiary_amine': 6,
                                'alcohol': 7,
                                'thiol': 8,
                                'phosphate': 9,
                                'sulfonic_acid': 10,
                                'nitrile': 11,
                                'hydroxyl_group': 12,
                                'carbonyl': 13,
                                'n_oxide': 14,
                                'selenium_hydride': 15,
                                'arsenic_acid': 16,
                                'unknown': 999
                            }
                            
                            alt_atoms.sort(key=lambda x: priority_order.get(x[1], 999))
                            dissociable_atoms = alt_atoms[:expected_num]
                            break
                    except Exception as e:
                        print(f"替代方法失敗: {e}")
                        continue
            
            # 如果找到解離位點
            if dissociable_atoms:
                # 提取索引和官能團類型
                indices = [str(idx) for idx, _ in dissociable_atoms]
                groups = [group for _, group in dissociable_atoms]
                
                dissociable_atoms_str = ':'.join(indices)
                functional_group_str = ':'.join(groups)
                
                # 添加到結果數據框
                result_df = pd.concat([result_df, pd.DataFrame({
                    'SMILES': [smiles],
                    'pKa': [pka],
                    'Dissociable_Atoms': [dissociable_atoms_str],
                    'Functional_Group': [functional_group_str],
                    'Name': [name],
                    'Equilibrium': [eq],
                    'Ionic': [ionic],
                    'Temperature': [temp]
                })], ignore_index=True)
                success += 1
            else:
                print(f"警告: 無法在分子中找到可解離原子: {smiles}, {name}")
                error_log.append(f"無法識別解離位點: {smiles}")
                
                # 最後嘗試: 使用結構分析和字符串搜索
                final_attempt = False
                
                # 檢查是否為吡啶或雜環結構
                if 'n' in smiles and ('1' in smiles or '2' in smiles or 'c' in smiles):
                    # 這可能是吡啶或雜環化合物，使用直接搜索
                    try:
                        # 尋找環氮
                        n_positions = []
                        for i, char in enumerate(smiles):
                            if char == 'n' and i+1 < len(smiles) and (smiles[i+1].isdigit() or smiles[i+1] == 'c'):
                                n_positions.append(i)
                        
                        if n_positions:
                            indices = [str(pos) for pos in n_positions]
                            groups = ['pyridine_nitrogen'] * len(n_positions)
                            
                            result_df = pd.concat([result_df, pd.DataFrame({
                                'SMILES': [smiles],
                                'pKa': [pka],
                                'Dissociable_Atoms': [':'.join(indices)],
                                'Functional_Group': [':'.join(groups)],
                                'Name': [name + ' (環氮估計)'],
                                'Equilibrium': [eq],
                                'Ionic': [ionic],
                                'Temperature': [temp]
                            })], ignore_index=True)
                            success += 1
                            final_attempt = True
                    except Exception as e:
                        print(f"環氮搜索失敗: {e}")
                
                # 檢查是否含有環狀結構
                if not final_attempt and ('1' in smiles or '2' in smiles or '3' in smiles) and any(x in smiles for x in ['c', 'n', 'o', 's']):
                    # 這是芳香或雜環化合物，使用SMILES直接分析
                    try:
                        fallback_result = analyze_smiles_directly(smiles)
                        if fallback_result:
                            indices = [str(idx) for idx, _ in fallback_result]
                            groups = [group for _, group in fallback_result]
                            
                            result_df = pd.concat([result_df, pd.DataFrame({
                                'SMILES': [smiles],
                                'pKa': [pka],
                                'Dissociable_Atoms': [':'.join(indices)],
                                'Functional_Group': [':'.join(groups)],
                                'Name': [name + ' (環狀結構估計)'],
                                'Equilibrium': [eq],
                                'Ionic': [ionic],
                                'Temperature': [temp]
                            })], ignore_index=True)
                            success += 1
                            final_attempt = True
                    except Exception as e:
                        print(f"環狀結構分析失敗: {e}")
                
                # 對於氨基酸結構，直接估算解離位點
                if not final_attempt and 'N' in smiles and ('C(O)=O' in smiles or 'C(=O)O' in smiles):
                    try:
                        # 這是氨基酸結構的基本猜測
                        n_idx = smiles.find('N')
                        cooh_idx = smiles.find('C(O)=O')
                        if cooh_idx == -1:
                            cooh_idx = smiles.find('C(=O)O')
                            if cooh_idx != -1:
                                cooh_idx += 4  # O的位置在C(=O)O中的索引4
                        else:
                            cooh_idx += 2  # O的位置在C(O)=O中的索引2
                        
                        indices = []
                        groups = []
                        
                        if n_idx != -1:
                            indices.append(str(n_idx))
                            groups.append('amine_primary')
                        
                        if cooh_idx != -1:
                            indices.append(str(cooh_idx))
                            groups.append('carboxylic_acid')
                        
                        if indices and groups:
                            result_df = pd.concat([result_df, pd.DataFrame({
                                'SMILES': [smiles],
                                'pKa': [pka],
                                'Dissociable_Atoms': [':'.join(indices)],
                                'Functional_Group': [':'.join(groups)],
                                'Name': [name + ' (估計解離位點)'],
                                'Equilibrium': [eq],
                                'Ionic': [ionic],
                                'Temperature': [temp]
                            })], ignore_index=True)
                            success += 1
                            final_attempt = True
                    except Exception as e:
                        print(f"氨基酸結構估計失敗: {e}")
                
                # 最後嘗試：使用原子符號和數字估算位置
                if not final_attempt:
                    # 檢查SMILES中包含的可解離原子
                    possible_atoms = []
                    for i, char in enumerate(smiles):
                        if char in ['O', 'N', 'S']:
                            # 判斷是否可能是可解離位點(例如是否後面跟著H)
                            if i+1 < len(smiles) and smiles[i+1] == 'H':
                                group = 'alcohol' if char == 'O' else 'amine_primary' if char == 'N' else 'thiol'
                                possible_atoms.append((i, group))
                    
                    # 如果找到可能的解離位點
                    if possible_atoms:
                        indices = [str(idx) for idx, _ in possible_atoms]
                        groups = [group for _, group in possible_atoms]
                        
                        result_df = pd.concat([result_df, pd.DataFrame({
                            'SMILES': [smiles],
                            'pKa': [pka],
                            'Dissociable_Atoms': [':'.join(indices)],
                            'Functional_Group': [':'.join(groups)],
                            'Name': [name + ' (字符估計)'],
                            'Equilibrium': [eq],
                            'Ionic': [ionic],
                            'Temperature': [temp]
                        })], ignore_index=True)
                        success += 1
                        final_attempt = True
                    
                # 完全失敗的情況
                if not final_attempt:
                    result_df = pd.concat([result_df, pd.DataFrame({
                        'SMILES': [smiles],
                        'pKa': [pka],
                        'Dissociable_Atoms': ['0'],  # 默認第一個原子
                        'Functional_Group': ['unknown'],
                        'Name': [name + ' (無法識別解離位點)'],
                        'Equilibrium': [eq],
                        'Ionic': [ionic],
                        'Temperature': [temp]
                    })], ignore_index=True)
                    failed += 1
                    
        except Exception as e:
            error_message = f"處理分子時出錯: {smiles if 'smiles' in locals() else 'unknown'}, {name if 'name' in locals() else 'unknown'}, 錯誤: {e}"
            print(error_message)
            error_log.append(error_message)
            failed += 1
    
    # 保存結果
    try:
        result_df.to_csv(output_file, index=False)
        print(f"轉換完成! 已保存到 {output_file}")
    except Exception as e:
        print(f"保存結果時出錯: {e}")
    
    # 保存錯誤日誌
    log_file = os.path.splitext(output_file)[0] + "_errors.log"
    with open(log_file, "w") as f:
        f.write("\n".join(error_log))
    
    print(f"統計: 處理 {processed} 條記錄, 成功 {success} 條, 失敗 {failed} 條")
    print(f"錯誤日誌已保存到 {log_file}")
    
    return result_df

# 添加對吡啶的專門處理函數
def analyze_pyridine_structures(mol, smiles):
    """專門針對吡啶結構的分析函數"""
    dissociable_atoms = []
    
    # 吡啶相關SMARTS模式
    pyridine_patterns = [
        'c1ccccn1',         # 基本吡啶
        'c1cccnc1',         # 吡啶異構體
        'c1ccncc1',         # 吡嗪
        'c1cnccn1',         # 嘧啶
        'c1cncnc1',         # 1,3,5-三嗪
        'c1ncncn1',         # 1,2,4,5-四嗪
        'c1ccc2ncccc2c1',   # 喹啉
        'c1ccnc2ccccc12',   # 異喹啉
        'c1cnc2ccccc2c1',   # 異喹啉異構體
        'c1nccc2ccccc12',   # 喹啉異構體
    ]
    
    # 檢查是否匹配基本吡啶模式
    for pattern in pyridine_patterns:
        if pattern in smiles:
            # 嘗試找到吡啶環中的氮原子
            patt = Chem.MolFromSmarts('n:1ccccc:1')  # 標準吡啶環中的氮
            if patt and mol.HasSubstructMatch(patt):
                matches = mol.GetSubstructMatches(patt)
                for match in matches:
                    # 吡啶氮原子是第一個原子
                    n_idx = match[0]
                    if not any(n_idx == x[0] for x in dissociable_atoms):
                        dissociable_atoms.append((n_idx, 'pyridine_nitrogen'))
            
            # 針對吡嗪等雙氮環
            patt = Chem.MolFromSmarts('n:1cccn:c:1')  # 吡嗪環
            if patt and mol.HasSubstructMatch(patt):
                matches = mol.GetSubstructMatches(patt)
                for match in matches:
                    # 兩個氮原子都是可解離位點
                    for idx in [0, 4]:  # 吡嗪的兩個氮位置
                        n_idx = match[idx]
                        if not any(n_idx == x[0] for x in dissociable_atoms):
                            dissociable_atoms.append((n_idx, 'pyridine_nitrogen'))
            
            # 如果仍未找到，嘗試更通用的方式
            if not dissociable_atoms:
                # 找出所有芳香氮原子
                for atom in mol.GetAtoms():
                    if atom.GetSymbol() == 'N' and atom.GetIsAromatic() and not atom.GetTotalNumHs() > 0:
                        # 確保是在六元環中
                        if any(atom.IsInRingSize(6) for ring in range(1, 7)):
                            if not any(atom.GetIdx() == x[0] for x in dissociable_atoms):
                                dissociable_atoms.append((atom.GetIdx(), 'pyridine_nitrogen'))
    
    # 使用SMARTS直接搜索吡啶氮
    pyridine_n_patterns = [
        'n1ccccc1',        # 標準吡啶氮
        'n1ccncc1',        # 吡嗪中的氮
        'n1cncnc1',        # 三嗪中的氮
        'n1c2ccccc2ccc1',  # 喹啉中的氮
        'n1c2ccccc2cc1',   # 異喹啉中的氮
    ]
    
    for pattern in pyridine_n_patterns:
        try:
            patt = Chem.MolFromSmarts(pattern)
            if patt and mol.HasSubstructMatch(patt):
                matches = mol.GetSubstructMatches(patt)
                for match in matches:
                    n_idx = match[0]  # 吡啶氮通常是模式中的第一個原子
                    if not any(n_idx == x[0] for x in dissociable_atoms):
                        dissociable_atoms.append((n_idx, 'pyridine_nitrogen'))
        except Exception as e:
            print(f"吡啶SMARTS處理出錯 {pattern}: {e}")
    
    return dissociable_atoms

# 直接分析SMILES字符串查找吡啶氮的函數
def find_pyridine_n_in_smiles(smiles):
    """直接在SMILES字符串中查找吡啶氮的位置"""
    dissociable_atoms = []
    
    # 尋找吡啶環模式
    pyridine_patterns = {
        'c1ccccn1': 5,      # 標準吡啶，氮在位置5
        'c1cccnc1': 4,      # 3-吡啶，氮在位置4
        'c1ccncc1': 3,      # 吡嗪，氮在位置3
        'n1ccccc1': 0,      # 吡啶異構寫法，氮在位置0
        'n1ccncc1': 0,      # 吡嗪異構寫法，氮在位置0
        'c1cnccn1': 2,      # 嘧啶，氮在位置2和5
        'c1ccc2ncccc2c1': 5, # 喹啉，氮在位置5
    }
    
    for pattern, n_pos in pyridine_patterns.items():
        start = 0
        while True:
            pos = smiles.find(pattern, start)
            if pos == -1:
                break
            
            # 將索引添加到解離位點
            idx = pos + n_pos
            dissociable_atoms.append((idx, 'pyridine_nitrogen'))
            start = pos + 1
    
    # 特殊處理多環系統中的吡啶氮
    if '1' in smiles and 'n' in smiles and 'c' in smiles:
        # 找出所有小寫n（表示芳香氮）
        for i, char in enumerate(smiles):
            if char == 'n' and (i == 0 or smiles[i-1] not in ['C', 'c', 'N', '[', '(']):
                # 檢查後續字符確認是環系統的一部分
                if i+1 < len(smiles) and smiles[i+1] in ['1', '2', '3', '4', '5', 'c']:
                    dissociable_atoms.append((i, 'pyridine_nitrogen'))
    
    return dissociable_atoms

# 定義新的咪唑及其衍生物解析函數
def analyze_imidazole_structures(mol, smiles):
    """專門針對咪唑及其衍生物的解析函數"""
    dissociable_atoms = []
    
    # 咪唑相關SMARTS模式
    imidazole_patterns = [
        'n1cncc1',          # 咪唑環
        'n1ccnc1',          # 咪唑環異構體
        'n1cncn1',          # 三唑環
        'n1ncnn1',          # 四唑環
        'n1cccn1',          # 吡唑環
        'n1c2ccccc2nc1',    # 喹喔啉環
        'n1c2ccccc2cn1',    # 苯并咪唑環
    ]
    
    # 檢查基本咪唑模式
    for pattern in imidazole_patterns:
        if pattern in smiles:
            try:
                patt = Chem.MolFromSmarts(pattern)
                if patt and mol.HasSubstructMatch(patt):
                    matches = mol.GetSubstructMatches(patt)
                    for match in matches:
                        # 咪唑環中通常第一個氮是解離位點
                        n_idx = match[0]
                        if not any(n_idx == x[0] for x in dissociable_atoms):
                            dissociable_atoms.append((n_idx, 'imidazole_nitrogen'))
                        
                        # 檢查環中是否還有其他氮原子
                        for i, atom_idx in enumerate(match):
                            atom = mol.GetAtomWithIdx(atom_idx)
                            if i > 0 and atom.GetSymbol() == 'N' and not any(atom_idx == x[0] for x in dissociable_atoms):
                                dissociable_atoms.append((atom_idx, 'imidazole_nitrogen'))
            except Exception as e:
                print(f"咪唑分析出錯 {pattern}: {e}")
    
    # 直接使用原子特性分析咪唑環
    if not dissociable_atoms:
        try:
            # 先找出所有環系統
            rings = mol.GetSSSR()
            for ring in rings:
                # 檢查是否是5員環且含有至少兩個氮
                if len(ring) == 5:
                    ring_atoms = [mol.GetAtomWithIdx(i) for i in ring]
                    n_atoms = [a for a in ring_atoms if a.GetSymbol() == 'N']
                    
                    if len(n_atoms) >= 1:
                        # 這可能是咪唑或類似的雜環
                        for n_atom in n_atoms:
                            if not any(n_atom.GetIdx() == x[0] for x in dissociable_atoms):
                                if n_atom.GetIsAromatic():
                                    # 判斷是否是咪唑型氮(在芳香環上並且沒有氫)
                                    dissociable_atoms.append((n_atom.GetIdx(), 'imidazole_nitrogen'))
        except Exception as e:
            print(f"咪唑環系統分析出錯: {e}")
    
    return dissociable_atoms

# 定義處理複雜含氮雜環的函數
def analyze_tertiary_amine_systems(mol, smiles):
    """專門針對三級胺及複雜雜環的解析函數"""
    dissociable_atoms = []
    
    # 特定三級胺模式
    tertiary_amine_patterns = [
        'CN1CCN', 'CN1CCC', 'C1NCNC', 'C1NCCN', 
        'N1CCN2', 'N1CCC2', 'N1CCCN'
    ]
    
    for pattern in tertiary_amine_patterns:
        if pattern in smiles:
            try:
                # 使用RDKit查找所有匹配
                patt = Chem.MolFromSmarts('[$([N;R]);!$([N+]);!$([N-])]')  # 環狀中性氮
                if patt and mol.HasSubstructMatch(patt):
                    matches = mol.GetSubstructMatches(patt)
                    for match in matches:
                        # 只關注環形三級胺
                        atom = mol.GetAtomWithIdx(match[0])
                        if atom.IsInRing() and atom.GetDegree() == 3:
                            dissociable_atoms.append((match[0], 'tertiary_amine'))
            except Exception as e:
                print(f"三級胺分析出錯 {pattern}: {e}")
    
    # 特殊處理形如CN1CCN(C)CC1這樣的結構
    if 'N1' in smiles and 'N(' in smiles and 'CC' in smiles:
        try:
            # 遍歷所有氮原子尋找三級胺
            for atom in mol.GetAtoms():
                if atom.GetSymbol() == 'N' and atom.GetDegree() == 3 and atom.IsInRing():
                    if not any(atom.GetIdx() == x[0] for x in dissociable_atoms):
                        dissociable_atoms.append((atom.GetIdx(), 'tertiary_amine'))
        except Exception as e:
            print(f"複雜三級胺分析出錯: {e}")
    
    return dissociable_atoms

# 定義處理含特殊原子(S, Se, As, F等)化合物的函數
def analyze_special_elements(mol, smiles):
    """專門針對含硫、硒、砷等特殊元素的解析函數"""
    dissociable_atoms = []
    
    # 分析含硫化合物
    if 'S' in smiles or 's' in smiles:
        try:
            # 硫原子相關模式
            sulfur_patterns = {
                'thiophene': '[s;R1]',  # 噻吩環中的硫
                'thiol': '[S;H1]',      # 硫醇
                'sulfonic': '[S](=O)(=O)[O;H1]', # 磺酸
                'sulfinic': '[S](=O)[O;H1]',     # 亞磺酸
                'thioether': '[S;D2](-[#6])-[#6]' # 硫醚
            }
            
            for group, pattern in sulfur_patterns.items():
                patt = Chem.MolFromSmarts(pattern)
                if patt and mol.HasSubstructMatch(patt):
                    matches = mol.GetSubstructMatches(patt)
                    for match in matches:
                        # 第一個原子是硫
                        s_idx = match[0]
                        if not any(s_idx == x[0] for x in dissociable_atoms):
                            dissociable_atoms.append((s_idx, f'sulfur_{group}'))
        except Exception as e:
            print(f"含硫化合物分析出錯: {e}")
    
    # 分析含硒化合物
    if 'Se' in smiles or '[SeH]' in smiles:
        try:
            # 直接尋找硒原子
            for atom in mol.GetAtoms():
                if atom.GetSymbol() == 'Se':
                    # 特別處理含氫的硒
                    if atom.GetTotalNumHs() > 0:
                        dissociable_atoms.append((atom.GetIdx(), 'selenium_hydride'))
                    else:
                        dissociable_atoms.append((atom.GetIdx(), 'selenium_compound'))
        except Exception as e:
            print(f"含硒化合物分析出錯: {e}")
    
    # 分析含砷化合物
    if 'As' in smiles:
        try:
            # 尋找砷原子
            for atom in mol.GetAtoms():
                if atom.GetSymbol() == 'As':
                    # 檢查是否有氧鍵接，可能是砷酸
                    has_oxygen = any(nei.GetSymbol() == 'O' for nei in atom.GetNeighbors())
                    if has_oxygen:
                        dissociable_atoms.append((atom.GetIdx(), 'arsenic_acid'))
                    else:
                        dissociable_atoms.append((atom.GetIdx(), 'arsenic_compound'))
        except Exception as e:
            print(f"含砷化合物分析出錯: {e}")
    
    # 分析含氟化合物（特別是三氟甲基）
    if 'F' in smiles and 'F(F)(F)' in smiles:
        try:
            # 識別三氟甲基旁邊的羰基
            patt = Chem.MolFromSmarts('C(F)(F)(F)C(=O)')
            if patt and mol.HasSubstructMatch(patt):
                matches = mol.GetSubstructMatches(patt)
                for match in matches:
                    # 关注C=O部分
                    c_idx = match[4]  # C=O中的C
                    o_idx = None
                    # 查找羰基氧
                    atom = mol.GetAtomWithIdx(c_idx)
                    for bond in atom.GetBonds():
                        other_idx = bond.GetOtherAtomIdx(c_idx)
                        if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE and mol.GetAtomWithIdx(other_idx).GetSymbol() == 'O':
                            o_idx = other_idx
                            break
                    
                    if o_idx is not None and not any(o_idx == x[0] for x in dissociable_atoms):
                        dissociable_atoms.append((o_idx, 'carbonyl_adjacent_to_CF3'))
        except Exception as e:
            print(f"含氟化合物分析出錯: {e}")
    
    # 分析N-氧化物
    if '[N+]([O-])' in smiles:
        try:
            # 直接尋找N+連接O-的模式
            patt = Chem.MolFromSmarts('[N+]([O-])')
            if patt and mol.HasSubstructMatch(patt):
                matches = mol.GetSubstructMatches(patt)
                for match in matches:
                    n_idx = match[0]
                    if not any(n_idx == x[0] for x in dissociable_atoms):
                        dissociable_atoms.append((n_idx, 'n_oxide'))
        except Exception as e:
            print(f"N-氧化物分析出錯: {e}")
    
    # 分析氰基
    if 'C#N' in smiles or 'N#C' in smiles:
        try:
            patt = Chem.MolFromSmarts('C#N')
            if patt and mol.HasSubstructMatch(patt):
                matches = mol.GetSubstructMatches(patt)
                for match in matches:
                    c_idx = match[0]  # 氰基碳
                    n_idx = match[1]  # 氰基氮
                    if not any(n_idx == x[0] for x in dissociable_atoms):
                        dissociable_atoms.append((n_idx, 'nitrile'))
        except Exception as e:
            print(f"氰基分析出錯: {e}")
    
    return dissociable_atoms

# 更新find_pyridine_n_in_smiles函數以包含更多雜環模式
def find_heterocyclic_n_in_smiles(smiles):
    """直接在SMILES字符串中查找雜環氮的位置（包括吡啶、咪唑等）"""
    dissociable_atoms = []
    
    # 擴展模式匹配更多雜環結構
    heterocyclic_patterns = {
        # 吡啶和衍生物模式
        'c1ccccn1': [(5, 'pyridine_nitrogen')],      # 吡啶
        'c1cccnc1': [(4, 'pyridine_nitrogen')],      # 吡啶異構體
        'c1ccncc1': [(3, 'pyridine_nitrogen')],      # 吡嗪
        
        # 咪唑和三唑模式
        'c1ncn1': [(1, 'imidazole_nitrogen'), (3, 'imidazole_nitrogen')],  # 基本咪唑
        'c1nccn1': [(1, 'imidazole_nitrogen'), (4, 'imidazole_nitrogen')], # 咪唑異構體
        'c1cncn1': [(2, 'triazole_nitrogen'), (4, 'triazole_nitrogen')],   # 1,2,4-三唑
        
        # 縮合雜環
        'c1nc2ccccc2[nH]1': [(1, 'imidazole_nitrogen'), (8, 'imidazole_nitrogen')], # 苯并咪唑
        'c1nc2cncnc2[nH]1': [(1, 'imidazole_nitrogen'), (8, 'imidazole_nitrogen')], # 嘌呤衍生物
        
        # N-取代雜環
        'Cn1ccnc1': [(1, 'imidazole_nitrogen')],     # N-甲基咪唑
        'CCn1ccnc1': [(2, 'imidazole_nitrogen')],    # N-乙基咪唑
        'CCCn1ccnc1': [(3, 'imidazole_nitrogen')],   # N-丙基咪唑
    }
    
    # 先嘗試直接匹配
    for pattern, positions in heterocyclic_patterns.items():
        start = 0
        while True:
            pos = smiles.find(pattern, start)
            if pos == -1:
                break
            
            # 添加所有指定位置的氮原子
            for idx_offset, group in positions:
                idx = pos + idx_offset
                dissociable_atoms.append((idx, group))
            
            start = pos + 1
    
    # 如果沒有找到匹配，尋找所有可能的環氮
    if not dissociable_atoms and ('1' in smiles and 'n' in smiles):
        # 搜索所有環形氮的位置
        for i, char in enumerate(smiles):
            if char == 'n' and i+1 < len(smiles) and smiles[i+1].isdigit():
                # 這是環形氮
                if i > 0 and smiles[i-1] in ['C', 'c', 'N', 'S', 's']:
                    # 可能是N-取代或雜環中的氮
                    dissociable_atoms.append((i, 'heterocyclic_nitrogen'))
                else:
                    # 不確定的環氮
                    dissociable_atoms.append((i, 'ring_nitrogen'))
    
    # 處理特殊的N-氧化物和硝基
    if '[N+]([O-])' in smiles or 'N(=O)=O' in smiles:
        for i in range(len(smiles)-5):
            if smiles[i:i+8] == '[N+]([O-])' or smiles[i:i+7] == 'N(=O)=O':
                dissociable_atoms.append((i, 'n_oxide'))
    
    return dissociable_atoms

# 更新預測函數，整合上述新功能
def predict_dissociable_atoms(smiles):
    """預測分子中可能的解離原子，增強對咪唑、三級氨和特殊元素的支持"""
    try:
        # 首先檢查特殊映射
        if smiles in SPECIAL_STRUCTURES_MAP:
            return SPECIAL_STRUCTURES_MAP[smiles]
        
        # 特殊情況：氰基化合物
        if smiles == 'C#N':
            return [(0, 'nitrile')]
        
        # 嘗試標準SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # 嘗試修復SMILES語法問題
            smiles_sanitized = sanitize_smiles(smiles)
            mol = Chem.MolFromSmiles(smiles_sanitized)
            
            if mol is None:
                print(f"無法解析SMILES（即使嘗試修復後）: {smiles}")
                
                # 如果仍然無法解析，直接從SMILES字符串分析
                heterocyclic_atoms = find_heterocyclic_n_in_smiles(smiles)
                if heterocyclic_atoms:
                    return heterocyclic_atoms
                
                return analyze_smiles_directly(smiles)
        
        # 添加隱式氫原子
        try:
            mol = Chem.AddHs(mol)
        except Exception as e:
            print(f"添加氫原子失敗: {smiles}，{e}，嘗試不添加氫繼續處理")
        
        # 優先檢查特殊元素
        special_elements_atoms = analyze_special_elements(mol, smiles)
        if special_elements_atoms:
            return special_elements_atoms
        
        # 檢查是否包含咪唑結構
        if 'n1c' in smiles or 'N1C' in smiles or 'ccn' in smiles:
            imidazole_atoms = analyze_imidazole_structures(mol, smiles)
            if imidazole_atoms:
                return imidazole_atoms
        
        # 檢查是否是複雜三級氨
        if 'N1' in smiles and ('CCN' in smiles or 'CNC' in smiles):
            tertiary_amine_atoms = analyze_tertiary_amine_systems(mol, smiles)
            if tertiary_amine_atoms:
                return tertiary_amine_atoms
        
        # 檢查是否包含吡啶結構
        if 'n1' in smiles or 'c1ccccn1' in smiles or 'c1cccnc1' in smiles:
            pyridine_atoms = analyze_pyridine_structures(mol, smiles)
            if pyridine_atoms:
                return pyridine_atoms
        
        # 嘗試標準解離原子檢測
        atoms = get_dissociable_atoms(mol)
        
        # 如果找不到解離位點，嘗試直接從SMILES字符串分析
        if not atoms:
            print(f"標準檢測無法找到解離位點，嘗試直接SMILES分析: {smiles}")
            atoms = analyze_smiles_directly(smiles)
        
        return atoms
        
    except Exception as e:
        print(f"預測解離原子時出錯: {e}, SMILES: {smiles}")
        # 嚴重錯誤時仍嘗試直接分析SMILES
        return analyze_smiles_directly(smiles)

# 更新analyze_smiles_directly函數以支持特殊結構
def analyze_smiles_directly(smiles):
    """增強版SMILES直接分析，支持更多特殊結構"""
    dissociable_atoms = []
    
    # 1. 檢查常見結構的直接映射
    if smiles in SPECIAL_STRUCTURES_MAP:
        return SPECIAL_STRUCTURES_MAP[smiles]
    
    # 2. 檢查特殊元素和官能團
    
    # 咪唑和三唑
    if 'n1ccnc1' in smiles or 'n1cncc1' in smiles or 'n1cncn1' in smiles:
        # 找出第一個'n'的位置
        n_pos = smiles.find('n')
        if n_pos != -1:
            dissociable_atoms.append((n_pos, 'imidazole_nitrogen'))
    
    # 三級胺
    if 'N1CCN' in smiles or 'N1CCC' in smiles:
        n_pos = smiles.find('N1')
        if n_pos != -1:
            dissociable_atoms.append((n_pos, 'tertiary_amine'))
    
    # 含硫化合物
    if 's1' in smiles or 'S1' in smiles:
        s_pos = smiles.find('s1') if 's1' in smiles else smiles.find('S1')
        if s_pos != -1:
            dissociable_atoms.append((s_pos, 'thiophene_sulfur'))
    
    # 含硒化合物
    if '[SeH]' in smiles:
        se_pos = smiles.find('[SeH]')
        if se_pos != -1:
            dissociable_atoms.append((se_pos, 'selenium_hydride'))
    
    # 含砷化合物
    if '[As]' in smiles:
        as_pos = smiles.find('[As]')
        if as_pos != -1:
            dissociable_atoms.append((as_pos, 'arsenic_compound'))
    
    # N-氧化物
    if '[N+]([O-])' in smiles or 'N(=O)=O' in smiles:
        n_pos = smiles.find('[N+]') if '[N+]' in smiles else smiles.find('N(=O)')
        if n_pos != -1:
            dissociable_atoms.append((n_pos, 'n_oxide'))
    
    # 氰基
    if 'C#N' in smiles:
        n_pos = smiles.find('C#N') + 2  # N的位置
        if n_pos - 2 >= 0:
            dissociable_atoms.append((n_pos, 'nitrile'))
    
    # 3. 如果上述方法仍無結果，嘗試雜環氮查找
    if not dissociable_atoms and ('n' in smiles or 'N' in smiles) and ('1' in smiles or '2' in smiles):
        hetero_atoms = find_heterocyclic_n_in_smiles(smiles)
        if hetero_atoms:
            return hetero_atoms
    
    # 4. 如果仍無結果，搜索所有可能的功能性原子
    if not dissociable_atoms:
        # 搜索所有氮原子並為三級胺設置默認位置
        if 'N(' in smiles:
            for i in range(len(smiles)-2):
                if smiles[i:i+2] == 'N(' and i > 0 and smiles[i-1] != '[':
                    dissociable_atoms.append((i, 'tertiary_amine'))
        
        # 搜索所有可能的酸性氫(OH, NH, SH)
        for i, char in enumerate(smiles):
            if char in ['O', 'N', 'S'] and i+1 < len(smiles) and smiles[i+1] == 'H':
                group = 'alcohol' if char == 'O' else 'amine_primary' if char == 'N' else 'thiol'
                dissociable_atoms.append((i, group))
    
    return dissociable_atoms

# 添加專門處理菲咯啉類化合物的函數
def analyze_phenanthroline_structures(mol, smiles):
    """專門識別菲咯啉類化合物中的氮原子位置"""
    dissociable_atoms = []
    
    # 菲咯啉的典型SMARTS模式
    phenanthroline_patterns = [
        # 基本菲咯啉模式
        'c1cnc2c(c1)ccc3ncccc23',          # 1,10-菲咯啉
        'c1cnc2ccc3ncccc3c2c1',            # 4,7-菲咯啉
        # 取代的菲咯啉
        'c1cnc2c(c1)c([*])c([*])c3ncccc23',  # 5,6-二取代-1,10-菲咯啉
        'c1c([*])nc2c(c1)ccc3ncccc23',      # 取代的1,10-菲咯啉
    ]
    
    for pattern in phenanthroline_patterns:
        try:
            # 將*替換為通配符以匹配任何取代基
            pattern_for_smarts = pattern.replace('[*]', '*')
            patt = Chem.MolFromSmarts(pattern_for_smarts)
            if patt and mol.HasSubstructMatch(patt):
                matches = mol.GetSubstructMatches(patt)
                for match in matches:
                    # 尋找兩個環上的氮原子
                    for idx in match:
                        atom = mol.GetAtomWithIdx(idx)
                        if atom.GetSymbol() == 'N' and atom.GetIsAromatic():
                            if not any(idx == x[0] for x in dissociable_atoms):
                                dissociable_atoms.append((idx, 'pyridine_nitrogen'))
        except Exception as e:
            print(f"菲咯啉分析出錯 {pattern}: {e}")
    
    return dissociable_atoms

# 添加專門處理聯吡啶類化合物的函數
def analyze_bipyridine_structures(mol, smiles):
    """專門識別聯吡啶類化合物中的氮原子位置"""
    dissociable_atoms = []
    
    # 聯吡啶的典型SMARTS模式
    bipyridine_patterns = [
        'c1ccc(nc1)c2cccnc2',   # 2,3'-聯吡啶
        'c1ccc(nc1)c2ccncc2',   # 2,4'-聯吡啶
        'c1cncc(c1)c2cccnc2',   # 3,3'-聯吡啶
        'c1cncc(c1)c2ccncc2',   # 3,4'-聯吡啶
    ]
    
    for pattern in bipyridine_patterns:
        if pattern in smiles:
            try:
                patt = Chem.MolFromSmarts(pattern)
                if patt and mol.HasSubstructMatch(patt):
                    matches = mol.GetSubstructMatches(patt)
                    for match in matches:
                        # 尋找兩個吡啶環上的氮原子
                        for idx in match:
                            atom = mol.GetAtomWithIdx(idx)
                            if atom.GetSymbol() == 'N' and atom.GetIsAromatic():
                                if not any(idx == x[0] for x in dissociable_atoms):
                                    dissociable_atoms.append((idx, 'pyridine_nitrogen'))
            except Exception as e:
                print(f"聯吡啶分析出錯 {pattern}: {e}")
    
    # 直接從SMILES字符串分析
    if not dissociable_atoms:
        try:
            # 尋找環上的氮原子
            n_positions = []
            for i, char in enumerate(smiles):
                if char == 'n' and i+1 < len(smiles) and smiles[i+1].isdigit():
                    n_positions.append(i)
            
            for pos in n_positions:
                dissociable_atoms.append((pos, 'pyridine_nitrogen'))
        except Exception as e:
            print(f"聯吡啶SMILES分析出錯: {e}")
    
    return dissociable_atoms

# 添加處理喹啉及其衍生物的函數
def analyze_quinoline_structures(mol, smiles):
    """專門識別喹啉及其衍生物中的氮原子和羥基位置"""
    dissociable_atoms = []
    
    # 喹啉的典型SMARTS模式
    quinoline_patterns = [
        'c1ccc2ncccc2c1',            # 基本喹啉
        'c1cc2cccnc2cc1',            # 異構喹啉
        'Oc1cccc2cccnc12',           # 8-羥基喹啉
        'Oc1c([*])cc([*])c2cccnc12', # 取代的8-羥基喹啉
    ]
    
    for pattern in quinoline_patterns:
        try:
            # 將*替換為通配符以匹配任何取代基
            pattern_for_smarts = pattern.replace('[*]', '*')
            patt = Chem.MolFromSmarts(pattern_for_smarts)
            if patt and mol.HasSubstructMatch(patt):
                matches = mol.GetSubstructMatches(patt)
                for match in matches:
                    # 尋找氮原子和羥基氧
                    for idx in match:
                        atom = mol.GetAtomWithIdx(idx)
                        if atom.GetSymbol() == 'N' and atom.GetIsAromatic():
                            if not any(idx == x[0] for x in dissociable_atoms):
                                dissociable_atoms.append((idx, 'pyridine_nitrogen'))
                        elif atom.GetSymbol() == 'O' and atom.GetTotalNumHs() > 0:
                            # 檢查是否連接到芳香環
                            if any(n.GetIsAromatic() for n in atom.GetNeighbors()):
                                if not any(idx == x[0] for x in dissociable_atoms):
                                    dissociable_atoms.append((idx, 'phenol'))
        except Exception as e:
            print(f"喹啉分析出錯 {pattern}: {e}")
    
    # 直接從SMILES字符串分析
    if not dissociable_atoms and 'c1ccc2' in smiles and 'n' in smiles:
        try:
            # 尋找喹啉環上的氮原子和羥基
            n_pos = smiles.find('n')
            if n_pos != -1:
                dissociable_atoms.append((n_pos, 'pyridine_nitrogen'))
            
            # 尋找羥基
            o_pos = smiles.find('O')
            if o_pos != -1 and o_pos+1 < len(smiles) and smiles[o_pos+1] == 'c':
                dissociable_atoms.append((o_pos, 'phenol'))
        except Exception as e:
            print(f"喹啉SMILES分析出錯: {e}")
    
    return dissociable_atoms

# 添加處理嘌呤類化合物的函數
def analyze_purine_structures(mol, smiles):
    """專門識別嘌呤及其衍生物中的氮原子位置"""
    dissociable_atoms = []
    
    # 嘌呤的典型SMARTS模式
    purine_patterns = [
        '[nH]1cnc2ncncc12',          # 基本嘌呤
        'Cn1cnc2ncncc12',            # N9-甲基嘌呤
    ]
    
    for pattern in purine_patterns:
        try:
            patt = Chem.MolFromSmarts(pattern)
            if patt and mol.HasSubstructMatch(patt):
                matches = mol.GetSubstructMatches(patt)
                for match in matches:
                    # 尋找嘌呤環上的氮原子
                    for idx in match:
                        atom = mol.GetAtomWithIdx(idx)
                        if atom.GetSymbol() == 'N':
                            if atom.GetTotalNumHs() > 0:  # 帶氫的氮是主要解離位點
                                if not any(idx == x[0] for x in dissociable_atoms):
                                    dissociable_atoms.append((idx, 'amine_secondary'))
                            elif atom.GetIsAromatic():  # 芳香環上的氮
                                if not any(idx == x[0] for x in dissociable_atoms):
                                    dissociable_atoms.append((idx, 'heterocyclic_nitrogen'))
        except Exception as e:
            print(f"嘌呤分析出錯 {pattern}: {e}")
    
    return dissociable_atoms

# 添加專門處理環狀酮的函數
def analyze_cyclic_ketones(mol, smiles):
    """專門識別環狀酮（如tropolone）中的羥基和羰基位置"""
    dissociable_atoms = []
    
    # 環狀酮的典型SMARTS模式
    cyclic_ketone_patterns = [
        'OC1=CC=CC=CC1=O',           # Tropolone
        'CC(C)C1=CC(=O)C(=CC=C1)O',  # beta-Isopropyltropolone
        'C1CC(=O)CC(=O)C1',          # Cyclohexane-1,3-dione
    ]
    
    for pattern in cyclic_ketone_patterns:
        try:
            patt = Chem.MolFromSmarts(pattern)
            if patt and mol.HasSubstructMatch(patt):
                matches = mol.GetSubstructMatches(patt)
                for match in matches:
                    # 尋找環上的羥基和羰基
                    for idx in match:
                        atom = mol.GetAtomWithIdx(idx)
                        if atom.GetSymbol() == 'O':
                            # 檢查是否是羥基或羰基
                            if atom.GetTotalNumHs() > 0:  # 羥基
                                if not any(idx == x[0] for x in dissociable_atoms):
                                    dissociable_atoms.append((idx, 'phenol'))
                            else:  # 可能是羰基
                                c_neighbors = [n for n in atom.GetNeighbors() if n.GetSymbol() == 'C']
                                if c_neighbors and any(bond.GetBondType() == Chem.rdchem.BondType.DOUBLE 
                                                     for bond in c_neighbors[0].GetBonds()):
                                    if not any(idx == x[0] for x in dissociable_atoms):
                                        dissociable_atoms.append((idx, 'carbonyl'))
        except Exception as e:
            print(f"環狀酮分析出錯 {pattern}: {e}")
    
    return dissociable_atoms

# 添加專門處理取代吡啶(azines)的函數
def analyze_substituted_pyridines(mol, smiles):
    """專門處理各種取代的吡啶類化合物"""
    dissociable_atoms = []
    
    # 各種取代吡啶的SMARTS模式
    pyridine_patterns = [
        # 甲基/多甲基取代的吡啶
        'c1cncc(C)c1',            # 3,5-二甲基吡啶
        'c1cc(C)nc(C)c1',         # 2,6-二甲基吡啶
        'c1cc(C)nc(C)c(C)1',      # 2,4,6-三甲基吡啶
        'c1c(C)cc(C)nc1',         # 2,4-二甲基吡啶
        'c1ccc(C)nc1',            # 2-甲基吡啶
        'c1cc(C)cnc1',            # 4-甲基吡啶
        'c1c(C)ccnc1',            # 3-甲基吡啶
        
        # 乙基/丙基取代的吡啶
        'c1cc(CC)cnc1',           # 4-乙基吡啶
        'c1c(CC)ccnc1',           # 3-乙基吡啶
        'c1ccc(CC)nc1',           # 2-乙基吡啶
        
        # 含官能團的吡啶
        'c1cc(OC)cnc1',           # 4-甲氧基吡啶
        'c1cc(C(=O)OCC)cnc1',     # 4-酯基吡啶
        'c1cc(C(=O)C)cnc1',       # 4-乙酰基吡啶
        
        # 含芳基的吡啶
        'c1cc(c2ccccc2)cnc1',     # 4-苯基吡啶
        'c1c(c2ccccc2)ccnc1',     # 3-苯基吡啶
        
        # 特殊環系統
        'C1=CC=NN=C1',            # 吡嗪
        'C=1N=CC=CC1',            # 異構吡嗪
    ]
    
    # 嘗試匹配模式
    for pattern in pyridine_patterns:
        try:
            patt = Chem.MolFromSmarts(pattern)
            if patt and mol.HasSubstructMatch(patt):
                matches = mol.GetSubstructMatches(patt)
                for match in matches:
                    # 找出環上的氮原子
                    for idx in match:
                        atom = mol.GetAtomWithIdx(idx)
                        if atom.GetSymbol() == 'N' and atom.GetIsAromatic():
                            if not any(idx == x[0] for x in dissociable_atoms):
                                dissociable_atoms.append((idx, 'pyridine_nitrogen'))
        except Exception as e:
            print(f"取代吡啶分析出錯 {pattern}: {e}")
    
    # 處理二氫吡啶酮類化合物 (特殊的吡啶結構)
    dihydropyridone_patterns = [
        'CN1C=CC(=O)C=C1',         # 1-甲基-4-吡啶酮
        'CCN1C=CC(=O)C=C1',        # 1-乙基-4-吡啶酮
        'CN1C=CC(=O)C(O)=C1C',     # 1-甲基-2-甲基-3-羥基-4-吡啶酮
    ]
    
    for pattern in dihydropyridone_patterns:
        try:
            patt = Chem.MolFromSmarts(pattern)
            if patt and mol.HasSubstructMatch(patt):
                matches = mol.GetSubstructMatches(patt)
                for match in matches:
                    for idx in match:
                        atom = mol.GetAtomWithIdx(idx)
                        # 檢查N原子(三級胺)
                        if atom.GetSymbol() == 'N' and atom.GetDegree() == 3:
                            if not any(idx == x[0] for x in dissociable_atoms):
                                dissociable_atoms.append((idx, 'tertiary_amine'))
                        # 檢查羥基(如果有)
                        elif atom.GetSymbol() == 'O' and atom.GetTotalNumHs() > 0:
                            # 確認不是羰基的氧
                            for neighbor in atom.GetNeighbors():
                                if neighbor.GetSymbol() == 'C':
                                    has_double_bond = False
                                    for bond in neighbor.GetBonds():
                                        if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE and \
                                           mol.GetAtomWithIdx(bond.GetOtherAtomIdx(neighbor.GetIdx())).GetSymbol() == 'O':
                                            has_double_bond = True
                                            break
                                    if not has_double_bond and not any(idx == x[0] for x in dissociable_atoms):
                                        dissociable_atoms.append((idx, 'hydroxyl_group'))
        except Exception as e:
            print(f"二氫吡啶酮分析出錯 {pattern}: {e}")
    
    return dissociable_atoms

# 添加處理吡嗪(pyridazine)類化合物的函數
def analyze_diazines(mol, smiles):
    """專門處理吡嗪(pyridazine)類化合物"""
    dissociable_atoms = []
    
    # 吡嗪類SMARTS模式
    diazine_patterns = [
        'c1ccnnc1',        # 吡嗪(pyridazine)
        'c1ncncc1',        # 嘧啶(pyrimidine)
        'c1ncncn1',        # 三嗪(triazine)
    ]
    
    for pattern in diazine_patterns:
        try:
            patt = Chem.MolFromSmarts(pattern)
            if patt and mol.HasSubstructMatch(patt):
                matches = mol.GetSubstructMatches(patt)
                for match in matches:
                    # 找出環上的所有氮原子
                    for idx in match:
                        atom = mol.GetAtomWithIdx(idx)
                        if atom.GetSymbol() == 'N' and atom.GetIsAromatic():
                            if not any(idx == x[0] for x in dissociable_atoms):
                                dissociable_atoms.append((idx, 'pyridine_nitrogen'))
        except Exception as e:
            print(f"吡嗪分析出錯 {pattern}: {e}")
    
    # 直接從SMILES字符串查找
    if not dissociable_atoms:
        # 尋找環上的n
        n_positions = []
        for i, char in enumerate(smiles):
            if char == 'n' and i+1 < len(smiles) and smiles[i+1] in '123456789c':
                n_positions.append(i)
        
        for pos in n_positions:
            if not any(pos == x[0] for x in dissociable_atoms):
                dissociable_atoms.append((pos, 'pyridine_nitrogen'))
    
    return dissociable_atoms

# 添加處理複雜喹啉衍生物的函數
def analyze_complex_quinolines(mol, smiles):
    """專門處理含多種官能團的複雜喹啉衍生物"""
    dissociable_atoms = []
    
    # 嘗試找出所有可能的解離位點
    try:
        # 1. 喹啉環中的氮
        quinoline_pattern = Chem.MolFromSmarts('c1ccc2ncccc2c1')
        if quinoline_pattern and mol.HasSubstructMatch(quinoline_pattern):
            matches = mol.GetSubstructMatches(quinoline_pattern)
            for match in matches:
                for idx in match:
                    atom = mol.GetAtomWithIdx(idx)
                    if atom.GetSymbol() == 'N' and atom.GetIsAromatic():
                        dissociable_atoms.append((idx, 'pyridine_nitrogen'))
        
        # 2. 磺酸基
        sulfonic_pattern = Chem.MolFromSmarts('[S](=O)(=O)[O-]')
        if sulfonic_pattern and mol.HasSubstructMatch(sulfonic_pattern):
            matches = mol.GetSubstructMatches(sulfonic_pattern)
            for match in matches:
                s_idx = match[0]  # 硫原子
                dissociable_atoms.append((s_idx, 'sulfonic_acid'))
        
        # 3. 羥基(酚羥基)
        phenol_pattern = Chem.MolFromSmarts('c1c2c(ccc1)c(O)c(N=N)cc2')
        if phenol_pattern and mol.HasSubstructMatch(phenol_pattern):
            matches = mol.GetSubstructMatches(phenol_pattern)
            for match in matches:
                for idx in match:
                    atom = mol.GetAtomWithIdx(idx)
                    if atom.GetSymbol() == 'O' and atom.GetTotalNumHs() > 0:
                        dissociable_atoms.append((idx, 'phenol'))
    
    except Exception as e:
        print(f"複雜喹啉分析出錯: {e}")
    
    # 如果分子非常複雜且之前未能找到解離位點，嘗試特定分子的直接處理
    if '[Na+].[Na+].[O-][S](=O)(=O)' in smiles and 'N=C' in smiles and 'cccnc' in smiles:
        # SNAZOXS: 8-Hydroxy-7-(4-sulfo-1-naphthylazo)quinoline-5-sulfonic acid
        # 手動指定解離位點
        n_pos = smiles.find('cccnc') + 4  # 喹啉中的氮
        if not any(n_pos == x[0] for x in dissociable_atoms):
            dissociable_atoms.append((n_pos, 'pyridine_nitrogen'))
        
        # 找出兩個磺酸基
        s_pos1 = smiles.find('[S]([O-])') 
        s_pos2 = smiles.find('[S]([O-])', s_pos1 + 1)
        
        if s_pos1 != -1 and not any(s_pos1 == x[0] for x in dissociable_atoms):
            dissociable_atoms.append((s_pos1, 'sulfonic_acid'))
        
        if s_pos2 != -1 and not any(s_pos2 == x[0] for x in dissociable_atoms):
            dissociable_atoms.append((s_pos2, 'sulfonic_acid'))
    
    return dissociable_atoms

# 改進analyze_pyridine_structures函數，使其更好地處理各種取代吡啶
def analyze_pyridine_structures(mol, smiles):
    """改進版的吡啶結構分析函數，支持更多取代模式"""
    dissociable_atoms = []
    
    # 嘗試基本吡啶環的模式
    pyridine_patterns = [
        'c1ccccn1',         # 基本吡啶
        'c1cccnc1',         # 吡啶異構體
        'c1ccncc1',         # 吡嗪
        
        # 甲基取代的吡啶
        'c1cc(C)ccn1',      # 4-甲基吡啶
        'c1c(C)cccn1',      # 3-甲基吡啶
        'c1cccc(C)n1',      # 6-甲基吡啶 (2-甲基吡啶)
        
        # 二甲基、三甲基取代的吡啶
        'c1cc(C)cc(C)n1',   # 2,4-二甲基吡啶
        'c1c(C)cc(C)cn1',   # 3,5-二甲基吡啶
        'c1cc(C)c(C)cn1',   # 4,5-二甲基吡啶
        'c1c(C)ccc(C)n1',   # 3,6-二甲基吡啶
        'c1c(C)c(C)ccn1',   # 3,4-二甲基吡啶
        'c1cc(C)c(C)c(C)n1', # 2,4,5-三甲基吡啶
        
        # 取代的吡啶
        'c1cc(OC)ccn1',     # 4-甲氧基吡啶
        'c1cc(c2ccccc2)ccn1', # 4-苯基吡啶
        'c1c(c2ccccc2)cccn1', # 3-苯基吡啶
    ]
    
    # 檢查是否匹配基本吡啶模式
    for pattern in pyridine_patterns:
        if pattern in smiles:
            # 嘗試找到吡啶環中的氮原子
            patt = Chem.MolFromSmarts('n:1ccccc:1')  # 標準吡啶環中的氮
            if patt and mol.HasSubstructMatch(patt):
                matches = mol.GetSubstructMatches(patt)
                for match in matches:
                    # 吡啶氮原子是第一個原子
                    n_idx = match[0]
                    if not any(n_idx == x[0] for x in dissociable_atoms):
                        dissociable_atoms.append((n_idx, 'pyridine_nitrogen'))
    
    # 如果上述方法未找到氮原子，嘗試直接尋找
    if not dissociable_atoms:
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'N' and atom.GetIsAromatic():
                # 確保是在六元環中
                if any(atom.IsInRingSize(6) for _ in range(1, 7)):
                    if not any(atom.GetIdx() == x[0] for x in dissociable_atoms):
                        dissociable_atoms.append((atom.GetIdx(), 'pyridine_nitrogen'))
    
    return dissociable_atoms

# 更新預測函數，整合所有專門處理函數
def predict_dissociable_atoms(smiles):
    """預測分子中可能的解離原子，增強對取代吡啶和複雜結構的支持"""
    try:
        # 首先檢查特殊映射
        if smiles in SPECIAL_STRUCTURES_MAP:
            return SPECIAL_STRUCTURES_MAP[smiles]
        
        # 嘗試標準SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # 嘗試修復SMILES語法問題
            smiles_sanitized = sanitize_smiles(smiles)
            mol = Chem.MolFromSmiles(smiles_sanitized)
            
            if mol is None:
                print(f"無法解析SMILES（即使嘗試修復後）: {smiles}")
                return analyze_smiles_directly(smiles)
        
        # 添加隱式氫原子
        try:
            mol = Chem.AddHs(mol)
        except Exception as e:
            print(f"添加氫原子失敗: {smiles}，{e}，嘗試不添加氫繼續處理")
        
        # 檢查是否是複雜的喹啉衍生物
        if '[Na+]' in smiles and '[S]' in smiles and 'cccnc' in smiles:
            atoms = analyze_complex_quinolines(mol, smiles)
            if atoms:
                return atoms
        
        # 檢查是否是取代的吡啶結構
        if ('c1cc(C)' in smiles or 'c1c(C)' in smiles) and 'n' in smiles:
            atoms = analyze_substituted_pyridines(mol, smiles)
            if atoms:
                return atoms
        
        # 檢查是否是吡嗪類結構
        if 'cnn' in smiles or 'ncn' in smiles:
            atoms = analyze_diazines(mol, smiles)
            if atoms:
                return atoms
        
        # 檢查是否是典型吡啶結構
        if ('ccccn' in smiles or 'cccnc' in smiles or 'ccncc' in smiles):
            atoms = analyze_pyridine_structures(mol, smiles)
            if atoms:
                return atoms
        
        # ... [繼續原有的檢查順序] ...
        
        # 使用標準解離原子分析
        atoms = get_dissociable_atoms(mol)
        
        # 如果仍未找到解離位點，使用直接SMILES分析
        if not atoms:
            atoms = analyze_smiles_directly(smiles)
        
        return atoms
        
    except Exception as e:
        print(f"預測解離原子時出錯: {e}, SMILES: {smiles}")
        return analyze_smiles_directly(smiles)


if __name__ == "__main__":
    import argparse
    import re
    
    parser = argparse.ArgumentParser(description='將NIST數據轉換為模型需要的格式，強化環狀化合物支持')
    parser.add_argument('--input', required=True, help='輸入NIST數據文件路徑')
    parser.add_argument('--output', default='OMGNN/data/nist_pka_data.csv', help='輸出文件路徑')
    parser.add_argument('--verbose', action='store_true', help='顯示詳細日誌')
    
    args = parser.parse_args()
    
    # 設置詳細日誌
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.INFO)
    
    # 執行轉換
    convert_nist_data(args.input, args.output)
