import warnings
from rdkit import Chem
from rdkit.Chem import AllChem, rdPartialCharges
from sklearn.cluster import KMeans
import numpy as np
import torch
from torch_geometric.data import Data
from typing import Union
import matplotlib.pyplot as plt
import pandas as pd

# 忽略 RDKit 的部分警告
warnings.filterwarnings("ignore", category=UserWarning, module='rdkit')

# --- 1. SMILES 轉 PyG Data (可選，此處主要用 RDKit) ---
def smiles_to_pyg_data(smiles: str) -> Union[Data, None]:
    """將 SMILES 轉換為 PyG Data 物件 (此範例中主要用 RDKit 特徵)"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # 計算 Gasteiger 部分電荷作為原子特徵示例
    try:
        AllChem.ComputeGasteigerCharges(mol)
    except Exception as e:
        print(f"無法計算 {smiles} 的 Gasteiger charges: {e}")
        # 分配預設電荷或跳過
        for atom in mol.GetAtoms():
             atom.SetDoubleProp('_GasteigerCharge', 0.0) # 設置預設值

    atom_features = []
    for atom in mol.GetAtoms():
        # 可以加入更多原子特徵，例如原子序數、雜化類型等
        charge = atom.GetDoubleProp('_GasteigerCharge')
        # 檢查 charge 是否為 NaN 或 inf
        if not np.isfinite(charge):
             charge = 0.0 # 如果無效，設為 0
        atom_features.append([charge])

    x = torch.tensor(atom_features, dtype=torch.float)

    # 獲取邊（鍵）
    adj = Chem.GetAdjacencyMatrix(mol)
    edge_index_list = np.nonzero(adj)
    edge_index = torch.tensor(edge_index_list, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, smiles=smiles)
    return data

# --- 2. 定義官能基 SMARTS ---
FUNCTIONAL_GROUPS = {
    "COOH": Chem.MolFromSmarts('[$([CX3](=O)[OX2H1]),$([CX3+1](=O)[OX1H0])]'), # 羧酸
    "PrimaryAmine": Chem.MolFromSmarts('[NX3;H2;!$(N=O)]'), # 一級胺 (NH2)
    # 可以添加更多官能基，例如：
    # "Alcohol": Chem.MolFromSmarts('[OX2H]'),
    # "Phenol": Chem.MolFromSmarts('[OX2H][c]'),
    # "Ammonium": Chem.MolFromSmarts('[N+;H3]'), # 例如質子化的胺基
}

# --- 3. 提取官能基特徵 ---
def extract_functional_group_features(smiles: str) -> list[tuple[str, np.ndarray, tuple[int, ...]]]:
    """
    從 SMILES 中提取官能基及其特徵。
    返回: [(group_name, feature_vector, atom_indices), ...]
    特徵向量: 這裡使用官能基內原子的平均 Gasteiger 電荷作為示例。
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"警告：無法解析 SMILES '{smiles}'")
        return []

    try:
        # 計算 Gasteiger 部分電荷
        rdPartialCharges.ComputeGasteigerCharges(mol)
    except Exception as e:
        print(f"警告：無法計算 {smiles} 的 Gasteiger charges: {e}")
        # 分配預設電荷或跳過
        for atom in mol.GetAtoms():
             atom.SetDoubleProp('_GasteigerCharge', 0.0) # 設置預設值

    features = []
    for name, pattern in FUNCTIONAL_GROUPS.items():
        if pattern is None:
            print(f"警告：官能基 '{name}' 的 SMARTS 模式無效")
            continue
        matches = mol.GetSubstructMatches(pattern)
        for match_indices in matches:
            group_charges = []
            valid_match = True
            for idx in match_indices:
                try:
                    charge = mol.GetAtomWithIdx(idx).GetDoubleProp('_GasteigerCharge')
                    # 檢查 charge 是否為 NaN 或 inf
                    if not np.isfinite(charge):
                         print(f"警告：原子 {idx} 在 {smiles} 中的 Gasteiger charge 無效 ({charge})，使用 0 代替。")
                         charge = 0.0 # 如果無效，設為 0
                    group_charges.append(charge)
                except KeyError:
                    print(f"警告：原子 {idx} 在 {smiles} 中缺少 '_GasteigerCharge' 屬性。")
                    valid_match = False
                    break # 如果任何一個原子缺少電荷，這個匹配可能無效
            
            if valid_match and group_charges:
                # 特徵向量：使用平均電荷作為簡單示例
                feature_vector = np.array([np.mean(group_charges)])
                features.append((name, feature_vector, match_indices))
            elif not group_charges and valid_match:
                 print(f"警告：官能基 {name} 在 {smiles} 的匹配 {match_indices} 中未找到有效電荷。")


    return features

# --- 4. 準備數據並進行 K-means 分群 ---
def cluster_functional_groups(smiles_list: list[str], n_clusters: int = 2):
    """
    對一系列 SMILES 中的官能基進行特徵提取和 K-means 分群。
    """
    all_features = []
    group_info = [] # (smiles, group_name, atom_indices)

    print("正在提取官能基特徵...")
    for smiles in smiles_list:
        extracted = extract_functional_group_features(smiles)
        for name, feat, indices in extracted:
            all_features.append(feat)
            group_info.append({"smiles": smiles, "group_name": name, "atom_indices": indices})

    if not all_features:
        print("錯誤：未找到任何官能基或無法提取特徵。")
        return None, None

    X = np.array(all_features)
    
    # 檢查 X 中是否有 NaN 或 Inf
    if not np.all(np.isfinite(X)):
        print("警告：特徵矩陣中包含 NaN 或 Inf 值。嘗試用 0 填充。")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
    if X.shape[0] < n_clusters:
        print(f"警告：找到的官能基數量 ({X.shape[0]}) 少於指定的集群數量 ({n_clusters})。無法執行 K-means。")
        return None, None
        
    print(f"使用 {X.shape[0]} 個官能基特徵進行 K-means 分群 (k={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) # n_init suppresses warning
    try:
       kmeans.fit(X)
       labels = kmeans.labels_
       centroids = kmeans.cluster_centers_
    except ValueError as e:
       print(f"K-means 執行時出錯：{e}")
       print("特徵矩陣 X:")
       print(X)
       return None, None


    # --- 5. 分析和解釋分群結果 ---
    print("\n分群結果分析:")
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(group_info[i])

    cluster_summary = {}
    for label, items in clusters.items():
        print(f"\n--- Cluster {label} (Centroid: {centroids[label]}) ---")
        group_counts = {}
        smiles_examples = set()
        for item in items:
            print(f"  SMILES: {item['smiles']}, Group: {item['group_name']}, Indices: {item['atom_indices']}")
            smiles_examples.add(item['smiles'])
            group_counts[item['group_name']] = group_counts.get(item['group_name'], 0) + 1
        
        # 試圖自動判斷這個 cluster 主要代表哪個官能基
        likely_group = max(group_counts, key=group_counts.get) if group_counts else "未知"
        # 根據官能基和/或中心點的特徵值（平均電荷）推斷 pKa 類別
        # 這裡是一個簡化的推斷：假設電荷越負（或平均值越小）對應 COOH (低 pKa)
        # 假設電荷越不負（或平均值越大）對應 Amine (高 pKa) - 這是一個非常粗略的假設！
        pka_category = "未知"
        if centroids[label][0] < -0.2: # 閾值需要根據實際數據調整
             pka_category = "low pKa (COOH)"
        elif centroids[label][0] > -0.1: # 閾值需要根據實際數據調整
             pka_category = "high pKa (Amine)"

        print(f"  主要官能基: {likely_group}")
        print(f"  官能基分佈: {group_counts}")
        print(f"  推斷的 pKa 類別: {pka_category}")
        cluster_summary[label] = {"centroid": centroids[label], "likely_group": likely_group, "pka_category": pka_category, "count": len(items)}


    # --- 6. 自標籤化 (示例) ---
    print("\n--- Self-Labeling 示例 ---")
    # 對於一個新的或已有的 SMILES，找到它的官能基，預測它們屬於哪個 cluster
    example_smiles = smiles_list[0] if smiles_list.size > 0 else None
    if example_smiles:
        print(f"分析 SMILES: {example_smiles}")
        example_features = extract_functional_group_features(example_smiles)
        if example_features:
             for name, feat, indices in example_features:
                 feat_clean = np.nan_to_num(feat.reshape(1, -1), nan=0.0, posinf=0.0, neginf=0.0)
                 if np.all(np.isfinite(feat_clean)):
                      pred_label = kmeans.predict(feat_clean)[0]
                      summary = cluster_summary.get(pred_label)
                      pka_label = summary['pka_category'] if summary else "無法預測"
                      print(f"  官能基: {name} (原子 {indices}), 特徵: {feat}, 預測 Cluster: {pred_label} ({pka_label})")
                 else:
                      print(f"  官能基: {name} (原子 {indices}), 特徵: {feat}, 無法預測 (無效特徵)")
        else:
             print(f"  在 {example_smiles} 中未找到定義的官能基。")

    # --- 7. 視覺化分群結果 (使用 Matplotlib) ---
    if labels is not None and X is not None:
        print("\n正在繪製分群結果圖...")
        plt.figure(figsize=(10, 4))

        # 為每個點添加一點垂直抖動以便觀察
        y_jitter = np.random.rand(X.shape[0]) * 0.1 - 0.05

        # 繪製每個數據點，根據其 cluster 標籤著色
        scatter = plt.scatter(X[:, 0], y_jitter, c=labels, cmap='viridis', alpha=0.7, label='Functional Groups')

        # 繪製 cluster 中心點
        plt.scatter(centroids[:, 0], np.zeros(centroids.shape[0]), c='red', marker='X', s=100, label='Centroids')

        plt.xlabel("Average Gasteiger Charge (Feature)")
        plt.ylabel("Jitter") # Y 軸沒有實際意義，僅用於分散點
        plt.title("K-Means Clustering of Functional Groups based on Average Charge")
        # 創建圖例
        handles, current_labels = scatter.legend_elements(prop='colors', alpha=0.6)
        # 更新圖例標籤以顯示 pKa 推斷
        legend_labels = []
        unique_labels = np.unique(labels)
        for label_idx in unique_labels:
            pka_cat = cluster_summary.get(label_idx, {}).get('pka_category', f'Cluster {label_idx}')
            legend_labels.append(f"{pka_cat} (Cluster {label_idx})")
        # 添加中心點的圖例
        handles.append(plt.Line2D([0], [0], marker='X', color='w', label='Centroids', markerfacecolor='red', markersize=10))
        legend_labels.append('Centroids')
        plt.legend(handles=handles, labels=legend_labels, title="Clusters & pKa")

        plt.grid(True, axis='x', linestyle='--', alpha=0.5)
        plt.yticks([]) # 隱藏 Y 軸刻度，因為它沒有實際意義
        print("圖表已生成，請查看彈出的視窗。")
        plt.savefig('../src/functional_group_clustering.png')
        plt.show()

    return kmeans, cluster_summary

# --- 示例用法 ---
if __name__ == "__main__":
    # 示例 SMILES 列表 (包含羧酸和胺基)
    df = pd.read_csv('../data/NIST_database_onlyH_6TypeEq_pos_match_max_fg_other.csv')
    smiles_list = df['SMILES'].unique()
    smiles_list = smiles_list[:10]
    
    example_smiles_list = [
        "CCO",                  # 乙醇 (無目標官能基)
        "CC(=O)O",              # 醋酸 (COOH)
        "N[C@@H](C)C(=O)O",     # 丙氨酸 (NH2, COOH)
        "C1=CC=C(C=C1)C(=O)O",  # 苯甲酸 (COOH)
        "CCN",                  # 乙胺 (PrimaryAmine)
        "O=C(O)CCC(=O)O",      # 丁二酸 (兩個 COOH)
        "NCCO",                 # 乙醇胺 (PrimaryAmine, Alcohol - 若定義)
        "C(C(=O)O)N",          # 甘氨酸 (COOH, PrimaryAmine)
        "CC(N)C",               # 異丙胺 (PrimaryAmine)
        "Oc1ccccc1C(=O)O",      # 水楊酸 (Phenol - 若定義, COOH)
        "N",                    # 氨 (會匹配 PrimaryAmine 嗎？SMARTS 需要精確) -> 不會，因為 H 數不對
        "CCCCN",                # 丁胺 (PrimaryAmine)
        "CCCCC(=O)O",          # 戊酸 (COOH)
        #"Invalid-SMILES",       # 無效SMILES測試
    ]

    # 執行分群
    # 我們期望 COOH 和 PrimaryAmine 分成兩群
    kmeans_model, summary = cluster_functional_groups(smiles_list, n_clusters=2)

    if kmeans_model and summary:
        print("\n分群模型已訓練完成。")
        print("集群摘要:")
        for label, data in summary.items():
             print(f" Cluster {label}: 重心={data['centroid']}, 主要官能基={data['likely_group']}, 預計pKa={data['pka_category']}, 數量={data['count']}")
    else:
        print("\n分群失敗。")

    # # 可選：測試 PyG 圖轉換
    # print("\n測試 PyG 圖轉換:")
    # for s in example_smiles_list[:3]:
    #     pyg_data = smiles_to_pyg_data(s)
    #     if pyg_data:
    #         print(f"SMILES: {s}, PyG Data: {pyg_data}")
    #     else:
    #         print(f"SMILES: {s}, 無法轉換")
