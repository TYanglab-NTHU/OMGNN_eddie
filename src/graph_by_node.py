import warnings
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdPartialCharges
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd # 如果要從 CSV 讀取訓練數據
from typing import Union
# 忽略 RDKit 的部分警告
warnings.filterwarnings("ignore", category=UserWarning, module='rdkit')
warnings.filterwarnings("ignore", category=FutureWarning) # 忽略 K-means 的 n_init 警告

# --- 1. 提取原子特徵 ---
def extract_atom_features(smiles_list: list) -> tuple:
    """
    從 SMILES 列表中提取所有原子的特徵（目前僅 Gasteiger 電荷）。
    返回: (特徵矩陣, 包含來源信息的列表)
    """
    all_atom_features = []
    atom_info = [] # (smiles, atom_index)

    print("正在從訓練數據中提取原子特徵...")
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"警告：無法解析訓練 SMILES '{smiles}'")
            continue

        try:
            rdPartialCharges.ComputeGasteigerCharges(mol)
        except Exception as e:
            print(f"警告：無法計算 {smiles} 的 Gasteiger charges: {e}。跳過此分子。")
            continue

        num_atoms = mol.GetNumAtoms()
        has_nan_charge = False
        features_for_mol = []
        indices_for_mol = []

        for atom_idx in range(num_atoms):
            try:
                charge = mol.GetAtomWithIdx(atom_idx).GetDoubleProp('_GasteigerCharge')
                if not np.isfinite(charge):
                    print(f"警告：原子 {atom_idx} 在 {smiles} 中的 Gasteiger charge 無效 ({charge})，使用 0 代替。")
                    charge = 0.0
                    has_nan_charge = True # 標記此分子有問題
                features_for_mol.append([charge]) # 特徵向量可以擴展
                indices_for_mol.append(atom_idx)
            except KeyError:
                print(f"警告：原子 {atom_idx} 在 {smiles} 中缺少 '_GasteigerCharge' 屬性。跳過此原子。")
                has_nan_charge = True # 標記此分子有問題
                features_for_mol.append([0.0]) # 添加一個預設值以保持長度一致，但此分子應被注意
                indices_for_mol.append(atom_idx)


        # 如果分子中沒有任何有效的電荷，則跳過
        if has_nan_charge and not features_for_mol:
             print(f"警告: 分子 {smiles} 因缺少有效電荷而被完全跳過。")
             continue

        all_atom_features.extend(features_for_mol)
        for atom_idx in indices_for_mol:
             atom_info.append({"source_smiles_idx": i, "atom_index": atom_idx})

    if not all_atom_features:
        print("錯誤：未能從任何分子中提取原子特徵。")
        return None, []

    X = np.array(all_atom_features)
    
    # 再次檢查 X 中是否有 NaN 或 Inf (理論上已被處理，但作為保險)
    if not np.all(np.isfinite(X)):
        print("警告：最終的原子特徵矩陣中包含 NaN 或 Inf 值。嘗試用 0 填充。")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
    print(f"從 {len(smiles_list)} 個 SMILES 中提取了 {X.shape[0]} 個原子的特徵。")
    return X, atom_info

# --- 2. 訓練原子 K-means 模型 ---
def train_atom_kmeans(atom_features: np.ndarray, n_clusters: int = 3):
    """
    使用原子特徵訓練 K-means 模型。
    """
    if atom_features is None or atom_features.shape[0] < n_clusters:
        print(f"錯誤：原子特徵數量不足 ({atom_features.shape[0] if atom_features is not None else 0}) 無法進行 K-means 分群 (k={n_clusters})。")
        return None

    print(f"使用 {atom_features.shape[0]} 個原子特徵進行 K-means 分群 (k={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    try:
       kmeans.fit(atom_features)
       print("原子 K-means 模型訓練完成。")
       print(f"中心點: {kmeans.cluster_centers_}")
       return kmeans
    except ValueError as e:
       print(f"K-means 執行時出錯：{e}")
       print("特徵矩陣 X:")
       print(atom_features)
       return None

# --- 3. 為單一分子標記原子 ---
def label_molecule_atoms(smiles: str, kmeans_model: KMeans) -> tuple[Union[Chem.Mol, None], Union[np.ndarray, None]]:
    """
    使用訓練好的 K-means 模型為單一分子的原子進行標籤預測。
    返回: (RDKit Mol 物件, 原子標籤陣列)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"錯誤：無法解析目標 SMILES '{smiles}'")
        return None, None

    try:
        rdPartialCharges.ComputeGasteigerCharges(mol)
    except Exception as e:
        print(f"警告：無法計算目標分子 {smiles} 的 Gasteiger charges: {e}。")
        # 嘗試分配預設電荷
        for atom in mol.GetAtoms():
            if not atom.HasProp('_GasteigerCharge'):
                 atom.SetDoubleProp('_GasteigerCharge', 0.0)

    atom_features = []
    valid_indices = []
    num_atoms = mol.GetNumAtoms()
    
    if num_atoms == 0:
        print(f"錯誤：目標分子 {smiles} 沒有原子。")
        return mol, None

    for atom_idx in range(num_atoms):
        try:
            charge = mol.GetAtomWithIdx(atom_idx).GetDoubleProp('_GasteigerCharge')
            if not np.isfinite(charge):
                 print(f"警告：目標分子原子 {atom_idx} 的 Gasteiger charge 無效 ({charge})，使用 0 代替。")
                 charge = 0.0
            atom_features.append([charge])
            valid_indices.append(atom_idx)
        except KeyError:
            print(f"警告：目標分子原子 {atom_idx} 缺少 '_GasteigerCharge' 屬性，使用 0 代替。")
            atom_features.append([0.0]) # 使用預設值
            valid_indices.append(atom_idx)

    if not atom_features:
        print(f"錯誤：未能為目標分子 {smiles} 提取任何原子特徵。")
        return mol, None

    X_target = np.array(atom_features)
    
    if not np.all(np.isfinite(X_target)):
        print("警告：目標分子的特徵矩陣中包含 NaN 或 Inf 值。嘗試用 0 填充。")
        X_target = np.nan_to_num(X_target, nan=0.0, posinf=0.0, neginf=0.0)
        
    if X_target.shape[0] == 0:
         print(f"錯誤: 目標分子 {smiles} 特徵矩陣為空。")
         return mol, None

    print(f"\n正在為目標分子 {smiles} 的 {X_target.shape[0]} 個原子預測 cluster...")
    atom_labels = kmeans_model.predict(X_target)
    
    # 確保標籤數量與原子數量一致 (以防萬一有原子被跳過)
    final_labels = np.full(num_atoms, -1, dtype=int) # -1 代表未標記或錯誤
    for i, idx in enumerate(valid_indices):
        if idx < num_atoms:
             final_labels[idx] = atom_labels[i]
        
    return mol, final_labels


# --- 4. 視覺化原子分群結果 ---
def visualize_atom_clusters(mol: Chem.Mol, atom_labels: np.ndarray, n_clusters: int, filename="atom_clustering.png"):
    """
    將分子繪製出來，並根據原子標籤為原子著色。
    """
    if mol is None or atom_labels is None:
        print("錯誤：缺少分子或原子標籤，無法進行視覺化。")
        return

    num_atoms = mol.GetNumAtoms()
    if num_atoms != len(atom_labels):
        print(f"警告：原子數量 ({num_atoms}) 與標籤數量 ({len(atom_labels)}) 不符。視覺化可能不準確。")
        # 嘗試調整標籤數組大小，用-1填充
        if len(atom_labels) < num_atoms:
            padded_labels = np.full(num_atoms, -1, dtype=int)
            padded_labels[:len(atom_labels)] = atom_labels
            atom_labels = padded_labels
        else: # 標籤比原子多，截斷
             atom_labels = atom_labels[:num_atoms]

    # 創建顏色映射
    # 使用 'viridis', 'plasma', 'inferno', 'magma', or 'cividis' 等感知統一的色圖
    cmap = plt.get_cmap('viridis', n_clusters)
    colors = [cmap(i) for i in range(n_clusters)]
    
    # 為 RDKit 準備顏色字典 {atom_index: (r, g, b, a)}
    highlight_colors = {}
    valid_labels_present = False
    for i, label in enumerate(atom_labels):
        if label >= 0 and label < n_clusters: # 確保標籤有效
            highlight_colors[i] = colors[label]
            valid_labels_present = True
        else:
             print(f"警告：原子 {i} 的標籤 {label} 無效，將不被著色。")

    if not valid_labels_present:
        print("錯誤: 沒有有效的原子標籤可供視覺化。")
        return
    
    if not highlight_colors:
        print("錯誤: 無法生成高亮顏色字典。")
        return


    # 繪製分子圖像
    img_size = (400, 400)
    try:
        # 計算 2D 坐標
        AllChem.Compute2DCoords(mol)
        
        drawer = Draw.MolDraw2DCairo(img_size[0], img_size[1])
        # 設置繪圖選項
        opts = drawer.drawOptions()
        opts.addAtomIndices = True # 顯示原子編號
        opts.legendFontSize = 15
        #opts.highlightBondWidthMultiplier = 10 # 調整高亮鍵的寬度倍數
        #opts.highlightRadius = 0.3 # 調整高亮原子的半徑
        opts.clearBackground = True # 設定背景不要透明，方便後續處理
        opts.padding = 0.1 # 增加邊距

        # 繪製分子並高亮原子
        drawer.DrawMolecule(
            mol,
            highlightAtoms=list(highlight_colors.keys()),
            highlightAtomColors=highlight_colors,
            highlightBonds=[] # 暫不高亮鍵
        )
        drawer.FinishDrawing()
        
        # 保存為 PNG 圖像
        drawer.WriteDrawingText(filename)
        print(f"分子圖像已保存至 {filename}")

        # --- 創建並保存獨立的圖例 ---
        fig_legend, ax_legend = plt.subplots(figsize=(3, n_clusters * 0.5))
        ax_legend.set_title("Atom Clusters")
        
        # 添加圖例元素
        legend_elements = []
        for i in range(n_clusters):
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {i}',
                               markersize=10, markerfacecolor=colors[i]))
                               
        ax_legend.legend(handles=legend_elements, loc='center')
        ax_legend.axis('off') # 隱藏坐標軸
        plt.tight_layout()
        legend_filename = filename.replace(".png", "_legend.png")
        plt.savefig(legend_filename)
        print(f"圖例已保存至 {legend_filename}")
        plt.close(fig_legend) # 關閉圖例視窗


    except Exception as e:
        print(f"繪製分子時發生錯誤: {e}")
        import traceback
        traceback.print_exc()


# --- 示例用法 ---
if __name__ == "__main__":
    # --- 1. 準備訓練數據 ---
    # 使用之前的示例列表或從 CSV 加載
    # 選項 A: 使用示例列表
    training_smiles_list = [
        "CCO", "CC(=O)O", "N[C@@H](C)C(=O)O", "C1=CC=C(C=C1)C(=O)O", "CCN",
        "O=C(O)CCC(=O)O", "NCCO", "C(C(=O)O)N", "CC(N)C", "Oc1ccccc1C(=O)O",
        "CCCCN", "CCCCC(=O)O"
    ]
    # 選項 B: 從 CSV 加載 (如果需要)
    # try:
    #     df = pd.read_csv('../data/your_data.csv') # 修改為你的 CSV 文件路徑
    #     training_smiles_list = df['SMILES'].unique()
    #     print(f"從 CSV 加載了 {len(training_smiles_list)} 個 unique SMILES 作為訓練數據。")
    # except FileNotFoundError:
    #     print("錯誤：找不到 CSV 文件，將使用內建的示例列表。")
    #     # 回退到選項 A

    # --- 2. 提取訓練特徵並訓練模型 ---
    num_atom_clusters = 3 # 嘗試將原子分為 3 群 (例如：缺電子、富電子、中性)
    atom_features_matrix, _ = extract_atom_features(training_smiles_list)
    kmeans_atom_model = train_atom_kmeans(atom_features_matrix, n_clusters=num_atom_clusters)

    # --- 3. 設定目標分子並進行標籤預測 ---
    target_smiles = "N[C@@H](C)C(=O)O" # 選擇一個分子進行標籤化，例如丙氨酸

    if kmeans_atom_model:
        target_mol, target_atom_labels = label_molecule_atoms(target_smiles, kmeans_atom_model)
        
        # 打印標籤結果
        if target_mol and target_atom_labels is not None:
            print(f"\n目標分子 {target_smiles} 的原子標籤:")
            for i in range(target_mol.GetNumAtoms()):
                atom = target_mol.GetAtomWithIdx(i)
                print(f"  原子 {i} ({atom.GetSymbol()}): Cluster {target_atom_labels[i]}")

            # --- 4. 視覺化結果 ---
            visualize_atom_clusters(target_mol, target_atom_labels, num_atom_clusters, filename="../src/atom_clustering_visualization.png")
    else:
        print("\n由於原子 K-means 模型訓練失敗，無法進行後續步驟。")
