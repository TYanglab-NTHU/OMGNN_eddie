import warnings
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdPartialCharges
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd # 如果要從 CSV 讀取訓練數據
from typing import Union, Dict, Tuple, List
import io # 需要 io 來處理圖像數據流 (如果用 matplotlib 整合法，現在先不用)
from rdkit.Geometry import Point2D # 新增導入 Point2D
import pandas as pd
import math # For ceiling function
# 忽略 RDKit 的部分警告
warnings.filterwarnings("ignore", category=UserWarning, module='rdkit')
warnings.filterwarnings("ignore", category=FutureWarning) # 忽略 K-means 的 n_init 警告

# --- 定義官能基 SMARTS 和標籤 ---
FUNCTIONAL_GROUPS_SMARTS = {
    # 優先級 1: 酸和胺 (最優先確定)
    "COOH": Chem.MolFromSmarts('[C;X3](=[O;X1])[O;X2H1]'),
    "SulfonicAcid": Chem.MolFromSmarts('S(=O)(=O)O'), # 再次簡化，期望能匹配
    "PhosphonicAcid": Chem.MolFromSmarts('P(=O)(O)O'), # 再次簡化
    "PrimaryAmine": Chem.MolFromSmarts('[NH2;!$(NC=O)]'), # N H2, 但 N 不連接到 C=O
    "SecondaryAmine": Chem.MolFromSmarts('[NH1;!$(NC=O)]'), # N H1, 但 N 不連接到 C=O
    "TertiaryAmine": Chem.MolFromSmarts('[NX3;H0;!$(NC=O)]'),# N H0, N 不連接到 C=O (理論上不需排除, 但保持一致)

    # 優先級 2: 醇和酚 (在酸和胺之後處理)
    "Phenol": Chem.MolFromSmarts('[OH1][c]'), # 匹配 O-H 和相鄰芳香 C
    "Alcohol": Chem.MolFromSmarts('[OH1][C;!c]'), # 匹配 O-H 和相鄰脂肪 C
}

# 定義標籤映射（保持不變）
GROUP_LABEL_MAP = {
    "COOH": 0, "SulfonicAcid": 1, "PhosphonicAcid": 2,
    "PrimaryAmine": 3, "SecondaryAmine": 4, "TertiaryAmine": 5,
    "Phenol": 6, "Alcohol": 7, "Other": 8
}
N_CLUSTERS = len(GROUP_LABEL_MAP)
LABEL_GROUP_MAP = {v: k for k, v in GROUP_LABEL_MAP.items()}

# --- 1. Labeling Function (分層邏輯) ---
def label_atoms_by_functional_group(smiles: str) -> tuple[Union[Chem.Mol, None], Union[np.ndarray, None]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None, None
    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0: return mol, None

    atom_labels = np.full(num_atoms, GROUP_LABEL_MAP["Other"], dtype=int)
    other_label = GROUP_LABEL_MAP["Other"]

    print(f"\nProcessing {smiles}...")

    # 定義標籤目標原子類型
    LABEL_TARGET_ATOMS = {
        "COOH": [6, 8], "SulfonicAcid": [16, 8], "PhosphonicAcid": [15, 8],
        "PrimaryAmine": [7], "SecondaryAmine": [7], "TertiaryAmine": [7],
        "Phenol": [8], "Alcohol": [8]
    }

    # --- 分層處理 ---
    # 層級 1: 酸和胺
    priority_1_groups = ["COOH", "SulfonicAcid", "PhosphonicAcid",
                         "PrimaryAmine", "SecondaryAmine", "TertiaryAmine"]
    for group_name in priority_1_groups:
        pattern = FUNCTIONAL_GROUPS_SMARTS.get(group_name)
        if pattern is None: continue
        label_to_assign = GROUP_LABEL_MAP.get(group_name)
        if label_to_assign is None: continue
        target_atom_rules = LABEL_TARGET_ATOMS.get(group_name)
        if target_atom_rules is None: continue

        try:
            matches = mol.GetSubstructMatches(pattern)
            if matches: print(f"  Found {group_name} matches: {matches}")
        except Exception as e:
            print(f"Warning: SMARTS match for '{group_name}' failed: {e}")
            continue

        for match_indices in matches:
            for atom_idx in match_indices:
                if atom_idx >= num_atoms: continue
                # 只有當前是 Other 才標記
                if atom_labels[atom_idx] == other_label:
                    atom = mol.GetAtomWithIdx(atom_idx)
                    if atom.GetAtomicNum() in target_atom_rules:
                        atom_labels[atom_idx] = label_to_assign
                        print(f"    ----> Labeled atom {atom_idx} ({atom.GetSymbol()}) as {group_name} ({label_to_assign}) [Priority 1]")

    # 層級 2: 酚和醇
    priority_2_groups = ["Phenol", "Alcohol"]
    for group_name in priority_2_groups:
        pattern = FUNCTIONAL_GROUPS_SMARTS.get(group_name)
        if pattern is None: continue
        label_to_assign = GROUP_LABEL_MAP.get(group_name)
        if label_to_assign is None: continue
        target_atom_rules = LABEL_TARGET_ATOMS.get(group_name) # 應該是 [8]
        if target_atom_rules is None: continue

        try:
            matches = mol.GetSubstructMatches(pattern)
            if matches: print(f"  Found {group_name} matches: {matches}")
        except Exception as e:
            print(f"Warning: SMARTS match for '{group_name}' failed: {e}")
            continue

        for match_indices in matches:
            # 對於酚/醇，我們只關心匹配到的氧原子
            for atom_idx in match_indices:
                if atom_idx >= num_atoms: continue
                atom = mol.GetAtomWithIdx(atom_idx)
                # 嚴格檢查：必須是氧原子(8) 且當前是 Other
                if atom.GetAtomicNum() == 8 and atom_labels[atom_idx] == other_label:
                    atom_labels[atom_idx] = label_to_assign
                    print(f"    ----> Labeled atom {atom_idx} ({atom.GetSymbol()}) as {group_name} ({label_to_assign}) [Priority 2]")
                    # 找到氧就可以跳出內層循環，因為 SMARTS 匹配可能包含相鄰碳
                    break

    print(f"Labeling for {smiles} complete.")
    mol.SetProp("_Name", smiles)
    return mol, atom_labels


# --- 2. 視覺化多個分子的原子分群結果 (子圖) ---
def visualize_multiple_atom_clusters(
    mols_labels_list: List[Tuple[Chem.Mol, np.ndarray]],
    label_map: Dict[int, str],
    n_clusters: int,
    filename="../src/atom_functional_group_subplots.png"
):
    """
    將多個分子繪製到子圖中，根據原子標籤（基於官能基）為原子著色。
    包含一個統一的圖例。
    """
    n_mols = len(mols_labels_list)
    if n_mols == 0:
        print("錯誤：沒有提供分子進行視覺化。")
        return

    # --- 計算子圖佈局 ---
    cols = math.ceil(math.sqrt(n_mols))
    rows = math.ceil(n_mols / cols)

    # --- 創建 Matplotlib 圖形和子圖 ---
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5 + 1.5, rows * 3.5 + 1.5), squeeze=False, dpi=300) # 稍微增加底部空間
    axes = axes.flatten()

    # --- ★ 自定義顏色列表 (確保區分度，特別是 SulfonicAcid 和 Other) ★ ---
    # 確保顏色數量與 N_CLUSTERS (來自 GROUP_LABEL_MAP 的長度) 匹配
    # 顏色順序對應標籤 0 到 8
    custom_colors_rgb = [
        (0.6, 0.8, 0.9, 1.0),  # 0 COOH: 淺藍
        (1.0, 0.6, 0.2, 1.0),  # 1 SulfonicAcid: 橘色
        (0.7, 0.7, 0.7, 1.0),  # 2 PhosphonicAcid: 灰色
        (1.0, 0.4, 0.4, 1.0),  # 3 PrimaryAmine: 紅色/粉紅
        (0.7, 0.5, 0.9, 1.0),  # 4 SecondaryAmine: 紫色
        (0.4, 0.8, 0.4, 1.0),  # 5 TertiaryAmine: 綠色
        (0.8, 0.8, 0.2, 1.0),  # 6 Phenol: 橄欖綠/黃綠
        (0.9, 0.5, 0.8, 1.0),  # 7 Alcohol: 紫紅/粉紫
        (1.0, 1.0, 0.6, 1.0),  # 8 Other: 淺黃
    ]
    # 檢查顏色數量是否正確
    if len(custom_colors_rgb) < N_CLUSTERS:
        print(f"警告：自定義顏色數量 ({len(custom_colors_rgb)}) 少於標籤類別數量 ({N_CLUSTERS})！將會出錯。")
        # 可以選擇填充默認顏色或拋出錯誤
        # 為了避免崩潰，這裡用灰色填充，但應修復顏色列表
        gray_color = (0.5, 0.5, 0.5, 1.0)
        custom_colors_rgb.extend([gray_color] * (N_CLUSTERS - len(custom_colors_rgb)))

    # 使用自定義顏色列表代替 colormap
    # colors_rgb = [cmap(i)[:3] for i in range(n_clusters)] # 舊代碼
    colors_rgb = [color[:3] for color in custom_colors_rgb[:N_CLUSTERS]] # 使用自定義顏色

    # --- 迭代繪製每個分子 ---
    img_size = (300, 300) # 每個子圖的大小
    for i, (mol, atom_labels) in enumerate(mols_labels_list):
        ax = axes[i]
        if mol is None or atom_labels is None:
            ax.text(0.5, 0.5, 'Error processing molecule', ha='center', va='center')
            ax.axis('off')
            continue

        num_atoms = mol.GetNumAtoms()
        # 確保標籤長度正確
        if num_atoms != len(atom_labels):
            print(f"警告：分子 {mol.GetProp('_Name')} 原子數與標籤數不符，跳過繪製。")
            ax.text(0.5, 0.5, 'Label mismatch', ha='center', va='center')
            ax.axis('off')
            continue

        # --- 準備高亮顏色 ---
        highlight_colors = {}
        for atom_idx, label in enumerate(atom_labels):
            if label >= 0 and label < n_clusters:
                highlight_colors[atom_idx] = colors_rgb[label]
            # else: # 不為無效標籤添加顏色

        # --- 使用 RDKit 繪製到 PNG 流 ---
        try:
            AllChem.Compute2DCoords(mol)
            drawer = Draw.MolDraw2DCairo(img_size[0], img_size[1])
            opts = drawer.drawOptions()
            opts.addAtomIndices = True
            opts.padding = 0.05
            opts.clearBackground = True
            opts.atomHighlightsAreCircles = True
            opts.highlightRadius = 0.35

            drawer.DrawMolecule(
                 mol,
                 highlightAtoms=list(highlight_colors.keys()),
                 highlightAtomColors=highlight_colors,
                 highlightBonds=[]
            )
            drawer.FinishDrawing()
            png_data = drawer.GetDrawingText()

            # --- 在 Matplotlib 子圖中顯示圖像 ---
            img = plt.imread(io.BytesIO(png_data), format='png')
            ax.imshow(img)
            smiles_title = mol.GetProp('_Name')
            # 如果 SMILES 太長，可以截斷顯示
            max_title_len = 31
            if len(smiles_title) > max_title_len:
                 # 如果標題太長，換到下一行顯示
                 smiles_title = smiles_title[:max_title_len] + "\n" + smiles_title[max_title_len:]
            ax.set_title(smiles_title, fontsize=14) # 設置子圖標題
            ax.axis('off') # 關閉坐標軸

        except Exception as e:
            print(f"繪製分子 {mol.GetProp('_Name')} 時發生錯誤: {e}")
            ax.text(0.5, 0.5, 'Drawing Error', ha='center', va='center')
            ax.axis('off')

    # --- 關閉多餘的子圖 ---
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # --- 創建並添加統一的圖例 --- (修改 ncol)
    legend_elements = []
    for label_idx in sorted(label_map.keys()):
        group_name = label_map[label_idx]
        # 使用對應的自定義顏色 (確保索引不出錯)
        if label_idx < len(custom_colors_rgb):
            color_rgba = custom_colors_rgb[label_idx]
        else:
            color_rgba = (0.0, 0.0, 0.0, 1.0) # 錯誤情況用黑色
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', label=group_name,
                           markersize=10, markerfacecolor=color_rgba))

    # 將圖例放在圖形底部，並 ★ 設置列數ncol ★
    ncol_legend = 5 # 嘗試分為 5 列，可以根據需要調整
    fig.legend(handles=legend_elements, loc='lower center', ncol=ncol_legend, title="Functional Groups")

    # --- 調整佈局並保存 --- (可能需要微調 rect)
    plt.tight_layout(rect=[0, 0.08, 1, 0.98]) # 稍微增加底部邊距以容納多行圖例
    plt.savefig(filename)
    print(f"\n包含 {n_mols} 個分子的子圖已保存至 {filename}")
    plt.close(fig)


# --- 示例用法 ---
if __name__ == "__main__":
    # --- 讀取 SMILES 數據 ---
    
    output_name = "../src/atom_functional_group_subplots_by_node.png"
    try:
        df = pd.read_csv('../data/NIST_database_onlyH_6TypeEq_pos_match_max_fg_other.csv')
        smiles_list_full = df['SMILES'].unique()
        # 只處理前 N 個分子以加快測試速度
        # N_to_process = 9 # 例如處理前 9 個
        # smiles_list = smiles_list_full[:N_to_process]
        smiles_list = smiles_list_full[119:128]
        #  smiles_list = [
        #         "CCO", "CC(=O)O", "N[C@@H](C)C(=O)O", "C1=CC=C(C=C1)C(=O)O", "CCN",
        #         "O=C(O)CCC(=O)O", "NCCO", "C(C(=O)O)N", "CC(N)C", "Oc1ccccc1C(=O)O",
        #         "CCCCN", "CCCCC(=O)O"
        #     ] # 使用範例
    except FileNotFoundError:
        print("錯誤：找不到 CSV 文件，將使用內建的示例列表。")
        smiles_list = [
            "CCO", "CC(=O)O", "N[C@@H](C)C(=O)O", "C1=CC=C(C=C1)C(=O)O", "CCN",
            "O=C(O)CCC(=O)O", "NCCO", "C(C(=O)O)N", "CC(N)C", "Oc1ccccc1C(=O)O",
            "CCCCN", "CCCCC(=O)O"
        ] # 回退示例

    print(f"將處理 {len(smiles_list)} 個 SMILES...")

    # --- 處理每個分子並收集結果 ---
    processed_molecules_labels = []
    for idx, smiles in enumerate(smiles_list):
        # print(f"Processing {idx+1}/{len(smiles_list)}: {smiles}") # 進度顯示
        target_mol, target_atom_labels = label_atoms_by_functional_group(smiles)
        if target_mol and target_atom_labels is not None:
            processed_molecules_labels.append((target_mol, target_atom_labels))
        else:
            print(f"  跳過分子 {smiles} (處理失敗)")


    # --- 視覺化結果 ---
    if processed_molecules_labels:
        visualize_multiple_atom_clusters(processed_molecules_labels, LABEL_GROUP_MAP, N_CLUSTERS, output_name)
    else:
        print("\n沒有成功處理的分子，無法生成視覺化圖像。")
