#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
化學分子指紋特徵降維與分群
============================
這個腳本對SMILES分子指紋進行降維與分群，包含以下步驟：
1. 將分子SMILES轉換為Morgan指紋
2. 使用PCA降維到100維
3. 使用t-SNE將PCA結果降到2維（用於視覺化）
4. 使用K-Means進行分群
5. 將結果可視化並保存
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import rdkit
# 關閉rdkit的警告
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem import rdFingerprintGenerator
from tqdm import tqdm
import argparse
import pickle  # 添加pickle模組

# ===================== 參數設定 =====================
# 可以透過命令列參數自訂
parser = argparse.ArgumentParser(description='分子降維與分群分析')
parser.add_argument('--input', type=str, default="all_ligand.txt", 
                    help='SMILES輸入檔案路徑')
parser.add_argument('--output', type=str, default="clustered_ligands.csv", 
                    help='結果輸出CSV檔案路徑')
parser.add_argument('--fp_size', type=int, default=2048, 
                    help='Morgan指紋大小')
parser.add_argument('--radius', type=int, default=2, 
                    help='Morgan指紋半徑')
parser.add_argument('--pca_components', type=int, default=100, 
                    help='PCA降維後的維度')
parser.add_argument('--n_clusters', type=int, default=10, 
                    help='K-Means分群數量')
parser.add_argument('--perplexity', type=int, default=30, 
                    help='t-SNE perplexity參數')
parser.add_argument('--tsne_iter', type=int, default=300, 
                    help='t-SNE迭代次數')
parser.add_argument('--target_cluster', type=int, default=5, 
                    help='要特別顯示的目標群組')
parser.add_argument('--random_state', type=int, default=42, 
                    help='隨機種子')

args = parser.parse_args()

def main():
    """執行整個分析流程"""
    
    # ===================== 步驟1: 讀取SMILES並計算指紋 =====================
    print("步驟1: 讀取SMILES並計算Morgan分子指紋")
    
    # 讀取SMILES列表
    with open(args.input, "r") as f:
        smiles_list = f.read().splitlines()
    
    # 建立Morgan指紋生成器
    fps_array = calculate_morgan_fingerprints(smiles_list)
    
    # ===================== 步驟2: PCA降維 =====================
    print(f"步驟2: 使用PCA降維到{args.pca_components}維")
    pca_result = run_pca(fps_array)
    
    # 使用pickle儲存PCA結果
    pca_file = f"pca_result_{args.pca_components}d.pkl"
    with open(pca_file, 'wb') as f:
        pickle.dump(pca_result, f)
    print(f"PCA結果已儲存至 {pca_file}")
    
    # ===================== 步驟3: t-SNE降維 (用於視覺化) =====================
    print("步驟3: 使用t-SNE降維到2維 (用於視覺化)")
    tsne_result = run_tsne(pca_result)
    
    # 使用pickle儲存t-SNE結果
    tsne_file = f"tsne_result_p{args.perplexity}.pkl"
    with open(tsne_file, 'wb') as f:
        pickle.dump(tsne_result, f)
    print(f"t-SNE結果已儲存至 {tsne_file}")
    
    # ===================== 步驟4: K-Means分群 =====================
    print(f"步驟4: 使用K-Means進行分群 (K={args.n_clusters})")
    km_labels = run_kmeans(pca_result)
    
    # 使用pickle儲存分群標籤
    pickle_file = f"kmeans_labels_k{args.n_clusters}.pkl"
    with open(pickle_file, 'wb') as f:
        pickle.dump(km_labels, f)
    print(f"分群標籤已儲存至 {pickle_file}")
    
    # 儲存完整的分析結果集
    results = {
        'pca_result': pca_result,
        'tsne_result': tsne_result,
        'km_labels': km_labels,
        'smiles': filtered_smiles,
        'parameters': {
            'pca_components': args.pca_components,
            'perplexity': args.perplexity,
            'n_clusters': args.n_clusters,
            'fp_size': args.fp_size,
            'radius': args.radius
        }
    }
    
    full_results_file = f"molecular_analysis_k{args.n_clusters}.pkl"
    with open(full_results_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"完整分析結果已儲存至 {full_results_file}")
    
    # ===================== 步驟5: 視覺化與保存結果 =====================
    print("步驟5: 視覺化分群結果並保存數據")
    # 視覺化結果
    visualize_clustering(tsne_result, km_labels)
    
    # 整合數據並保存
    save_results(smiles_list, km_labels, tsne_result)
    
    # 顯示特定群組
    show_target_cluster(smiles_list, km_labels, tsne_result)
    
    print(f"分析完成! 結果已保存至 {args.output}")

def calculate_morgan_fingerprints(smiles_list):
    """計算SMILES的Morgan分子指紋"""
    
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=args.radius, fpSize=args.fp_size)
    fps = []
    valid_smiles = []
    
    for smiles in tqdm(smiles_list, desc="處理SMILES"):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                # 使用生成器計算fingerprint
                fp = mfpgen.GetFingerprint(mol)
                fps.append(fp)
                valid_smiles.append(smiles)
        except Exception as e:
            print(f"無法解析SMILES: {smiles} 因為 {e}")
    
    # 將fingerprint轉換為numpy array
    fps_array = []
    for fp in fps:
        arr = np.zeros((args.fp_size,), dtype=int)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps_array.append(arr)
    
    # 更新smiles_list只包含有效的SMILES
    global filtered_smiles
    filtered_smiles = valid_smiles
    
    return np.array(fps_array)

def run_pca(X):
    """執行PCA降維"""
    
    start_time = time.time()
    
    # 計算PCA
    pca = PCA(n_components=args.pca_components)
    pca_result = pca.fit_transform(X)
    
    end_time = time.time()
    print(f"PCA完成，耗時 {end_time - start_time:.2f} 秒")
    
    # 顯示解釋變異量
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    # 繪製解釋變異量
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(explained_variance) + 1), cumulative_variance, 'o-', markersize=4)
    plt.axhline(y=0.8, color='r', linestyle='--', label='80% Explained Variance')
    plt.axhline(y=0.9, color='g', linestyle='--', label='90% Explained Variance')
    plt.grid(True)
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Cumulative Explained Variance')
    plt.legend()
    plt.savefig('PCA_explained_variance.png')
    plt.close()
    
    return pca_result

def run_tsne(pca_result):
    """執行t-SNE降維"""
    
    start_time = time.time()
    
    # 執行t-SNE
    tsne = TSNE(n_components=2, 
                perplexity=args.perplexity, 
                n_iter=args.tsne_iter, 
                random_state=args.random_state, 
                n_jobs=-1)
    tsne_result = tsne.fit_transform(pca_result)
    
    end_time = time.time()
    print(f"t-SNE完成，耗時 {end_time - start_time:.2f} 秒")
    
    return tsne_result

def run_kmeans(pca_result):
    """執行K-Means分群"""
    
    start_time = time.time()
    
    # 執行K-Means
    kmeans = KMeans(n_clusters=args.n_clusters, 
                    init='k-means++', 
                    n_init='auto', 
                    random_state=args.random_state)
    km_labels = kmeans.fit_predict(pca_result)
    
    end_time = time.time()
    print(f"K-Means完成，耗時 {end_time - start_time:.2f} 秒")
    
    return km_labels

def visualize_clustering(tsne_result, km_labels):
    """視覺化分群結果"""
    
    plt.figure(figsize=(12, 10))
    
    # 使用tab10顏色表，最多支援10種顏色
    colors = plt.cm.tab10(np.linspace(0, 1, args.n_clusters))
    
    for k, col in enumerate(colors):
        # 找出屬於當前群集k的點
        class_member_mask = (km_labels == k)
        
        # 獲取這些點在t-SNE上的座標
        xy = tsne_result[class_member_mask]
        
        # 繪製這些點
        plt.plot(xy[:, 0], xy[:, 1], 'o', 
                markerfacecolor=tuple(col),
                markeredgecolor='k',  # 加上黑色邊框
                markersize=6,
                alpha=0.6,
                label=f'Cluster {k}')  # 添加圖例標籤
    
    plt.title(f't-SNE Result Colored by K-Means Clustering (K={args.n_clusters})')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(True)
    
    # 如果群集數量不多，可以顯示圖例
    if args.n_clusters <= 15:
        plt.legend(loc='best', markerscale=1.5)
    
    plt.savefig(f'K-Means_tSNE_{args.n_clusters}.png', dpi=300)
    plt.show()

def save_results(smiles_list, km_labels, tsne_result):
    """整合並保存結果"""
    
    # 將SMILES, 分群標籤, t-SNE座標整合成DataFrame
    # 使用已過濾的SMILES列表（排除無法解析的SMILES）
    smiles_df = pd.DataFrame({
        'SMILES': filtered_smiles,  # 使用已過濾的SMILES列表
        'Cluster': km_labels,
        'tSNE-1': tsne_result[:, 0],
        'tSNE-2': tsne_result[:, 1]
    })
    
    # 依照Cluster排序
    smiles_df = smiles_df.sort_values(by='Cluster')
    
    # 儲存CSV
    smiles_df.to_csv(args.output, index=False)
    
    # 計算各群組的大小
    cluster_counts = smiles_df['Cluster'].value_counts().sort_index()
    print("\n各群集大小:")
    for cluster, count in cluster_counts.items():
        print(f"群集 {cluster}: {count} 個分子")

def show_target_cluster(smiles_list, km_labels, tsne_result):
    """顯示特定群組的詳細資訊"""
    
    # 讀取DataFrame
    smiles_df = pd.DataFrame({
        'SMILES': filtered_smiles,
        'Cluster': km_labels,
        'tSNE-1': tsne_result[:, 0],
        'tSNE-2': tsne_result[:, 1]
    })
    
    # 篩選目標群組
    target_cluster = args.target_cluster
    subset = smiles_df[smiles_df['Cluster'] == target_cluster].copy()
    
    # 顯示前幾個分子
    print(f"\n群集 {target_cluster} 的前10個SMILES：")
    print(subset[['SMILES']].head(10))
    
    # 繪製這個群組在t-SNE上的分佈
    plt.figure(figsize=(8, 6))
    plt.scatter(subset['tSNE-1'], subset['tSNE-2'], c='blue', alpha=0.6)
    plt.title(f'Cluster {target_cluster}')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(True)
    plt.savefig(f'Cluster_{target_cluster}.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    # 全局變量用於存儲有效的SMILES列表
    filtered_smiles = []
    main()