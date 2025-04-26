import pubchempy as pcp
from tqdm import tqdm
import pandas as pd
import multiprocessing as mp
from functools import partial
import time
import os
from pathlib import Path
import requests
import random
from urllib.parse import quote
import json
import ast

# 設定查詢超時時間（秒）
TIMEOUT = 5

# 設定重試次數
MAX_RETRIES = 3

# 設定最大執行緒數量
MAX_THREADS = 5

# 緩存檔案路徑
CACHE_FILE = "pubchem_cache.csv"

# 讀取或創建緩存
if os.path.exists(CACHE_FILE):
    cache_df = pd.read_csv(CACHE_FILE)
    cache = dict(zip(cache_df["SMILES"], cache_df["Name"]))
else:
    cache = {}

# 使用PubChem API查詢化學名稱
def query_pubchem(smi):
    try:
        compounds = pcp.get_compounds(smi, namespace='smiles')
        if compounds and compounds[0].iupac_name:
            return compounds[0].iupac_name
    except Exception as e:
        pass
    return None


# 使用NCI化學識別符解析器查詢
def query_nci(smi):
    try:
        encoded_smiles = quote(smi)
        url = f"https://cactus.nci.nih.gov/chemical/structure/{encoded_smiles}/iupac_name"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.text.strip()
    except:
        pass
    return None

# 處理單個SMILES的函數
def process_smiles(smi):
    # 檢查緩存
    if smi in cache and not pd.isna(cache[smi]):
        return cache[smi]
    
    # 依序嘗試不同的查詢服務，一旦有結果就返回
    name = None
    
    # 嘗試使用PubChem，帶重試機制
    for attempt in range(MAX_RETRIES):
        name = query_pubchem(smi)
        if name:
            break
        time.sleep(random.uniform(0.5, 2))  # 隨機延遲，避免被API限制
    
    # 如果PubChem沒有結果，嘗試NCI
    if not name:
        name = query_nci(smi)
        
    # 更新緩存
    cache[smi] = name
    return name

# 保存緩存
def save_cache():
    cache_df = pd.DataFrame({
        "SMILES": list(cache.keys()),
        "Name": list(cache.values())
    })
    cache_df.to_csv(CACHE_FILE, index=False)

# 讀取分子資料
print("讀取分子資料...")
smiles_df = pd.read_csv("clustered_ligands.csv")

# 手動設定要處理的範圍
smiles_df = smiles_df[:100]
# 設置進程池
# num_processes = max(1, mp.cpu_count() - 1)  # 留一個CPU核心給系統
num_processes = MAX_THREADS  # 使用固定數量的執行緒
print(f"使用 {num_processes} 個處理核心進行並行查詢")
pool = mp.Pool(processes=num_processes)

# 創建結果儲存DataFrame
result_df = pd.DataFrame({
    "Cluster": smiles_df["Cluster"],  # 保留群集信息
    "SMILES": smiles_df["SMILES"],  # 原始SMILES
    "Name": None  # 初始化Name為None
})

try:
    # 所有分子的SMILES列表
    all_smiles = smiles_df["SMILES"].tolist()
    
    # 使用多進程查詢所有分子
    print(f"開始查詢 {len(all_smiles)} 個分子的名稱...")
    names = list(tqdm(
        pool.imap(process_smiles, all_smiles),
        total=len(all_smiles),
        desc="查詢化學名稱"
    ))
    
    # 更新結果DataFrame的Name列
    result_df["Name"] = names
    
    # 保存緩存
    save_cache()
    
finally:
    # 關閉進程池
    pool.close()
    pool.join()
    
    # 最後保存緩存
    save_cache()


# 保存結果為新的格式
result_df.to_csv("ligand_names.csv", index=False)

# 計算統計信息
total_smiles = len(result_df)
total_names = result_df["Name"].notna().sum()

# 計算成功率並顯示統計
if total_smiles > 0:
    success_rate = (total_names / float(total_smiles)) * 100
else:
    success_rate = 0

print(f"總共處理了 {total_smiles} 個SMILES，成功識別了 {total_names} 個名稱 ({success_rate:.1f}%)")
print(f"成功識別的分子名稱已保存到 ligand_names.csv")

