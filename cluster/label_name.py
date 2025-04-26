from collections import Counter
import pandas as pd
import ast
import re

def extract_common_term(name_list):
    # 拆成小寫單詞（過濾太短的詞）
    all_words = []
    for name in name_list:
        words = re.findall(r'\b[a-zA-Z\-]+\b', name.lower())
        all_words.extend([w for w in words if len(w) > 3])
    
    # 統計最常出現的詞（排除過於通用的詞）
    common = Counter(all_words)
    for word, count in common.most_common():
        if word not in ['acid', 'methyl', 'ethyl', 'group', 'yl']:  # 可擴充 stopwords
            return word
    return 'Unnamed'

# 讀進代表 SMILES 和名稱的檔案
rep_df = pd.read_csv("cluster_representatives.csv")

# 自動產生群組名稱
cluster_name_map = {}
for idx, row in rep_df.iterrows():
    try:
        names_list = ast.literal_eval(row["Sample_Names"])
        common_word = extract_common_term(names_list)
        cluster_name_map[row["Cluster"]] = f"{row['Cluster']}: {common_word}"
    except Exception:
        cluster_name_map[row["Cluster"]] = f"{row['Cluster']}: Unnamed"