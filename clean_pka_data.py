import pandas as pd
import csv
import re
import os
import shutil

# 讀取數據文件
file_path = '../data/processed_pka_data.csv'
output_path = '../data/processed_pka_data_cleaned.csv'

# 儲存不一致的行
inconsistent_rows = []
# 儲存所有正確的行
consistent_rows = []

print("開始檢查 pKa 數據一致性...")

# 讀取文件並檢查每一行
with open(file_path, 'r') as file:
    reader = csv.reader(file)
    for row_idx, row in enumerate(reader, 1):
        if len(row) < 3:  # 跳過格式不正確的行
            continue
        
        try:
            smiles = row[0]
            pka_num = int(row[1])  # pKa 數量
            
            # 檢查 pKa 值的數量
            pka_values = row[2].strip('"').split(',')
            if pka_values == ['']:  # 處理空值情況
                actual_pka_count = 0
            else:
                actual_pka_count = len(pka_values)
            
            # 檢查是否一致
            if pka_num != actual_pka_count:
                inconsistent_rows.append({
                    'row_number': row_idx,
                    'smiles': smiles,
                    'expected_pka_count': pka_num,
                    'actual_pka_count': actual_pka_count,
                    'pka_values': row[2]
                })
            else:
                # 儲存一致的行
                consistent_rows.append(row)
        except Exception as e:
            inconsistent_rows.append({
                'row_number': row_idx,
                'error': str(e),
                'row_content': ','.join(row)
            })

# 輸出結果
print(f"總共發現 {len(inconsistent_rows)} 行資料的 pKa 數量不一致")
print(f"保留 {len(consistent_rows)} 行正確的資料")

# 將不一致的行保存到CSV文件
if inconsistent_rows:
    df = pd.DataFrame(inconsistent_rows)
    df.to_csv('pka_inconsistency_report.csv', index=False)
    print("已將不一致的行資料保存到 pka_inconsistency_report.csv")

# 將一致的行保存到新的CSV文件
with open(output_path, 'w', newline='') as file:
    writer = csv.writer(file)
    for row in consistent_rows:
        writer.writerow(row)

print(f"已將正確的資料保存到 {output_path}")

# 備份原始文件
backup_path = '../data/processed_pka_data_backup.csv'
if not os.path.exists(backup_path):
    shutil.copy(file_path, backup_path)
    print(f"已備份原始資料到 {backup_path}")

# 直接替換原始文件
shutil.copy(output_path, file_path)
print(f"已將清理後的資料複製到 {file_path}")
print("處理完成！") 