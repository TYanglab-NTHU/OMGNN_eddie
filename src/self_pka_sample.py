import torch
from self_pka_models import pka_GNN
import pandas as pd
import ast
import csv
import os
from rdkit import Chem
import numpy as np
from self_pka_trainutils import load_config_by_version, pka_Dataloader
from sklearn.model_selection import train_test_split


def print_result(smiles, result, true_pka):
    # ============打印結果============
    if result:
        print(f"SMILES: {smiles}")
        
        # 解析 true_pka 字符串為真實的位置和數值對
        true_pka_pairs = []
        if len(true_pka) > 0:
            try:
                # 將字符串轉換為 Python 對象
                true_pka_str = true_pka[0]
                true_pka_list = ast.literal_eval(true_pka_str)
                true_pka_pairs = [(int(pos), float(val)) for pos, val in true_pka_list]
            except Exception as e:
                print(f"解析真實 pKa 時出錯: {e}")
        
        # 創建預測的位置和數值對
        pred_pka_pairs = [(pos, result['pka_values'][i]) for i, pos in enumerate(result['pka_positions'])]
        
        # 打印結果比較表
        print("\npKa 位置和數值比較:")
        print("="*80)
        print(f"{'真實位置':<10}{'預測位置':<10}{'真實 pKa':<15}{'預測 pKa':<15}")
        print("-"*80)
        
        # 分別處理真實和預測的數據
        true_pos_set = {pos for pos, _ in true_pka_pairs}
        pred_pos_set = {pos for pos, _ in pred_pka_pairs}
        
        # 創建位置映射 - 嘗試將真實位置與最接近的預測位置配對
        pos_mapping = {}
        unmatched_true = set(true_pos_set)
        unmatched_pred = set(pred_pos_set)
        
        # 先輸出匹配的位置（相同位置的真實值和預測值）
        matched_positions = true_pos_set.intersection(pred_pos_set)
        for pos in sorted(matched_positions):
            true_val = next((v for p, v in true_pka_pairs if p == pos), None)
            pred_val = next((v for p, v in pred_pka_pairs if p == pos), None)
            print(f"{pos:<10}{pos:<10}{true_val:<15.2f}{pred_val:<15.2f}")
            unmatched_true.discard(pos)
            unmatched_pred.discard(pos)
        
        # 輸出未匹配的真實值
        for pos in sorted(unmatched_true):
            true_val = next((v for p, v in true_pka_pairs if p == pos), None)
            print(f"{pos:<10}{'-':<10}{true_val:<15.2f}{'-':<15}")
        
        # 輸出未匹配的預測值
        for pos in sorted(unmatched_pred):
            pred_val = next((v for p, v in pred_pka_pairs if p == pos), None)
            print(f"{'-':<10}{pos:<10}{'-':<15}{pred_val:<15.2f}")
        
        print("="*80)
    return

# 互動模式
def IM():
    while True:
        print("請輸入分子SMILES: (輸入q離開)")
        smiles = input()
        if smiles == "q":
            break
        df = pd.read_csv('../output/pka_mapping_results.csv')
        df = df[df['smiles'] == smiles]
        true_pka = df['pka_matrix']
        true_pka = true_pka.to_numpy()
        result = model.sample(smiles, device)
        print_result(smiles, result, true_pka)
    return

# 為parity plot格式保存數據
def save_for_parity_plot(results_data, output_file="../output/pka_prediction_results.csv"):
    """
    保存預測結果為適合繪製parity plot的CSV格式
    
    Args:
        results_data: 包含預測結果的列表，每個元素為一個字典
        output_file: 輸出CSV文件路徑
    """
    # 確保輸出目錄存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 打開CSV文件並寫入
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['molecule_name', 'smiles', 'atom_position', 'true_pka', 'predicted_pka', 'difference']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # 寫入每個預測結果
        for result in results_data:
            writer.writerow(result)
    
    print(f"Parity plot 數據已保存至: {output_file}")

# 為分子視覺化保存數據
def save_for_visualization(viz_data, output_file="../output/pka_visualization_data.csv"):
    """
    保存預測結果為適合分子視覺化的CSV格式
    
    Args:
        viz_data: 包含分子和pKa位置/值的列表
        output_file: 輸出CSV文件路徑
    """
    # 確保輸出目錄存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 打開CSV文件並寫入
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['molecule_name', 'smiles', 'pka_positions', 'pka_values', 'true_pka_positions', 'true_pka_values']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # 寫入每個分子的數據
        for molecule_data in viz_data:
            writer.writerow(molecule_data)
    
    print(f"分子視覺化數據已保存至: {output_file}")

# 遍歷整個database，輸出完整的預測結果
def export_all_results():
    df = pd.read_csv('../output/pka_mapping_results.csv')
    
    # 存儲parity plot數據的列表
    parity_results = []
    
    # 存儲分子視覺化數據的列表
    viz_results = []
    
    # 計數器
    total = len(df)
    processed = 0
    
    print(f"開始處理 {total} 個分子的預測結果...")
    
    for index, row in df.iterrows():
        smiles = row['smiles']
        molecule_name = f"Molecule_{index}"  # 使用索引作為分子名稱
        true_pka_str = row['pka_matrix']
        
        # 解析 true_pka 字符串為位置和值的對
        try:
            true_pka_list = ast.literal_eval(true_pka_str)
            true_pka_positions = [int(pos) for pos, _ in true_pka_list]
            true_pka_values = [float(val) for _, val in true_pka_list]
        except:
            true_pka_list = []
            true_pka_positions = []
            true_pka_values = []
        
        true_pka_dict = {int(pos): float(val) for pos, val in true_pka_list}
        
        # 獲取預測結果
        try:
            result = model.sample(smiles, device)
            
            if result:
                pred_positions = result['pka_positions']
                pred_values = result['pka_values']
                
                # 為parity plot收集數據 - 只使用匹配的位置
                for i, pos in enumerate(pred_positions):
                    pred_val = pred_values[i]
                    if pos in true_pka_dict:
                        true_val = true_pka_dict[pos]
                        parity_results.append({
                            'molecule_name': molecule_name,
                            'smiles': smiles,
                            'atom_position': pos,
                            'true_pka': true_val,
                            'predicted_pka': pred_val,
                            'difference': abs(pred_val - true_val)
                        })
                
                # 為分子視覺化收集數據
                viz_results.append({
                    'molecule_name': molecule_name,
                    'smiles': smiles,
                    'pka_positions': str(pred_positions),
                    'pka_values': str(pred_values),
                    'true_pka_positions': str(true_pka_positions),
                    'true_pka_values': str(true_pka_values)
                })
        except Exception as e:
            print(f"處理分子 {smiles} 時出錯: {e}")
        
        # 更新進度
        processed += 1
        if processed % 10 == 0:
            print(f"已處理: {processed}/{total} ({processed/total*100:.1f}%)")
    
    # 保存parity plot數據
    save_for_parity_plot(parity_results)
    
    # 保存分子視覺化數據
    save_for_visualization(viz_results)
    
    # 創建匯總的pKa預測結果檔案 (每個分子一行)
    summary_data = []
    molecules = {}
    
    # 按分子名稱分組
    for result in parity_results:
        mol_name = result['molecule_name']
        if mol_name not in molecules:
            molecules[mol_name] = {
                'molecule_name': mol_name,
                'smiles': result['smiles'],
                'true_pka_count': 0,
                'predicted_pka_count': 0,
                'matched_count': 0,
                'rmse': 0.0,
                'mae': 0.0
            }
    
    # 計算每個分子的統計數據
    for mol_name, mol_data in molecules.items():
        mol_results = [r for r in parity_results if r['molecule_name'] == mol_name]
        
        # 獲取這個分子的第一筆數據來獲取SMILES
        mol_smiles = mol_results[0]['smiles'] if mol_results else ""
        
        # 找回原始數據行
        original_row = df[df['smiles'] == mol_smiles]
        if not original_row.empty:
            true_pka_str = original_row.iloc[0]['pka_matrix']
            try:
                true_pka_list = ast.literal_eval(true_pka_str)
                true_count = len(true_pka_list)
            except:
                true_count = 0
        else:
            true_count = 0
        
        # 計算這個分子的預測pKa數量
        try:
            result = model.sample(mol_smiles, device)
            pred_count = len(result['pka_positions']) if result else 0
        except:
            pred_count = 0
        
        # 計算這個分子的統計資訊
        mol_data['true_pka_count'] = true_count
        mol_data['predicted_pka_count'] = pred_count
        mol_data['matched_count'] = len(mol_results)
        
        if len(mol_results) > 0:
            true_values = [r['true_pka'] for r in mol_results]
            pred_values = [r['predicted_pka'] for r in mol_results]
            
            squared_errors = [(t - p)**2 for t, p in zip(true_values, pred_values)]
            abs_errors = [abs(t - p) for t, p in zip(true_values, pred_values)]
            
            mol_data['rmse'] = np.sqrt(np.mean(squared_errors))
            mol_data['mae'] = np.mean(abs_errors)
        
        summary_data.append(mol_data)
    
    # 保存分子匯總數據
    summary_file = "../output/pka_prediction_summary.csv"
    pd.DataFrame(summary_data).to_csv(summary_file, index=False)
    print(f"分子匯總數據已保存至: {summary_file}")
    
    # 計算整體統計資料
    if parity_results:
        all_true = [r['true_pka'] for r in parity_results]
        all_pred = [r['predicted_pka'] for r in parity_results]
        
        rmse = np.sqrt(np.mean([(t - p)**2 for t, p in zip(all_true, all_pred)]))
        mae = np.mean([abs(t - p) for t, p in zip(all_true, all_pred)])
        
        print(f"\n整體預測效能:")
        print(f"總樣本數: {len(parity_results)}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
    
    return

# 遍歷整個database，把預測pka差距大於0.5的分子列出來
def check_database():
    df = pd.read_csv('../output/pka_mapping_results.csv')
    
    # 創建用於存儲錯誤結果的列表
    error_results = []
    
    for index, row in df.iterrows():
        smiles = row['smiles']
        true_pka_str = row['pka_matrix']
        
        # 解析 true_pka 字符串為位置和值的對
        true_pka_list = ast.literal_eval(true_pka_str)
        true_pka_dict = {int(pos): float(val) for pos, val in true_pka_list}
            
        # 獲取預測結果
        result = model.sample(smiles, device)
            
        # 比較預測值和真實值
        for i, pos in enumerate(result['pka_positions']):
            pred_val = result['pka_values'][i]
            if pos in true_pka_dict:
                true_val = true_pka_dict[pos]
                diff = abs(pred_val - true_val)
                if diff > 0.5:
                    error_results.append({
                        'smiles': smiles,
                        'position': pos,
                        'pred_pKa': round(pred_val, 2),
                        'true_pKa': round(true_val, 2),
                        'diff': round(diff, 2)
                    })
    
    # 將結果轉換為 DataFrame 並儲存
    if error_results:
        error_df = pd.DataFrame(error_results)
        error_df.to_csv('../output/pka_mapping_results_error.csv', index=False)
        print(f"已將差異大於0.5的預測結果寫入 ../output/pka_mapping_results_error.csv ，共 {len(error_results)} 筆資料")
    else:
        print("沒有發現差異大於0.5的預測結果")
    
    return
    
# 載入模型
node_dim = 153    # 節點特徵維度
bond_dim = 11     # 鍵特徵維度
hidden_dim = 153  # 隱藏層維度
output_dim = 1    # 輸出維度
dropout = 0.2     # Dropout率

# 初始化模型
model = pka_GNN(node_dim, bond_dim, hidden_dim, output_dim, dropout)

# 載入預訓練權重
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load('../results/pka_ver6/pka_ver6_best.pkl', map_location=device, weights_only=True)

# 處理模型結構不匹配問題
new_state_dict = {}
for k, v in checkpoint.items():
    # 將舊的rnn_gate映射到新的gate
    if k.startswith('rnn_gate'):
        new_key = k.replace('rnn_gate', 'gate')
        new_state_dict[new_key] = v
    # 忽略舊模型中的GCN1層
    elif k.startswith('GCN1'):
        continue
    else:
        new_state_dict[k] = v

# 檢查是否有缺失的關鍵權重
missing_keys = []
for k in model.state_dict().keys():
    if k not in new_state_dict:
        missing_keys.append(k)

if missing_keys:
    print(f"警告：以下權重未在預訓練模型中找到，將使用隨機初始化：{missing_keys}")

# 用兼容的方式載入權重
model.load_state_dict(new_state_dict, strict=False)
model = model.to(device)
model.eval()

# 將資料分割為訓練集和測試集，並分別進行採樣
def split_and_sample_dataset(model=None, device=None):
    # 如果沒有提供模型，使用全局模型
    if model is None:
        model = globals()['model']
    
    # 如果沒有提供設備，使用全局設備
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 確保模型在評估模式
    model.eval()
    
    # 從pka_mapping_results.csv載入原始資料
    df = pd.read_csv('../output/pka_mapping_results.csv')
    
    # 分割資料集為訓練集和測試集
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    print(f"總資料數量: {len(df)}")
    print(f"訓練集數量: {len(train_df)}")
    print(f"測試集數量: {len(test_df)}")
    
    # 將訓練集和測試集分別保存到CSV檔案
    train_df.to_csv('../output/pka_train_dataset.csv', index=False)
    test_df.to_csv('../output/pka_test_dataset.csv', index=False)
    
    # 針對訓練集和測試集分別進行採樣和預測
    
    # 用於存儲訓練集和測試集的預測結果
    train_parity_results = []
    test_parity_results = []
    
    train_viz_results = []
    test_viz_results = []
    
    # 處理訓練集
    print("正在處理訓練集...")
    process_dataset(train_df, train_parity_results, train_viz_results, "Train", model, device)
    
    # 處理測試集
    print("\n正在處理測試集...")
    process_dataset(test_df, test_parity_results, test_viz_results, "Test", model, device)
    
    # 保存訓練集和測試集的結果
    save_for_parity_plot(train_parity_results, "../output/pka_train_prediction_results.csv")
    save_for_parity_plot(test_parity_results, "../output/pka_test_prediction_results.csv")
    
    save_for_visualization(train_viz_results, "../output/pka_train_visualization_data.csv")
    save_for_visualization(test_viz_results, "../output/pka_test_visualization_data.csv")
    
    # 計算並打印訓練集和測試集的整體性能
    print_performance_metrics(train_parity_results, "訓練集")
    print_performance_metrics(test_parity_results, "測試集")
    
    return

# 處理資料集的輔助函數
def process_dataset(df, parity_results, viz_results, dataset_name, model=None, device=None):
    # 如果沒有提供模型，使用全局模型
    if model is None:
        model = globals()['model']
    
    # 如果沒有提供設備，使用全局設備
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    total = len(df)
    processed = 0
    
    for index, row in df.iterrows():
        smiles = row['smiles']
        molecule_name = f"{dataset_name}_Molecule_{index}"
        true_pka_str = row['pka_matrix']
        
        # 解析 true_pka 字符串為位置和值的對
        try:
            true_pka_list = ast.literal_eval(true_pka_str)
            true_pka_positions = [int(pos) for pos, _ in true_pka_list]
            true_pka_values = [float(val) for _, val in true_pka_list]
            true_pka_dict = {int(pos): float(val) for pos, val in true_pka_list}
        except:
            true_pka_list = []
            true_pka_positions = []
            true_pka_values = []
            true_pka_dict = {}
        
        # 獲取預測結果
        try:
            result = model.sample(smiles, device)
            
            if result:
                pred_positions = result['pka_positions']
                pred_values = result['pka_values']
                
                # 為parity plot收集數據 - 只使用匹配的位置
                for i, pos in enumerate(pred_positions):
                    pred_val = pred_values[i]
                    if pos in true_pka_dict:
                        true_val = true_pka_dict[pos]
                        parity_results.append({
                            'molecule_name': molecule_name,
                            'smiles': smiles,
                            'atom_position': pos,
                            'true_pka': true_val,
                            'predicted_pka': pred_val,
                            'difference': abs(pred_val - true_val)
                        })
                
                # 為分子視覺化收集數據
                viz_results.append({
                    'molecule_name': molecule_name,
                    'smiles': smiles,
                    'pka_positions': str(pred_positions),
                    'pka_values': str(pred_values),
                    'true_pka_positions': str(true_pka_positions),
                    'true_pka_values': str(true_pka_values)
                })
        except Exception as e:
            print(f"處理分子 {smiles} 時出錯: {e}")
        
        # 更新進度
        processed += 1
        if processed % 10 == 0:
            print(f"已處理: {processed}/{total} ({processed/total*100:.1f}%)")

# 打印性能指標的輔助函數
def print_performance_metrics(results, dataset_name):
    if results:
        all_true = [r['true_pka'] for r in results]
        all_pred = [r['predicted_pka'] for r in results]
        
        rmse = np.sqrt(np.mean([(t - p)**2 for t, p in zip(all_true, all_pred)]))
        mae = np.mean([abs(t - p) for t, p in zip(all_true, all_pred)])
        
        print(f"\n{dataset_name}預測效能:")
        print(f"總樣本數: {len(results)}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")

# 將所有分子的樣本預測結果保存到一個檔案中
def save_all_samples():
    df = pd.read_csv('../output/pka_mapping_results.csv')
    
    # 創建用於儲存所有樣本數據的列表
    all_samples = []
    
    # 計數器
    total = len(df)
    processed = 0
    
    print(f"開始處理 {total} 個分子的樣本預測...")
    
    for index, row in df.iterrows():
        smiles = row['smiles']
        molecule_name = f"Molecule_{index}"
        
        # 獲取預測結果
        try:
            result = model.sample(smiles, device)
            
            if result:
                pred_positions = result['pka_positions']
                pred_values = result['pka_values']
                
                # 將每個預測的pKa值添加到結果列表中
                for i, pos in enumerate(pred_positions):
                    pred_val = pred_values[i]
                    all_samples.append({
                        'molecule_id': index,
                        'molecule_name': molecule_name,
                        'smiles': smiles,
                        'atom_position': pos,
                        'predicted_pka': round(pred_val, 2)
                    })
        except Exception as e:
            print(f"處理分子 {smiles} 時出錯: {e}")
        
        # 更新進度
        processed += 1
        if processed % 10 == 0:
            print(f"已處理: {processed}/{total} ({processed/total*100:.1f}%)")
    
    # 將結果保存為CSV檔案
    output_file = "../output/all_pka_samples.csv"
    pd.DataFrame(all_samples).to_csv(output_file, index=False)
    print(f"所有樣本預測結果已保存至: {output_file}")
    print(f"總共預測了 {len(all_samples)} 個pKa值，來自 {processed} 個分子")
    
    return

if __name__ == "__main__":
    print("選擇操作模式:")
    print("1. 互動模式 (手動輸入SMILES)")
    print("2. 檢查資料庫中誤差大於0.5的預測")
    print("3. 導出完整預測結果 (適合畫圖)")
    print("4. 分割資料集並分別進行採樣")
    print("5. 將所有分子的樣本預測結果保存到一個檔案")
    choice = input("請輸入選項 (1/2/3/4/5): ")
    
    if choice == "1":
        IM()
    elif choice == "2":
        check_database()
    elif choice == "3":
        export_all_results()
    elif choice == "4":
        split_and_sample_dataset()
    elif choice == "5":
        save_all_samples()
    else:
        print("無效的選項!")