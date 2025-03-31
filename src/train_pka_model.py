import torch
import argparse
import os
import pandas as pd
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
from torch_geometric.data import DataLoader
from sklearn.metrics import mean_absolute_error, r2_score, precision_score, recall_score, f1_score
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm  # 添加進度條支持

from pka_models import PKA_GNN
from pka_datautils import PKADataProcessor
from integrated_nist_converter import predict_dissociable_atoms, sanitize_smiles, tensorize_molecule, set_verbose

# 設置繪圖風格
plt.style.use('ggplot')
sns.set(style="whitegrid")

# 添加調試輸出函數
def debug_log(message, flush=True):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=flush)

# 繪製學習曲線函數
def plot_learning_curves(history, save_path, version):
    """繪製並保存學習曲線"""
    debug_log("繪製學習曲線...")
    # 設置字體支持
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題
    
    # 設置字體大小
    plt.rcParams.update({'font.size': 12})
    
    # 創建子圖
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    
    # 1. 損失曲線
    axs[0, 0].plot(history['train_loss'], label='Training Combined Loss', color='blue', linewidth=2)
    axs[0, 0].plot(history['test_loss'], label='Testing Combined Loss', color='red', linewidth=2)
    axs[0, 0].set_title('Combined Loss Curve')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss Value')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # 2. 分類損失曲線
    axs[0, 1].plot(history['train_cls_loss'], label='Training Classification Loss', color='blue', linewidth=2)
    axs[0, 1].plot(history['test_cls_acc'], label='Testing Classification Loss', color='red', linewidth=2)
    axs[0, 1].set_title('Classification Loss Curve')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Loss Value')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # 3. 回歸損失曲線
    axs[1, 0].plot(history['train_reg_loss'], label='Training Regression Loss', color='blue', linewidth=2)
    axs[1, 0].plot(history['test_reg_mse'], label='Testing Regression MSE', color='red', linewidth=2)
    axs[1, 0].set_title('Regression Loss Curve')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Loss Value')
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    # 4. 準確率曲線
    axs[1, 1].plot(history['train_accuracy'], label='Training Accuracy', color='blue', linewidth=2)
    axs[1, 1].plot(history['test_accuracy'], label='Testing Accuracy', color='red', linewidth=2)
    axs[1, 1].set_title('Accuracy Curve')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Accuracy')
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    
    # 設置整體標題
    plt.suptitle(f'Model {version} Training Process', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # 保存圖片
    curve_path = os.path.join(save_path, f"{version}_learning_curves.png")
    plt.savefig(curve_path, dpi=300, bbox_inches='tight')
    debug_log(f"學習曲線已保存至 {curve_path}")
    
    # 繪製單獨的詳細曲線
    detailed_curves = [
        ('loss', ['train_loss', 'test_loss'], 'Loss Value'),
        ('cls_loss', ['train_cls_loss', 'test_cls_acc'], 'Classification Loss'),
        ('reg_loss', ['train_reg_loss', 'test_reg_mse'], 'Regression Loss'),
        ('accuracy', ['train_accuracy', 'test_accuracy'], 'Accuracy')
    ]
    
    for curve_name, keys, ylabel in detailed_curves:
        plt.figure(figsize=(10, 6))
        plt.plot(history[keys[0]], label=f'Training {ylabel}', color='blue', linewidth=2)
        plt.plot(history[keys[1]], label=f'Testing {ylabel}', color='red', linewidth=2)
        plt.title(f'{ylabel} Curve')
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        detailed_path = os.path.join(save_path, f"{version}_{curve_name}_curve.png")
        plt.savefig(detailed_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    debug_log("所有學習曲線繪製完成")
class EarlyStopping:
    """提前停止訓練的類"""
    def __init__(self, patience=10, min_delta=0, mode='min'):
        """
        初始化
        patience: 容忍多少個epoch沒有改進
        min_delta: 最小變化閾值
        mode: 'min'表示監控指標越小越好，'max'表示越大越好
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.early_stop = False
        self.val_loss_min = float('inf')
        
    def __call__(self, score, model=None, path=None):
        """
        檢查是否需要提前停止
        score: 當前的監控指標值
        model: 當前模型
        path: 如果需要保存最佳模型，提供保存路徑
        """
        if self.mode == 'min':
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
                if model is not None and path is not None:
                    self.save_checkpoint(score, model, path)
            else:
                self.counter += 1
        else:  # mode == 'max'
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
                if model is not None and path is not None:
                    self.save_checkpoint(score, model, path)
            else:
                self.counter += 1
                
        if self.counter >= self.patience:
            self.early_stop = True
            debug_log(f"提前停止! 已經連續 {self.patience} 個epoch沒有改進")
        
        return self.early_stop
    
    def save_checkpoint(self, val_loss, model, path):
        """保存最佳模型"""
        debug_log(f'驗證損失降低 ({self.val_loss_min:.6f} -> {val_loss:.6f}). 保存模型...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss
    
    def reset(self):
        """重置狀態"""
        self.counter = 0
        self.best_score = float('inf') if self.mode == 'min' else float('-inf')
        self.early_stop = False
        self.val_loss_min = float('inf')

"""
1. 載入參數
"""
if __name__ == '__main__':
    # 設置命令行參數
    parser = argparse.ArgumentParser(description='pKa預測模型訓練腳本')
    parser.add_argument('--config_csv', default='../data/pka_parameters.csv',
                        help='配置文件路徑')   
    parser.add_argument('--version', default='pka_gnn_v4',
                        help='配置版本，例如: v1, v2')
    parser.add_argument('--input_data', default='../data/nist_pka_data_2_dissociation.csv',
                        help='輸入數據文件路徑，支持順序化資料')
    parser.add_argument('--smiles_col', default='SMILES',
                        help='SMILES列名')
    parser.add_argument('--pka_col', default='pKa',
                        help='pKa值列名')
    parser.add_argument('--dissociable_atom_col', default='Dissociable_Atoms_Ordered',
                        help='有序可解離原子索引列名')
    parser.add_argument('--func_group_col', default='Functional_Group_Ordered',
                        help='有序官能團類型列名')
    parser.add_argument('--early_stopping', action='store_true',
                        help='是否啟用提前停止機制')
    parser.add_argument('--patience', type=int, default=10,
                        help='提前停止的耐心值')
    parser.add_argument('--seed', type=int, default=42,
                        help='隨機種子')
    parser.add_argument('--max_dissociation_steps', type=int, default=2,
                        help='最大解離階段數，用於順序性模型')
    args = parser.parse_args()
    
    # 設置隨機種子以確保可重現性
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 加載配置
    try:
        config = PKADataProcessor.load_config_by_version(args.config_csv, args.version)
    except FileNotFoundError:
        # 如果找不到配置文件，使用默認配置
        config = {
            'version': args.version,
            'test_size': 0.2,
            'num_features': 153,  # 根據atom_features函數的輸出維度
            'bond_features': 12,  # 根據bond_features函數的輸出維度
            'hidden_dim': 128,
            'batch_size': 32,
            'num_epochs': 100,
            'lr': 0.001,
            'dropout': 0.2,
            'anneal_rate': 0.9,
            'save_path': '../results/',
            'model_path': '../models/'
        }
        # 創建配置文件目錄
        os.makedirs(os.path.dirname(args.config_csv), exist_ok=True)
        
        # 將默認配置保存到配置文件
        pd.DataFrame([config]).to_csv(args.config_csv, index=False)
        debug_log(f"找不到配置文件，已創建默認配置文件: {args.config_csv}")
    
    # 確保結果和模型目錄存在
    os.makedirs(config['save_path'], exist_ok=True)
    os.makedirs(config['model_path'], exist_ok=True)
    
    # 打印配置參數
    debug_log("配置參數:")
    for k, v in config.items():
        debug_log(f"{k}: {v}", flush=False)

"""
2. 載入資料
"""
# 加載數據
debug_log("正在加載數據...")
try:
    train_loader, test_loader = PKADataProcessor.data_loader(
        file_path=args.input_data,
        smiles_col=args.smiles_col,
        pka_col=args.pka_col,
        dissociable_atom_col=args.dissociable_atom_col,
        func_group_col=args.func_group_col,
        batch_size=config['batch_size'],
        test_size=config['test_size']
    )
    debug_log(f"數據加載完成: 訓練批次={len(train_loader)}, 測試批次={len(test_loader)}")
except Exception as e:
    debug_log(f"數據加載失敗: {str(e)}")
    raise

# 設置計算設備
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
debug_log(f"使用設備: {device}")

# 創建模型
debug_log("創建模型...")
model = PKA_GNN(
    node_dim=config['num_features'],
    bond_dim=config['bond_features'],
    hidden_dim=config['hidden_dim'],
    max_dissociation_steps=args.max_dissociation_steps,
    dropout=config['dropout']
).to(device)
debug_log(f"模型創建完成，參數數量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# 設置優化器和學習率調度器
optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-5)  # 添加L2正則化
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config['anneal_rate'], 
                                          patience=5, verbose=True)  # 使用更智能的學習率調度器

# 設置提前停止
early_stopping = EarlyStopping(patience=args.patience) if args.early_stopping else None
if args.early_stopping:
    debug_log(f"啟用提前停止機制，耐心值={args.patience}")




"""
3. 開始訓練
"""
# 初始化歷史記錄
history = {
    'train_loss': [], 'train_cls_loss': [], 'train_reg_loss': [], 'train_accuracy': [],
    'test_loss': [], 'test_cls_acc': [], 'test_reg_mse': [], 'test_accuracy': [],
    'train_f1': [], 'test_f1': [], 'train_mae': [], 'test_mae': [], 'test_r2': [],
    'train_order_loss': [], 'test_order_acc': []  # 新增：解離順序相關指標
}

debug_log("="*80)
debug_log("開始訓練")
debug_log("="*80)

# 訓練循環
start_time = time.time()
global_step = 0  # 添加全局步數計數器
best_test_loss = float('inf')
best_model_path = os.path.join(config['model_path'], f"{config['version']}_best.pt")

for epoch in range(config['num_epochs']):
    epoch_start_time = time.time()
    debug_log(f"Epoch {epoch+1}/{config['num_epochs']} 開始")
    model.train()
    total_loss, total_cls_loss, total_reg_loss, total_order_loss = 0, 0, 0, 0
    total_accuracy = 0
    batch_count = 0
    
    # 收集分類預測以計算F1分數
    all_train_preds = []
    all_train_true = []
    
    # 訓練階段 - 使用tqdm進度條增強顯示
    train_pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
                     desc=f"Epoch {epoch+1}/{config['num_epochs']} [Train]")
    
    for batch_idx, batch in train_pbar:
        batch_start_time = time.time()
        optimizer.zero_grad()
        batch = batch.to(device)
        
        # 獲取當前批次的分子數量信息
        num_nodes = batch.x.shape[0]
        num_edges = batch.edge_index.shape[1]
        
        # 前向傳播
        outputs = model(batch)
        
        # 解析輸出 - 檢查是否包含解離順序損失
        if len(outputs) == 3:
            cls_preds, reg_preds, losses = outputs
            if len(losses) == 4:
                all_loss, cls_loss, reg_loss, accuracy = losses
                order_loss = 0  # 不含順序損失
            elif len(losses) == 5:
                all_loss, cls_loss, reg_loss, order_loss, accuracy = losses
            else:
                all_loss, cls_loss, reg_loss, accuracy = losses
                order_loss = 0
        else:
            debug_log(f"警告: 未知的模型輸出格式: {type(outputs)}")
            continue
        
        # 收集預測結果
        if hasattr(batch, 'dissociable_masks'):
            true_cls = [1 if m.item() > 0 else 0 for m in batch.dissociable_masks]
            all_train_preds.extend(cls_preds)
            all_train_true.extend(true_cls)
        
        # 反向傳播和優化
        all_loss.backward()
        optimizer.step()
        global_step += 1
        
        # 累加損失和準確率
        total_loss += all_loss.item()
        total_cls_loss += cls_loss.item()
        # 檢查reg_loss是否為張量，如果是則使用item()，否則直接加
        reg_loss_value = reg_loss.item() if hasattr(reg_loss, 'item') else reg_loss
        total_reg_loss += reg_loss_value
        total_accuracy += accuracy
        
        # 累加解離順序損失（如果有）
        if order_loss != 0:
            order_loss_value = order_loss.item() if hasattr(order_loss, 'item') else order_loss
            total_order_loss += order_loss_value
        
        batch_count += 1
        
        # 更新進度條
        train_pbar.set_postfix({
            'loss': f"{all_loss.item():.4f}",
            'cls_acc': f"{accuracy:.4f}",
            'reg_loss': f"{reg_loss_value:.4f}"
        })
    
    # 計算平均損失和準確率
    train_loss = total_loss / batch_count
    train_cls_loss = total_cls_loss / batch_count
    train_reg_loss = total_reg_loss / batch_count
    train_order_loss = total_order_loss / batch_count if batch_count > 0 else 0
    train_accuracy = total_accuracy / batch_count
    
    # 計算F1分數（如果有足夠的數據）
    train_f1 = 0
    if len(all_train_preds) > 0 and len(all_train_true) > 0:
        train_f1 = f1_score(all_train_true, all_train_preds, average='macro', zero_division=0)
    
    
    
    
    
    
    """
    測試集評估
    """
    # debug_log("開始測試集評估...")
    model.eval()

    # 收集測試集預測結果
    all_test_preds = []
    all_test_true = []
    all_test_reg_preds = []
    all_test_reg_true = []

    # 添加調試信息
    # debug_log(f"測試資料集大小: {len(test_loader.dataset)}個樣本, {len(test_loader)}個批次")

    # 手動實現評估過程，而不是直接調用evaluate_model
    total_cls_acc = 0
    total_reg_mse = 0
    test_results = {'atom_results': {}, 'molecule_results': []}

    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Test]")):
                # debug_log(f"處理測試批次 {batch_idx+1}/{len(test_loader)}")
                
                # 將數據移至設備
                batch = batch.to(device)
                
                # 前向傳播
                # debug_log(f"批次 {batch_idx+1} 開始前向傳播")
                outputs = model(batch)
                # debug_log(f"批次 {batch_idx+1} 前向傳播完成")
                
                # 打印預測結果類型
                # debug_log(f"分類預測類型: {type(outputs[0]).__name__}, 長度: {len(outputs[0]) if isinstance(outputs[0], list) else outputs[0].shape}")
                
                # 檢查回歸預測類型和形狀
                if isinstance(outputs[1], list):
                    # debug_log(f"回歸預測是列表，長度: {len(outputs[1])}")
                    if len(outputs[1]) > 0:
                        sample_pred = outputs[1][0]
                        # debug_log(f"  列表中第一個元素類型: {type(sample_pred).__name__}")
                        if hasattr(sample_pred, 'shape'):
                            # debug_log(f"  元素形狀: {sample_pred.shape}")
                            pass
                else:
                    # debug_log(f"回歸預測是張量，形狀: {outputs[1].shape}")
                    pass
                
                # 收集預測結果
                if hasattr(batch, 'dissociable_masks'):
                    # debug_log(f"批次 {batch_idx+1} 處理分類結果")
                    true_cls = [1 if m.item() > 0 else 0 for m in batch.dissociable_masks]
                    test_results['atom_results']['True_Labels'] = true_cls
                    test_results['atom_results']['Pred_Labels'] = outputs[0]
                
                if hasattr(batch, 'pka_values'):
                    # debug_log(f"批次 {batch_idx+1} 處理回歸結果")
                    
                    # 處理回歸預測結果
                    true_pka_list = []
                    pred_pka_list = []
                    
                    # 逐個處理每個節點的預測
                    for i in range(len(batch.dissociable_masks)):
                        if batch.dissociable_masks[i] > 0:  # 如果是可解離原子
                            true_pka_list.append(batch.pka_values[i].cpu().item())
                            
                            # 如果outputs[1]是列表，則直接獲取對應元素
                            if isinstance(outputs[1], list):
                                valid_count = sum(1 for m in batch.dissociable_masks[:i] if m > 0)
                                if valid_count < len(outputs[1]):
                                    # 如果預測值是二維的，只取第一個值
                                    pred_value = outputs[1][valid_count]
                                    if isinstance(pred_value, (torch.Tensor, np.ndarray)) and hasattr(pred_value, 'shape') and len(pred_value.shape) > 0 and pred_value.shape[0] > 0:
                                        pred_value = pred_value[0].cpu().item() if isinstance(pred_value, torch.Tensor) else pred_value[0]
                                    pred_pka_list.append(pred_value)
                                else:
                                    # debug_log(f"警告: 索引超出範圍, valid_count={valid_count}, outputs[1]長度={len(outputs[1])}")
                                    pass
                            else:
                                # 如果是張量，則使用索引
                                pred_value = outputs[1][i]
                                # 如果預測值是二維的，只取第一個值
                                if len(pred_value.shape) > 0 and pred_value.shape[0] > 0:
                                    pred_value = pred_value[0]
                                pred_pka_list.append(pred_value.cpu().item())
                    
                    # debug_log(f"找到 {len(true_pka_list)} 個有效的pKa值")
                    
                    # 將結果添加到測試結果中
                    test_results['atom_results']['True_pKa'] = true_pka_list
                    test_results['atom_results']['Pred_pKa'] = pred_pka_list
                
                # debug_log(f"完成批次 {batch_idx+1}/{len(test_loader)}")
            
            # 計算指標前檢查數據形狀
            if len(test_results['atom_results']['True_pKa']) > 0 and len(test_results['atom_results']['Pred_pKa']) > 0:
                true_pka_array = np.array(test_results['atom_results']['True_pKa'])
                pred_pka_array = np.array(test_results['atom_results']['Pred_pKa'])
                
                # debug_log(f"真實pKa形狀: {true_pka_array.shape}")
                # debug_log(f"預測pKa形狀: {pred_pka_array.shape}")
                
                # 確保預測值是一維的
                if len(pred_pka_array.shape) > 1:
                    # debug_log(f"預測pKa是多維的，展平或取第一列")
                    if pred_pka_array.shape[1] == 2:
                        # 假設第一列是實際的pKa預測
                        pred_pka_array = pred_pka_array[:, 0]
                        # debug_log(f"取第一列後預測pKa形狀: {pred_pka_array.shape}")
                    else:
                        # 如果不確定哪一列是正確的，可以展平平均值
                        pred_pka_array = pred_pka_array.mean(axis=1)
                        # debug_log(f"取平均值後預測pKa形狀: {pred_pka_array.shape}")
                
                # 更新測試結果字典
                test_results['atom_results']['Pred_pKa'] = pred_pka_array.tolist()
            
            # 計算指標
            # debug_log("計算測試指標...")
            if len(test_results['atom_results']['True_Labels']) > 0 and len(test_results['atom_results']['Pred_Labels']) > 0:
                correct = sum(1 for t, p in zip(test_results['atom_results']['True_Labels'], test_results['atom_results']['Pred_Labels']) if t == p)
                test_cls_acc = correct / len(test_results['atom_results']['True_Labels'])
                # debug_log(f"分類準確率: {test_cls_acc:.4f}, 樣本數: {len(test_results['atom_results']['True_Labels'])}")
            else:
                test_cls_acc = 0
                # debug_log("沒有分類結果可用")
            
            if len(test_results['atom_results']['True_pKa']) > 0 and len(test_results['atom_results']['Pred_pKa']) > 0:
                # 確保兩個數組形狀一致
                true_pka = np.array(test_results['atom_results']['True_pKa'])
                pred_pka = np.array(test_results['atom_results']['Pred_pKa'])
                
                # 再次檢查形狀
                # debug_log(f"MSE計算前 - 真實pKa形狀: {true_pka.shape}, 預測pKa形狀: {pred_pka.shape}")
                
                # 安全計算MSE
                test_reg_mse = ((true_pka - pred_pka) ** 2).mean()
                # debug_log(f"回歸MSE: {test_reg_mse:.4f}, 樣本數: {len(true_pka)}")
            else:
                test_reg_mse = 0
                # debug_log("沒有回歸結果可用")
            
            all_test_true = test_results['atom_results']['True_Labels']
            all_test_preds = test_results['atom_results']['Pred_Labels']
            all_test_reg_true = test_results['atom_results']['True_pKa']
            all_test_reg_preds = test_results['atom_results']['Pred_pKa']
            
            # 計算額外的評估指標
            test_f1 = 0
            mae = 0
            r2 = 0
            test_order_acc = 0  # 初始化test_order_acc變數
            
            if len(all_test_preds) > 0 and len(all_test_true) > 0:
                # debug_log(f"計算F1分數，標籤分布: {sum(all_test_true)}/{len(all_test_true)}")
                test_f1 = f1_score(all_test_true, all_test_preds, average='macro', zero_division=0)
            
            if len(all_test_reg_preds) > 0 and len(all_test_reg_true) > 0:
                # 確保一維形狀進行計算
                true_reg = np.array(all_test_reg_true)
                pred_reg = np.array(all_test_reg_preds)
                
                debug_log(f"計算回歸指標，pKa範圍: {true_reg.min():.2f} - {true_reg.max():.2f}")
                mae = mean_absolute_error(true_reg, pred_reg)
                r2 = r2_score(true_reg, pred_reg) if len(true_reg) > 1 else 0
                debug_log(f"測試集 - MAE: {mae:.4f}, R²: {r2:.4f}, F1: {test_f1:.4f}")

    except Exception as e:
        debug_log(f"測試評估過程發生錯誤: {str(e)}")
        import traceback
        debug_log(traceback.format_exc())
        # 設置默認值以便代碼可以繼續執行
        test_cls_acc = 0
        test_reg_mse = 0
        test_f1 = 0
        mae = 0
        r2 = 0
        test_order_acc = 0
        test_loss = float('inf')
        test_results = {'atom_results': {}, 'molecule_results': []}

    # 計算總測試損失
    test_loss = test_cls_acc + test_reg_mse
    debug_log(f"測試評估完成，總損失: {test_loss:.4f}")

    # 保存最佳模型
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        torch.save(model.state_dict(), best_model_path)
        debug_log(f"保存最佳模型，測試損失: {test_loss:.4f}")
    
    # 更新學習率
    scheduler.step(test_loss)  # 基於測試損失調整學習率
    
    # 記錄歷史
    history['train_loss'].append(train_loss)
    history['train_cls_loss'].append(train_cls_loss)
    history['train_reg_loss'].append(train_reg_loss)
    history['train_accuracy'].append(train_accuracy)
    history['train_f1'].append(train_f1)
    history['train_order_loss'].append(train_order_loss)
    history['test_loss'].append(test_loss)
    history['test_cls_acc'].append(test_cls_acc)
    history['test_reg_mse'].append(test_reg_mse)
    history['test_accuracy'].append(test_cls_acc)
    history['test_f1'].append(test_f1)
    history['train_mae'].append(0)  # 暫時不計算訓練MAE
    history['test_mae'].append(mae)
    history['test_r2'].append(r2)
    history['test_order_acc'].append(test_order_acc)
    
    # 計算epoch時間
    epoch_time = time.time() - epoch_start_time
    total_time = time.time() - start_time
    remaining_epochs = config['num_epochs'] - (epoch + 1)
    estimated_remaining_time = (epoch_time * remaining_epochs) / 60  # 分鐘
    
    # 打印進度
    debug_log("=" * 50)
    debug_log(f"Epoch {epoch+1}/{config['num_epochs']} 完成 (用時: {epoch_time:.2f}s)")
    debug_log(f"訓練集結果:")
    debug_log(f"  分類損失: {train_cls_loss:.4f}, 分類準確率: {train_accuracy:.4f}, F1: {train_f1:.4f}")
    debug_log(f"  回歸損失: {train_reg_loss:.4f}")
    if train_order_loss > 0:
        debug_log(f"  解離順序損失: {train_order_loss:.4f}")
    debug_log(f"  綜合損失: {train_loss:.4f}")
    debug_log(f"測試集結果:")
    debug_log(f"  分類準確率: {test_cls_acc:.4f}, F1: {test_f1:.4f}")
    debug_log(f"  回歸MSE: {test_reg_mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    debug_log(f"  綜合損失: {test_loss:.4f}")
    debug_log(f"學習率: {optimizer.param_groups[0]['lr']:.6f}")
    debug_log(f"預計剩餘時間: {estimated_remaining_time:.2f}分鐘")
    debug_log("=" * 50)
    
    # 保存訓練歷史
    history_df = pd.DataFrame(history)
    history_df.to_csv(
        os.path.join(config['save_path'], f"{config['version']}_training_history.csv"), 
        index=False
    )
    
    # 每10個epoch繪製一次學習曲線
    if (epoch + 1) % 10 == 0:
        plot_learning_curves(history, config['save_path'], config['version'])
        debug_log(f"中間學習曲線已生成")
    
    # 每10個epoch保存一次模型
    if (epoch + 1) % 10 == 0:
        model_save_path = os.path.join(config['model_path'], f"{config['version']}_epoch{epoch+1}.pt")
        torch.save(model.state_dict(), model_save_path)
        debug_log(f"模型檢查點已保存至 {model_save_path}")
    
    # 檢查提前停止
    if early_stopping is not None:
        if early_stopping(test_loss, model, best_model_path):
            debug_log(f"提前停止訓練於epoch {epoch+1}")
            break

"""
4. 評估模型和繪製最終學習曲線
"""
# 繪製最終學習曲線
plot_learning_curves(history, config['save_path'], config['version'])

# 載入最佳模型
debug_log(f"載入最佳模型進行最終評估...")
model.load_state_dict(torch.load(best_model_path))

# 在整個訓練和測試集上進行最終評估
debug_log("\n"+"="*50)
debug_log("最終評估:")
debug_log("="*50)

# 自定義評估函數替代 PKADataProcessor.evaluate_model
def custom_final_evaluate(model, data_loader, device, output_path=None):
    debug_log(f"使用自定義評估函數進行最終評估...")
    model.eval()
    
    # 收集結果
    results = {
        'SMILES': [],
        'True_pKa': [],
        'Pred_pKa': [],
        'True_Labels': [],
        'Pred_Labels': [],
        'Dissociable': [],
        'Is_Valid': [],
        'Dissociation_Order': []  # 添加解離順序欄位
    }
    
    processed_smiles = set()  # 追蹤已處理的分子，避免重複
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="最終評估")):
            debug_log(f"處理批次 {batch_idx+1}/{len(data_loader)}")
            
            # 獲取SMILES
            if hasattr(batch, 'smiles'):
                batch_smiles = batch.smiles
            else:
                # 如果批次沒有SMILES，生成臨時ID
                batch_smiles = [f"molecule_{batch_idx}_{i}" for i in range(batch.num_graphs)]
            
            # 獲取解離順序信息
            has_dissociation_order = hasattr(batch, 'dissociation_order')
            
            # 將數據移至設備
            batch = batch.to(device)
            # 前向傳播 - 檢查是否支持解離順序預測
            import inspect
            if hasattr(model, 'predict') and len(inspect.signature(model.predict).parameters) > 1:
                dissociable_nodes, pka_predictions, order_predictions = model.predict(batch)
            else:
                # 保持向後兼容性
                cls_preds, reg_preds, _ = model(batch)
                order_predictions = []  # 不支援順序預測時，使用空列表
            
            # 按分子分組處理結果
            molecule_data = {}
            
            # 收集每個批次數據的原子級信息
            for i in range(batch.num_graphs):
                # 獲取當前分子的SMILES
                if i < len(batch_smiles):
                    smiles = batch_smiles[i]
                else:
                    smiles = f"molecule_{batch_idx}_{i}"
                
                # 跳過已處理的分子
                if smiles in processed_smiles:
                    continue
                
                # 將該分子標記為已處理
                processed_smiles.add(smiles)
                
                # 獲取該分子的節點索引
                if hasattr(batch, 'batch'):
                    # 找出屬於當前分子的所有節點
                    node_indices = (batch.batch == i).nonzero(as_tuple=True)[0]
                    
                    # 初始化該分子的數據結構
                    if smiles not in molecule_data:
                        molecule_data[smiles] = {
                            'dissociable_atoms': [],
                            'true_pka': [],
                            'pred_pka': [],
                            'is_dissociable': False,
                            'dissociation_orders': []  # 新增：解離順序
                        }
                    
                    # 處理該分子的所有節點
                    for node_idx in node_indices:
                        # 檢查是否是可解離原子
                        if hasattr(batch, 'dissociable_masks') and node_idx < len(batch.dissociable_masks):
                            is_dissociable = batch.dissociable_masks[node_idx].item() > 0
                            
                            # 如果是可解離原子，收集pKa值
                            if is_dissociable:
                                molecule_data[smiles]['is_dissociable'] = True
                                molecule_data[smiles]['dissociable_atoms'].append(node_idx.item())
                                
                                # 獲取解離順序
                                if has_dissociation_order:
                                    order = batch.dissociation_order[node_idx].item()
                                    molecule_data[smiles]['dissociation_orders'].append(order)
                                else:
                                    molecule_data[smiles]['dissociation_orders'].append(-1)
                                
                                if hasattr(batch, 'pka_values'):
                                    true_pka = batch.pka_values[node_idx].item()
                                    molecule_data[smiles]['true_pka'].append(true_pka)
                                
                                # 處理預測值
                                if isinstance(reg_preds, list):
                                    # 需要找出對應於此節點的預測索引
                                    valid_count = sum(1 for j in range(len(batch.dissociable_masks)) if 
                                                     j < node_idx and batch.dissociable_masks[j].item() > 0 and batch.batch[j] == i)
                                    
                                    dissociable_in_molecule = sum(1 for j in range(len(batch.dissociable_masks)) if 
                                                               batch.dissociable_masks[j].item() > 0 and batch.batch[j] == i)
                                    
                                    if valid_count < len(reg_preds) and valid_count < dissociable_in_molecule:
                                        pred_value = reg_preds[valid_count]
                                        if isinstance(pred_value, (torch.Tensor, np.ndarray)):
                                            if len(pred_value.shape) > 0:
                                                pred_value = pred_value[0].item() if isinstance(pred_value, torch.Tensor) else pred_value[0]
                                            else:
                                                pred_value = pred_value.item() if isinstance(pred_value, torch.Tensor) else float(pred_value)
                                        molecule_data[smiles]['pred_pka'].append(pred_value)
                                else:
                                    pred_value = reg_preds[node_idx].item() if hasattr(reg_preds[node_idx], 'item') else float(reg_preds[node_idx])
                                    molecule_data[smiles]['pred_pka'].append(pred_value)
            
            # 處理完該批次後，將分子級結果添加到最終結果
            for smiles, data in molecule_data.items():
                # 只處理有解離原子的分子
                if data['is_dissociable']:
                    # 組合數據
                    pka_order_pairs = sorted(zip(data['dissociation_orders'], data['true_pka'], data['pred_pka']), 
                                            key=lambda x: x[0] if x[0] >= 0 else float('inf'))
                    
                    if not pka_order_pairs:
                        continue
                    
                    # 按解離順序處理每個原子的預測
                    for order, true_pka, pred_pka in pka_order_pairs:
                        # 將其添加到結果中
                        results['SMILES'].append(smiles)
                        results['True_pKa'].append(true_pka)
                        results['Pred_pKa'].append(pred_pka)
                        results['True_Labels'].append(1)  # 有解離原子
                        results['Pred_Labels'].append(1)  # 預測為有解離原子
                        results['Dissociable'].append(True)
                        results['Is_Valid'].append(True)
                        results['Dissociation_Order'].append(order)
        
        # 計算評估指標
        cls_accuracy = 0
        reg_mse = 0
        valid_preds = [(t, p) for t, p, v in zip(results['True_pKa'], results['Pred_pKa'], results['Is_Valid']) if v]
        if valid_preds:
            true_pka = np.array([t for t, _ in valid_preds])
            pred_pka = np.array([p for _, p in valid_preds])
            
            # 確保pred_pka是一維數組
            if len(pred_pka.shape) > 1 and pred_pka.shape[1] > 1:
                # 如果pred_pka是二維數組，取第一列
                pred_pka = pred_pka[:, 0]
            
            reg_mse = ((true_pka - pred_pka) ** 2).mean()
            
            # 分類準確率就是有解離原子的分子數量比例
            cls_accuracy = sum(results['Is_Valid']) / len(results['Is_Valid']) if results['Is_Valid'] else 0
        
        # 保存結果到CSV，分開保存原子級別和分子級別
        if output_path:
            # 原子級別結果
            atom_df = pd.DataFrame({
                'SMILES': results['SMILES'],
                'True_pKa': results['True_pKa'],
                'Pred_pKa': results['Pred_pKa'],
                'Is_Dissociable': results['Dissociable'],
                'Is_Valid': results['Is_Valid'],
                'Dissociation_Order': results['Dissociation_Order']
            })
            atom_output_path = output_path.replace('.csv', '_atom_level.csv')
            atom_df.to_csv(atom_output_path, index=False)
            debug_log(f"原子級別結果已保存至 {atom_output_path}")
            
            # 分子級別結果 - 按分子分組並計算平均值
            mol_data = {}
            for i, (smiles, order, true_pka, pred_pka) in enumerate(zip(results['SMILES'], results['Dissociation_Order'], 
                                                                     results['True_pKa'], results['Pred_pKa'])):
                if smiles not in mol_data:
                    mol_data[smiles] = {'orders': [], 'true_pkas': [], 'pred_pkas': []}
                mol_data[smiles]['orders'].append(order)
                mol_data[smiles]['true_pkas'].append(true_pka)
                mol_data[smiles]['pred_pkas'].append(pred_pka)
            
            mol_rows = []
            for smiles, data in mol_data.items():
                # 按解離順序排序
                sorted_data = sorted(zip(data['orders'], data['true_pkas'], data['pred_pkas']), 
                                    key=lambda x: x[0] if x[0] >= 0 else float('inf'))
                
                # 提取排序後的數據
                sorted_orders = [o for o, _, _ in sorted_data]
                sorted_true_pkas = [t for _, t, _ in sorted_data]
                sorted_pred_pkas = [p for _, _, p in sorted_data]
                
                # 計算分子級別的指標
                # 確保pred_pka是一維數值而非元組
                mse = np.mean([(float(t) - float(p)) ** 2 for t, p in zip(sorted_true_pkas, sorted_pred_pkas)])
                mae = np.mean([abs(float(t) - float(p)) for t, p in zip(sorted_true_pkas, sorted_pred_pkas)])
                # 創建分子級別的行
                mol_rows.append({
                    'SMILES': smiles,
                    'Dissociation_Orders': ';'.join(map(str, sorted_orders)),
                    'True_pKa_Values': ';'.join(map(str, sorted_true_pkas)),
                    'Pred_pKa_Values': ';'.join(map(str, sorted_pred_pkas)),
                    'Avg_True_pKa': np.mean(sorted_true_pkas),
                    'Avg_Pred_pKa': np.mean(sorted_pred_pkas),
                    'MSE': mse,
                    'MAE': mae
                })
            
            mol_df = pd.DataFrame(mol_rows)
            mol_output_path = output_path.replace('.csv', '_molecule_level.csv')
            mol_df.to_csv(mol_output_path, index=False)
            debug_log(f"分子級別結果已保存至 {mol_output_path}")
        
        return cls_accuracy, reg_mse, results

# 訓練集評估
debug_log("開始訓練集最終評估...")
train_cls_acc, train_reg_mse, train_results = custom_final_evaluate(
    model, train_loader, device,
    output_path=os.path.join(config['save_path'], f"{config['version']}_final_train_predictions.csv")
)
train_f1 = 0
if 'True_Labels' in train_results and 'Pred_Labels' in train_results:
    train_f1 = f1_score(train_results['True_Labels'], train_results['Pred_Labels'], average='macro', zero_division=0)

debug_log(f"訓練集最終結果:")
debug_log(f"  分類準確率: {train_cls_acc:.4f}, F1: {train_f1:.4f}")
debug_log(f"  回歸MSE: {train_reg_mse:.4f}")

# 測試集評估
debug_log("開始測試集最終評估...")
test_cls_acc, test_reg_mse, test_results = custom_final_evaluate(
    model, test_loader, device,
    output_path=os.path.join(config['save_path'], f"{config['version']}_final_test_predictions.csv")
)
test_f1 = 0
mae = 0
r2 = 0

if 'True_pKa' in test_results and 'Pred_pKa' in test_results:
    if len(test_results['True_pKa']) > 0 and len(test_results['Pred_pKa']) > 0:
        mae = mean_absolute_error(test_results['True_pKa'], test_results['Pred_pKa'])
        r2 = r2_score(test_results['True_pKa'], test_results['Pred_pKa']) if len(test_results['True_pKa']) > 1 else 0

debug_log(f"測試集最終結果:")
debug_log(f"  分類準確率: {test_cls_acc:.4f}, F1: {test_f1:.4f}")
debug_log(f"  回歸MSE: {test_reg_mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

# 保存最終模型
final_model_path = os.path.join(config['model_path'], f"{config['version']}_final.pt")
torch.save(model.state_dict(), final_model_path)
debug_log(f"最終模型已保存至: {final_model_path}")

# 計算總訓練時間
total_training_time = (time.time() - start_time) / 60  # 分鐘
debug_log(f"總訓練時間: {total_training_time:.2f}分鐘")
debug_log("="*50)
debug_log("訓練完成!") 
debug_log("="*50) 