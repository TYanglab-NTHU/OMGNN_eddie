import torch 
import torch.nn as nn
import logging
import os
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import MessagePassing, GCNConv,  Linear, BatchNorm, GlobalAttention, GATConv
from torch.autograd import set_detect_anomaly
import math

# 配置logging，設置文件處理器
def setup_logger():
    """設置日誌記錄器"""
    # 確保logs目錄存在
    os.makedirs("logs", exist_ok=True)
    
    # 配置日誌記錄器
    logger = logging.getLogger('pka_model')
    logger.setLevel(logging.DEBUG)
    
    # 創建文件處理器
    file_handler = logging.FileHandler('logs/pka_model_debug.log')
    file_handler.setLevel(logging.DEBUG)
    
    # 創建控制台處理器，僅用於ERROR級別的消息
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR)
    
    # 創建格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加處理器到記錄器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# 初始化logger
logger = setup_logger()

# 輔助函數：啟用autograd異常檢測
def enable_anomaly_detection():
    """啟用PyTorch梯度異常檢測，幫助定位就地操作問題"""
    set_detect_anomaly(True)
    logger.info("已啟用PyTorch梯度異常檢測，這會減慢訓練速度但能幫助定位問題")

class BondMessagePassing(nn.Module):
    def __init__(self, node_features, bond_features, hidden_size, depth=5, dropout=0.3):
        super(BondMessagePassing, self).__init__()
        self.W_i  = nn.Linear(node_features + bond_features, hidden_size)
        self.W_h  = nn.Linear(hidden_size, hidden_size)
        self.W_o  = nn.Linear(node_features + hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.depth = depth
        
        # 使用更好的權重初始化方法
        self._init_weights()
    
    def _init_weights(self):
        """使用適當的初始化方法來防止梯度消失/爆炸"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用Kaiming初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def update(self, M_t, H_0):
        # 檢查並處理NaN和Inf值
        if torch.isnan(M_t).any() or torch.isinf(M_t).any():
            M_t = torch.nan_to_num(M_t, nan=0.0, posinf=1e3, neginf=-1e3)
            
        H_t = self.W_h(M_t)
        # 使用新變數而非就地修改
        H_t = self.relu(H_0 + H_t)  # 使用剩餘連接
        H_t = self.dropout(H_t)

        return H_t

    def message(self, H, edge_index, rev_edge_index):
        """消息傳遞函數，現在支持直接接收邊索引和反向邊索引"""
        # 確保所有張量在同一設備上
        device = H.device
        edge_index = edge_index.to(device)
        rev_edge_index = rev_edge_index.to(device)
        
        # 檢查並處理NaN和Inf值
        if torch.isnan(H).any() or torch.isinf(H).any():
            H = torch.nan_to_num(H, nan=0.0, posinf=1e3, neginf=-1e3)
        
        # 確保index_torch在正確的設備上
        index_torch = edge_index[1].unsqueeze(1).repeat(1, H.shape[1]).to(device)
        
        # A: 使用更安全的方式計算消息
        try:
            # 1. 創建空的消息張量，並確保它在正確的設備上
            M_all = torch.zeros(H.shape[0], H.shape[1], dtype=H.dtype, device=device)
            
            # 2. 使用scatter_add_聚合消息，注意避免就地操作
            src_features = H[edge_index[0]]  # 源節點特徵
            # 創建新的張量而非使用就地操作
            M_all_accumulated = torch.zeros_like(M_all)
            M_all_accumulated.scatter_add_(0, index_torch, src_features)
            M_all = M_all_accumulated[edge_index[0]]  # 獲取特定位置的特徵
            
            # 3. 處理反向邊，首先創建一個與H相同形狀的零張量
            M_rev = torch.zeros_like(H[edge_index[0]], device=device)
            
            # 4. 只對有效的反向邊索引(非-1)進行查詢
            valid_rev_mask = rev_edge_index != -1
            if valid_rev_mask.any():
                valid_indices = rev_edge_index[valid_rev_mask]
                # 創建臨時張量來保存查詢結果
                temp_features = H[valid_indices]
                # 使用索引賦值而非就地修改
                new_M_rev = M_rev.clone()
                new_M_rev[valid_rev_mask] = temp_features
                M_rev = new_M_rev
            
            # 5. 返回差異作為最終消息
            return M_all - M_rev
            
        except Exception as e:
            print(f"消息傳遞出錯: {e}")
            # 返回一個形狀正確且在同一設備上的零張量
            return torch.zeros_like(H[edge_index[0]], device=device)

    def forward(self, x, edge_index, edge_attr, rev_edge_index):
        """修改接口以支持更靈活的輸入方式"""
        try:
            # 確保所有輸入張量在同一設備上
            device = x.device
            edge_index = edge_index.to(device)
            edge_attr = edge_attr.to(device)
            rev_edge_index = rev_edge_index.to(device)
            
            # 確保線性層也在相同設備上
            self.W_i = self.W_i.to(device)
            self.W_h = self.W_h.to(device)
            self.W_o = self.W_o.to(device)
            
            # 檢查並處理NaN和Inf值
            if torch.isnan(x).any() or torch.isinf(x).any():
                x = torch.nan_to_num(x, nan=0.0, posinf=1e3, neginf=-1e3)
            if torch.isnan(edge_attr).any() or torch.isinf(edge_attr).any():
                edge_attr = torch.nan_to_num(edge_attr, nan=0.0, posinf=1e3, neginf=-1e3)
            
            # 結合節點和邊特徵
            combined = torch.cat([x[edge_index[0]], edge_attr], dim=1)
            H_0 = self.W_i(combined)
            H   = self.relu(H_0)
            
            # 多層消息傳遞
            current_H = H
            for _ in range(1, self.depth):
                M = self.message(current_H, edge_index, rev_edge_index)
                current_H = self.update(M, H_0)
            
            # 使用最終的H來計算
            H = current_H
            
            # 使用更安全的方式聚合最終結果
            index_torch = edge_index[1].unsqueeze(1).repeat(1, H.shape[1]).to(device)
            M_temp = torch.zeros(x.shape[0], H.shape[1], dtype=H.dtype, device=device)
            
            # 使用scatter_add_聚合消息，避免就地操作
            M_accumulated = torch.zeros_like(M_temp)
            M_accumulated.scatter_add_(0, index_torch, H)
            M = M_accumulated
            
            # 處理孤立節點，避免就地操作
            isolated_mask = M.sum(dim=1, keepdim=True) == 0
            if isolated_mask.any():
                M_new = torch.where(isolated_mask, x, M)
                M = M_new
            
            # 結合原始節點特徵和聚合消息
            final_features = torch.cat([x, M], dim=1)
            
            # 最終變換
            H_out = self.W_o(final_features)
            H_out = self.relu(H_out)    
            H_out = self.dropout(H_out)
            
            # 確保輸出不包含NaN或Inf
            if torch.isnan(H_out).any() or torch.isinf(H_out).any():
                print("警告: BondMessagePassing輸出包含NaN或Inf值，已替換為有效值")
                H_out = torch.nan_to_num(H_out, nan=0.0, posinf=1e3, neginf=-1e3)
                
            return H_out
        except Exception as e:
            print(f"BondMessagePassing前向傳播出錯: {e}")
            # 返回一個安全的回退值，確保在正確的設備上
            return torch.zeros(x.shape[0], self.W_o.out_features, device=device)

class Squeeze1D(nn.Module):
    """自定義層，確保張量的最後一個維度為1時被壓縮"""
    def forward(self, x):
        # 如果最後一個維度為1，則移除該維度
        if x.size(-1) == 1:
            return x.squeeze(-1)
        return x

class pka_GNN(nn.Module):
    def __init__(self, node_dim, bond_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()

        # --- (1) 3 個固定 GCN ---
        
        DEPTH = 4
        
        self.gcn1 = BondMessagePassing(node_dim, bond_dim, hidden_dim,
                                       depth=DEPTH, dropout=dropout)
        # self.gcn2 = BondMessagePassing(node_dim, bond_dim, hidden_dim,
        #                                depth=DEPTH, dropout=dropout)
        self.gcn3 = BondMessagePassing(node_dim, bond_dim, hidden_dim,
                                       depth=DEPTH, dropout=dropout)

        # --- (2) gate ---
        self.gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())

        # --- (3) classifier / regressor ---
        self.atom_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 2)
        )
        self.atom_regressor = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, output_dim),  # scalar
            Squeeze1D()
        )

        # 損失
        self.criterion_reg = nn.SmoothL1Loss()
        # pKa 標準化常數 (可由 set_pka_normalization 更新)
        self.register_buffer('pka_mean', torch.tensor([7.0]))
        self.register_buffer('pka_std',  torch.tensor([3.0]))
        
    @staticmethod
    def _rev_edge_index(edge_index):
        device = edge_index.device
        edge_to_index = {(edge_index[0, i].item(), edge_index[1, i].item()): i 
                         for i in range(edge_index.shape[1])}
        rev_edge_index = torch.full((edge_index.shape[1],), -1, dtype=torch.long, device=device)
        for i in range(edge_index.shape[1]):
            u, v = edge_index[0, i].item(), edge_index[1, i].item()
            if (v, u) in edge_to_index:
                rev_edge_index[i] = edge_to_index[(v, u)]
        return rev_edge_index

    def forward(self, batch):
        """
        固定第一層 (gcn1)：
        – 只計算一次 gcn1 得到 h_static
        – 再跑一次 gcn3 取得初始 h_cur
        Teacher-forcing 逐真實 pKa 原子：
        – 分類 / 回歸 → 累積 loss
        – gate 更新該原子特徵
        – 將其餘原子重置成 h_static（= gcn1(x)）
        – gcn3 再傳遞一次
        最後回傳 logits 與 raw pKa
        """
        device = batch.x.device
        x, ei, ea = batch.x, batch.edge_index, batch.edge_attr
        rev_ei = self._rev_edge_index(ei)

        # ---------- (1) 固定「一層」GCN ---------- #
        # 只做 gcn1 一次，並把結果存成 h_static 供後續重置使用
        h_static = self.gcn1(x, ei, ea, rev_ei)              # [N, H]
        h_cur    = self.gcn3(h_static, ei, ea, rev_ei)        # 初始 h3

        # ---------- 取出 ground-truth pKa 原子順序 ----------
        gt_mask = (batch.pka_labels > 0)
        idx_gt  = torch.nonzero(gt_mask).squeeze(1)           # [K]
        
        if idx_gt.numel() == 0:                               # 無可解離原子
            logits  = self.atom_classifier(h_cur)
            pka_raw = self.atom_regressor(h_cur).view(-1)
            loss_cla = nn.functional.cross_entropy(
                logits, torch.zeros_like(gt_mask, dtype=torch.long)
            )
            return logits, pka_raw, (0.5 * loss_cla, loss_cla,
                                    torch.tensor(0., device=device))

        # 依真實 pKa 值排序
        idx_sorted = idx_gt[torch.argsort(batch.pka_labels[idx_gt])]

        # ---------- 逐-site loop ----------
        loss_cla_steps, loss_reg_steps = [], []
        for idx in idx_sorted:
            # 1) 分類 / 回歸
            logits = self.atom_classifier(h_cur)

            target = torch.zeros_like(gt_mask, dtype=torch.long)
            target[idx] = 1

            ratio  = float((target == 0).sum()) / (target.sum() + 1e-6)
            loss_c = nn.functional.cross_entropy(
                logits, target, weight=torch.tensor([1.0, ratio], device=device), reduction='none'
            )
            loss_cla_steps.extend(loss_c)

            pred_pka_norm  = (self.atom_regressor(h_cur)[idx] - self.pka_mean) / self.pka_std
            true_pka_norm  = (batch.pka_labels[idx] - self.pka_mean) / self.pka_std
            loss_r = self.criterion_reg(pred_pka_norm, true_pka_norm)
            loss_reg_steps.append(loss_r)

            # 2) gate 更新該原子特徵並重置其他原子 ---------------------- #
            h_upd = h_static.clone()                           # (2) 其餘原子 → h_static
            h_upd[idx] = h_cur[idx] * self.gate(h_cur[idx]) + h_cur[idx]

            # 3) 重新跑一次 gcn3
            h_cur = self.gcn3(h_upd, ei, ea, rev_ei)
            
        # ---------- 匯總損失 ----------
        loss_cla = torch.stack(loss_cla_steps).mean()
        loss_reg = torch.stack(loss_reg_steps).mean()
        total    = loss_cla + loss_reg

        # ---------- 最終輸出 ----------
        final_logits = logits                # 來自最後一輪 h_cur
        final_pka    = self.atom_regressor(h_cur).view(-1)  # 輸出原始預測值，未反標準化

        return final_logits, final_pka, (total, loss_cla, loss_reg)

        
    def update_epoch(self, current_epoch, max_epochs=None):
        """保留此方法以維持向後兼容性，但在新模型中不再使用"""
        pass
        
    def set_pka_normalization(self, mean, std):
        """
        設置 pKa 標準化參數，應使用全數據集的統計值
        
        Args:
            mean: 全數據集 pKa 值的均值
            std: 全數據集 pKa 值的標準差
        """
        # 確保這些值是浮點數
        mean_val = float(mean)
        std_val = max(float(std), 1e-6)  # 確保標準差不為零
        
        # 更新模型的標準化參數
        self.pka_mean[0] = mean_val
        self.pka_std[0] = std_val
        
        logger.info(f"設置 pKa 標準化參數: 均值={mean_val:.4f}, 標準差={std_val:.4f}")
        return mean_val, std_val

    def sample(self, smiles: str, device=None, eval_mode="predicted"):
        """
        eval_mode = 'predicted' → 只回傳 model 判斷有 pKa 的原子
                'all'        → 回傳每一顆原子的 pKa 預測
        """
        from rdkit import Chem
        from self_pka_chemutils import tensorize_for_pka
        from torch_geometric.data import Data

        if device is None:
            device = next(self.parameters()).device
        self.to(device).eval()

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Can't parse SMILES: {smiles}")

        fatoms, ei, ea = tensorize_for_pka(smiles)
        n = fatoms.size(0)
        data = Data(x=fatoms, edge_index=ei, edge_attr=ea,
                    pka_labels=torch.zeros(n), batch=torch.zeros(n, dtype=torch.long),
                    smiles=smiles).to(device)

        with torch.no_grad():
            logits, pka_pred, _ = self(data)          # forward 回傳 raw pKa 值
            # 在這裡進行反標準化，因為 forward 已不再反標準化
            pka_pred = pka_pred * self.pka_std + self.pka_mean

        has_pka = (logits.argmax(1) == 1).cpu().numpy()
        pka_pred = pka_pred.cpu().numpy()

        if eval_mode == "predicted":
            idx = np.where(has_pka == 1)[0]
        else:  # 'all'
            idx = np.arange(n)

        return {
            "smiles": smiles,
            "mol": mol,
            "atom_has_pka": has_pka,
            "atom_pka_values": pka_pred,
            "pka_positions": idx.tolist(),
            "pka_values": pka_pred[idx].tolist(),
            "eval_mode": eval_mode
        }






