import torch 
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import MessagePassing, GCNConv, Linear, BatchNorm, GlobalAttention
import time  # 添加時間模組
import sys   # 添加系統模組

# 添加全局調試開關
DEBUG = False
MAX_RETRIES = 3  # 最大重試次數，避免無限循環

def debug_log(message):
    """調試日誌函數"""
    if DEBUG:
        timestamp = time.strftime('%H:%M:%S', time.localtime())
        print(f"[DEBUG {timestamp}] {message}", flush=True)
        sys.stdout.flush()  # 強制立即輸出

class BondMessagePassing(nn.Module):
    def __init__(self, node_features, bond_features, hidden_size, depth=5, dropout=0.3):
        super(BondMessagePassing, self).__init__()
        self.expected_node_features = node_features
        self.expected_bond_features = bond_features
        self.hidden_size = hidden_size
        self.depth = depth
        self.dropout_rate = dropout
        self.initialized = False
        self.device = None
        self.retry_count = 0  # 添加重試計數器
        
        debug_log(f"BondMessagePassing初始化: 預期節點特徵={node_features}, 邊特徵={bond_features}, 隱藏大小={hidden_size}")
    
    def _initialize_layers(self, node_dim, edge_dim):
        """根據實際數據維度動態初始化所有層"""
        debug_log(f"正在初始化層：節點特徵={node_dim}, 邊特徵={edge_dim}, 隱藏層={self.hidden_size}")
        
        # 核心網絡層
        self.W_i = nn.Linear(node_dim + edge_dim, self.hidden_size).to(self.device)
        self.W_h = nn.Linear(self.hidden_size, self.hidden_size).to(self.device)
        self.W_o = nn.Linear(node_dim + self.hidden_size, self.hidden_size).to(self.device)
        self.node_transform = nn.Linear(node_dim, self.hidden_size).to(self.device)
        
        # 激活函數和正則化
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # 標記初始化完成
        self.initialized = True
        debug_log(f"層初始化完成: W_i.shape={self.W_i.weight.shape}, W_o.shape={self.W_o.weight.shape}")

    def update(self, M_t, H_0):
        # debug_log(f"執行update: M_t.shape={M_t.shape}, H_0.shape={H_0.shape}")
        H_t = self.W_h(M_t)
        H_t = self.relu(H_0 + H_t)
        H_t = self.dropout(H_t)
        return H_t

    def message(self, H, batch):
        # debug_log(f"執行message: H.shape={H.shape}, edge_index.shape={batch.edge_index.shape}")
        index_torch = batch.edge_index[1].unsqueeze(1).repeat(1, H.shape[1])
        M_all = torch.zeros(len(batch.x), H.shape[1], dtype=H.dtype, device=H.device).scatter_reduce_(0, index_torch, H, reduce="sum", include_self=False)[batch.edge_index[0]]
        M_rev = H[batch.rev_edge_index]
        return M_all - M_rev
        
    def forward(self, batch):
        # 添加進入forward的調試訊息
        debug_log(f"進入BondMessagePassing.forward: batch.x.shape={batch.x.shape}, batch.edge_attr.shape={batch.edge_attr.shape}")
        
        # 保存設備信息
        self.device = batch.x.device
        
        # 獲取實際數據維度
        actual_node_dim = batch.x.shape[1]
        actual_edge_dim = batch.edge_attr.shape[1]
        
        # 檢查是否需要初始化或重新初始化
        if not self.initialized:
            debug_log(f"首次執行: 節點特徵={actual_node_dim}, 邊特徵={actual_edge_dim}, 連接特徵總維度={actual_node_dim + actual_edge_dim}")
            self._initialize_layers(actual_node_dim, actual_edge_dim)
        
        # 核心前向傳播邏輯
        try:
            debug_log(f"開始前向傳播計算...")
            combined_features = torch.cat([batch.x[batch.edge_index[0]], batch.edge_attr], dim=1)
            debug_log(f"連接特徵完成: shape={combined_features.shape}")
            
            H_0 = self.W_i(combined_features)
            debug_log(f"W_i計算完成: H_0.shape={H_0.shape}")
            H = self.relu(H_0)
            
            # 報告循環進度
            for i in range(1, self.depth):
                debug_log(f"執行message-passing迭代 {i}/{self.depth-1}...")
                M = self.message(H, batch)
                H = self.update(M, H_0)
            
            debug_log("開始計算最終嵌入...")    
            index_torch = batch.edge_index[1].unsqueeze(1).repeat(1, H.shape[1]) 
            M = torch.zeros(len(batch.x), H.shape[1], dtype=H.dtype, device=H.device).scatter_reduce_(0, index_torch, H, reduce="sum", include_self=False)
            
            debug_log(f"計算節點轉換: batch.x.shape={batch.x.shape}, node_transform.weight.shape={self.node_transform.weight.shape}")
            transformed_x = self.node_transform(batch.x)
            M = torch.where(M.sum(dim=1, keepdim=True) == 0, transformed_x, M)
            
            debug_log(f"組合最終嵌入...")
            final_features = torch.cat([batch.x, M], dim=1)
            debug_log(f"最終特徵: shape={final_features.shape}, W_o.weight.shape={self.W_o.weight.shape}")
            H = self.W_o(final_features)
            H = self.relu(H)    
            H = self.dropout(H)
            
            debug_log(f"BondMessagePassing.forward完成: 輸出H.shape={H.shape}")
            # 重置重試計數器
            self.retry_count = 0
            return H
            
        except RuntimeError as e:
            # 詳細的錯誤訊息和診斷
            debug_log(f"錯誤發生在前向傳播中：{str(e)}")
            
            try:
                # 嘗試取得更多診斷信息
                cat_shape = torch.cat([batch.x[batch.edge_index[0]], batch.edge_attr], dim=1).shape
                debug_log(f"診斷信息：")
                debug_log(f"  - 連接張量形狀: {cat_shape}")
                debug_log(f"  - W_i權重形狀: {self.W_i.weight.shape}")
                debug_log(f"  - 節點特徵: 實際={actual_node_dim}, 預期={self.W_o.weight.shape[1] - self.hidden_size}")
            except Exception as diag_e:
                debug_log(f"無法獲取完整診斷信息: {diag_e}")
            
            # 防止無限重試
            self.retry_count += 1
            if self.retry_count > MAX_RETRIES:
                debug_log(f"達到最大重試次數({MAX_RETRIES})，終止執行")
                raise RuntimeError(f"達到最大重試次數({MAX_RETRIES})，原始錯誤: {str(e)}")
            
            # 重新初始化並重試
            if "shapes cannot be multiplied" in str(e):
                debug_log(f"重試 #{self.retry_count}: 重新初始化層以適應新的特徵維度...")
                self.initialized = False
                self._initialize_layers(actual_node_dim, actual_edge_dim)
                # 重新嘗試前向傳播
                debug_log(f"重新嘗試前向傳播...")
                return self.forward(batch)
            else:
                raise e

class SequentialBondMessagePassing(nn.Module):
    """
    按解離順序進行多層消息傳遞的模型
    """
    def __init__(self, node_dim, bond_dim, hidden_dim, num_layers=2, dropout=0.3):
        super(SequentialBondMessagePassing, self).__init__()
        self.num_layers = num_layers
        
        # 創建多個消息傳遞層，每層對應一個解離階段
        self.mp_layers = nn.ModuleList([
            BondMessagePassing(node_dim, bond_dim, hidden_dim, depth=3, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # 層間轉換，用於將一層的輸出轉換為下一層的輸入
        self.layer_transitions = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) 
            for _ in range(num_layers-1)
        ])
        
        debug_log(f"初始化SequentialBondMessagePassing: num_layers={num_layers}")
    
    def forward(self, batch):
        debug_log(f"進入SequentialBondMessagePassing.forward")
        
        # 檢查是否有解離順序信息
        has_dissociation_order = hasattr(batch, 'dissociation_order')
        dissociation_order = batch.dissociation_order if has_dissociation_order else None
        
        # 獲取批次中最大的解離順序
        max_order = self.num_layers - 1
        if has_dissociation_order:
            max_order = max(0, dissociation_order.max().item())
            max_order = min(max_order, self.num_layers - 1)  # 確保不超過層數
        
        debug_log(f"檢測到最大解離順序: {max_order}")
        
        # 第一層處理所有節點
        H = self.mp_layers[0](batch)
        
        # 後續層根據解離順序處理
        for i in range(1, min(self.num_layers, max_order + 2)):
            debug_log(f"處理第{i}層消息傳遞...")
            
            # 應用層間轉換
            H_next = self.layer_transitions[i-1](H)
            
            # 第i層處理解離順序 >= i 的節點
            if has_dissociation_order:
                # 創建掩碼，標記當前層需要處理的節點
                mask = dissociation_order >= i
                
                # 如果沒有需要處理的節點，可以提前退出
                if not mask.any():
                    debug_log(f"第{i}層沒有需要處理的節點，提前退出")
                    break
                
                # 處理這些節點
                H_updated = self.mp_layers[i](batch)
                
                # 只更新需要處理的節點
                H = torch.where(mask.unsqueeze(1), H_updated, H_next)
            else:
                # 如果沒有解離順序信息，處理所有節點
                H = self.mp_layers[i](batch)
        
        debug_log(f"SequentialBondMessagePassing.forward完成")
        return H

class PKA_GNN(nn.Module):
    def __init__(self, node_dim, bond_dim, hidden_dim, max_dissociation_steps=2, dropout=0.1):
        super(PKA_GNN, self).__init__()
        debug_log(f"初始化PKA_GNN: node_dim={node_dim}, bond_dim={bond_dim}, hidden_dim={hidden_dim}, max_dissociation_steps={max_dissociation_steps}")
        
        # 使用順序性消息傳遞層
        self.sequential_mp = SequentialBondMessagePassing(
            node_dim, bond_dim, hidden_dim, num_layers=max_dissociation_steps, dropout=dropout)
        
        self.pool = global_mean_pool
        
        # 分類任務 - 判斷原子是否可解離
        self.dissociable_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2))  # [0, 1] - 不可解離/可解離
        
        # 回歸任務 - 預測pKa值，添加解離順序的考慮
        self.pka_regressor = nn.Sequential(
            nn.Linear(hidden_dim + 1, 256),  # 增加1個特徵用於解離順序
            nn.ReLU(),
            nn.Dropout(dropout * 1.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # 解離順序預測器 - 新增
        self.order_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, max_dissociation_steps)  # 預測最多max_dissociation_steps階段的解離
        )
        
        # 損失函數
        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_reg = nn.MSELoss()
        self.criterion_order = nn.CrossEntropyLoss()  # 新增解離順序損失
        
        # 門控RNN
        self.rnn_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid())
        
        # 記錄最大解離階段
        self.max_dissociation_steps = max_dissociation_steps
        
        debug_log("PKA_GNN初始化完成")
    
    # 獲取反向邊緣索引
    @staticmethod
    def _rev_edge_index(edge_index):
        debug_log(f"計算反向邊緣索引: edge_index.shape={edge_index.shape}")
        edge_to_index = {(edge_index[0, i].item(), edge_index[1, i].item()): i 
                         for i in range(edge_index.shape[1])}
        rev_edge_index = torch.full((edge_index.shape[1],), -1, dtype=torch.long)
        for i in range(edge_index.shape[1]):
            u, v = edge_index[0, i].item(), edge_index[1, i].item()
            if (v, u) in edge_to_index:
                rev_edge_index[i] = edge_to_index[(v, u)]
        debug_log(f"反向邊緣索引計算完成")
        return rev_edge_index

    # 圖的前向傳播
    def forward_graph(self, x, edge_index, batch, edge_attr, dissociation_order=None):
        debug_log(f"進入forward_graph: x.shape={x.shape}, edge_index.shape={edge_index.shape}, edge_attr.shape={edge_attr.shape}")
        # 獲取反向邊緣索引
        rev_edge_index = self._rev_edge_index(edge_index)
        
        # 構建圖
        graph_batch = Data(x=x,
                          edge_index=edge_index,
                          rev_edge_index=rev_edge_index,
                          edge_attr=edge_attr)
        
        # 如果有解離順序信息，添加到圖中
        if dissociation_order is not None:
            graph_batch.dissociation_order = dissociation_order
        
        debug_log("開始調用SequentialBondMessagePassing...")
        # 進行前向傳播，考慮解離順序
        node_embeddings = self.sequential_mp(graph_batch)
        debug_log(f"forward_graph完成: node_embeddings.shape={node_embeddings.shape}")
        
        # 返回節點嵌入
        return node_embeddings

    # 主函數
    def forward(self, batch):
        debug_log("進入PKA_GNN.forward")
        # 從批次中提取數據
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        dissociable_masks = batch.dissociable_masks if hasattr(batch, 'dissociable_masks') else None
        pka_values = batch.pka_values if hasattr(batch, 'pka_values') else None
        batch_idx = batch.batch if hasattr(batch, 'batch') else None
        
        # 解離順序信息
        dissociation_order = batch.dissociation_order if hasattr(batch, 'dissociation_order') else None
        current_dissociation = batch.current_dissociation if hasattr(batch, 'current_dissociation') else None
        
        debug_log(f"批次數據: x.shape={x.shape}, edge_attr.shape={edge_attr.shape}, 邊數量={edge_index.shape[1]}")
        
        # 處理圖，獲取節點嵌入，傳遞解離順序
        debug_log("開始處理圖...")
        node_embeddings = self.forward_graph(x, edge_index, batch_idx, edge_attr, dissociation_order)
        
        # 初始化損失
        total_cls_loss, total_reg_loss, total_order_loss = 0, 0, 0
        correct_predictions = 0
        total_dissociable_atoms = 0
        
        # 記錄預測結果
        all_cls_predictions = []
        all_reg_predictions = []
        all_order_predictions = []
        
        # 遍歷每個節點
        debug_log(f"開始遍歷節點進行預測，共{len(x)}個節點")
        progress_step = max(1, len(x) // 10)  # 每10%報告一次進度
        
        for i in range(len(x)):
            # 定期報告進度
            if i % progress_step == 0:
                debug_log(f"處理節點 {i}/{len(x)} ({i/len(x)*100:.1f}%)")
            
            # 進行分類任務 - 判斷是否可解離
            cls_pred = self.dissociable_classifier(node_embeddings[i].unsqueeze(0))
            cls_pred_label = torch.argmax(cls_pred, dim=1).item()
            all_cls_predictions.append(cls_pred_label)
            
            # 如果有標籤信息
            if dissociable_masks is not None:
                target_cls = 1 if dissociable_masks[i].item() > 0 else 0
                cls_loss = self.criterion_cls(cls_pred, torch.tensor([target_cls], device=cls_pred.device))
                total_cls_loss += cls_loss
                
                # 計算分類準確率
                if cls_pred_label == target_cls:
                    correct_predictions += 1
                total_dissociable_atoms += 1
            
            # 如果預測為可解離，則預測pKa值和解離順序
            if cls_pred_label == 1 or (dissociable_masks is not None and dissociable_masks[i].item() > 0):
                # 預測解離順序
                if dissociation_order is not None:
                    order_pred = self.order_predictor(node_embeddings[i].unsqueeze(0))
                    pred_order = torch.argmax(order_pred, dim=1).item()
                    all_order_predictions.append((i, pred_order))
                    
                    # 如果有真實的解離順序
                    true_order = dissociation_order[i].item()
                    if true_order >= 0 and true_order < self.max_dissociation_steps:
                        order_loss = self.criterion_order(order_pred, torch.tensor([true_order], device=order_pred.device))
                        total_order_loss += order_loss
                else:
                    # 如果沒有解離順序信息，默認為0
                    pred_order = 0
                
                # 將解離順序作為回歸輸入的一部分
                order_feature = torch.tensor([[pred_order]], dtype=torch.float32, device=node_embeddings.device)
                regression_input = torch.cat([node_embeddings[i].unsqueeze(0), order_feature / self.max_dissociation_steps], dim=1)
                
                # 預測pKa值
                pka_pred = self.pka_regressor(regression_input)
                all_reg_predictions.append((i, pka_pred.item()))
                
                # 如果有真實pKa值
                if pka_values is not None and i < len(pka_values) and not torch.isnan(pka_values[i]):
                    reg_loss = self.criterion_reg(pka_pred, pka_values[i].unsqueeze(0).unsqueeze(0))
                    total_reg_loss += reg_loss
                    
                    # 更新節點嵌入，使用門控機制
                    node_embeddings = node_embeddings * (1 - self.rnn_gate(node_embeddings))
        
        # 計算總損失，加入解離順序損失
        total_loss = total_cls_loss + total_reg_loss + 0.5 * total_order_loss  # 權重可調整
        
        # 計算分類準確率
        accuracy = correct_predictions / total_dissociable_atoms if total_dissociable_atoms > 0 else 0
        
        debug_log(f"PKA_GNN.forward完成: 分類準確率={accuracy:.4f}, 總損失={total_loss.item():.4f}")
        return all_cls_predictions, all_reg_predictions, (total_loss, total_cls_loss, total_reg_loss, accuracy)

    def predict(self, batch):
        debug_log("進入PKA_GNN.predict")
        # 預測模式，與forward類似但不計算損失
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        batch_idx = batch.batch if hasattr(batch, 'batch') else None
        
        # 解離順序信息
        dissociation_order = batch.dissociation_order if hasattr(batch, 'dissociation_order') else None
        
        # 處理圖，獲取節點嵌入
        debug_log("開始圖嵌入計算...")
        node_embeddings = self.forward_graph(x, edge_index, batch_idx, edge_attr, dissociation_order)
        
        # 記錄預測結果
        dissociable_nodes = []
        pka_predictions = []
        order_predictions = []
        
        # 遍歷每個節點
        debug_log(f"開始遍歷節點進行預測，共{len(x)}個節點")
        for i in range(len(x)):
            if i % 100 == 0:
                debug_log(f"處理節點 {i}/{len(x)} ({i/len(x)*100:.1f}%)")
                
            # 進行分類任務 - 判斷是否可解離
            cls_pred = self.dissociable_classifier(node_embeddings[i].unsqueeze(0))
            cls_pred_label = torch.argmax(cls_pred, dim=1).item()
            
            # 如果預測為可解離，則預測pKa值和解離順序
            if cls_pred_label == 1:
                dissociable_nodes.append(i)
                
                # 預測解離順序
                pred_order = 0
                if dissociation_order is not None:
                    order_pred = self.order_predictor(node_embeddings[i].unsqueeze(0))
                    pred_order = torch.argmax(order_pred, dim=1).item()
                order_predictions.append(pred_order)
                
                # 將解離順序作為回歸輸入的一部分
                order_feature = torch.tensor([[pred_order]], dtype=torch.float32, device=node_embeddings.device)
                regression_input = torch.cat([node_embeddings[i].unsqueeze(0), order_feature / self.max_dissociation_steps], dim=1)
                
                # 預測pKa值
                pka_pred = self.pka_regressor(regression_input)
                pka_predictions.append(pka_pred.item())
                
                # 更新節點嵌入，使用門控機制
                node_embeddings = node_embeddings * (1 - self.rnn_gate(node_embeddings))
        
        debug_log(f"預測完成: 識別出{len(dissociable_nodes)}個可解離節點")
        return dissociable_nodes, pka_predictions, order_predictions 