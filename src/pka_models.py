import torch 
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import MessagePassing, GCNConv, Linear, BatchNorm, GlobalAttention

class BondMessagePassing(nn.Module):
    def __init__(self, node_features, bond_features, hidden_size, depth=5, dropout=0.3):
        super(BondMessagePassing, self).__init__()
        self.W_i  = nn.Linear(node_features + bond_features, hidden_size)
        self.W_h  = nn.Linear(hidden_size, hidden_size)
        self.W_o  = nn.Linear(node_features + hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.depth = depth 
        # 新增轉換層，用於將原始特徵轉換為隱藏層大小
        self.node_transform = nn.Linear(node_features, hidden_size)

    def update(self, M_t, H_0):
        H_t = self.W_h(M_t)
        H_t = self.relu(H_0 + H_t)
        H_t = self.dropout(H_t)
        return H_t

    def message(self, H, batch):
        index_torch = batch.edge_index[1].unsqueeze(1).repeat(1, H.shape[1])
        M_all = torch.zeros(len(batch.x), H.shape[1], dtype=H.dtype, device=H.device).scatter_reduce_(0, index_torch, H, reduce="sum", include_self=False)[batch.edge_index[0]]
        M_rev = H[batch.rev_edge_index]
        return M_all - M_rev

    def forward(self, batch):
        H_0 = self.W_i(torch.cat([batch.x[batch.edge_index[0]], batch.edge_attr], dim=1))
        H   = self.relu(H_0)
        for _ in range(1, self.depth):
            M = self.message(H, batch)
            H = self.update(M, H_0)
        index_torch = batch.edge_index[1].unsqueeze(1).repeat(1, H.shape[1]) # Normal GNN transform
        M = torch.zeros(len(batch.x), H.shape[1], dtype=H.dtype, device=H.device).scatter_reduce_(0, index_torch, H, reduce="sum", include_self=False)
        
        # 修改這行，使用轉換層將batch.x轉換為相同維度
        transformed_x = self.node_transform(batch.x)
        M = torch.where(M.sum(dim=1, keepdim=True) == 0, transformed_x, M)
            
        H = self.W_o(torch.cat([batch.x, M], dim=1))
        H = self.relu(H)    
        H = self.dropout(H)
        return H             

class PKA_GNN(nn.Module):
    def __init__(self, node_dim, bond_dim, hidden_dim, dropout=0.1):
        super(PKA_GNN, self).__init__()
        self.GCN1  = BondMessagePassing(node_dim, bond_dim, hidden_dim, depth=3, dropout=dropout)
        self.pool  = global_mean_pool
        
        # 分類任務 - 判斷原子是否可解離
        self.dissociable_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2))  # [0, 1] - 不可解離/可解離
        
        # 回歸任務 - 預測pKa值
        self.pka_regressor = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout * 1.5),  # 增加dropout
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 1.5),  # 增加dropout
            nn.Linear(64, 1),
            nn.Sigmoid(),  # 限制輸出在0-1之間
            nn.Hardtanh(0, 14)  # 將輸出映射到合理的pKa範圍(0-14)
        )
        
        # 損失函數
        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_reg = nn.MSELoss()
        
        # 門控RNN
        self.rnn_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid())  # 改為Sigmoid確保輸出範圍在0-1之間
    
    # 獲取反向邊緣索引
    @staticmethod
    def _rev_edge_index(edge_index):
        edge_to_index = {(edge_index[0, i].item(), edge_index[1, i].item()): i 
                         for i in range(edge_index.shape[1])}
        rev_edge_index = torch.full((edge_index.shape[1],), -1, dtype=torch.long)
        for i in range(edge_index.shape[1]):
            u, v = edge_index[0, i].item(), edge_index[1, i].item()
            if (v, u) in edge_to_index:
                rev_edge_index[i] = edge_to_index[(v, u)]
        return rev_edge_index

    # 圖的前向傳播
    def forward_graph(self, x, edge_index, batch, edge_attr):
        # 獲取反向邊緣索引
        rev_edge_index = self._rev_edge_index(edge_index)
        # 構建圖
        graph_batch = Data(x=x,
                          edge_index=edge_index,
                          rev_edge_index=rev_edge_index,
                          edge_attr=edge_attr)
        # 進行前向傳播
        node_embeddings = self.GCN1(graph_batch)
        # 返回節點嵌入
        return node_embeddings

    # 主函數
    def forward(self, batch):
        # 從批次中提取數據
        x, edge_index, edge_attr, dissociable_masks, pka_values = batch.x, batch.edge_index, batch.edge_attr, batch.dissociable_masks, batch.pka_values
        batch_idx = batch.batch if hasattr(batch, 'batch') else None
        
        # 處理圖，獲取節點嵌入
        node_embeddings = self.forward_graph(x, edge_index, batch_idx, edge_attr)
        
        # 初始化損失
        total_cls_loss, total_reg_loss = 0, 0
        correct_predictions = 0
        total_dissociable_atoms = 0
        
        # 記錄預測結果
        all_cls_predictions = []
        all_reg_predictions = []
        
        # 遍歷每個節點
        for i in range(len(x)):
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
            
            # 如果預測為可解離，則預測pKa值
            if cls_pred_label == 1 or (dissociable_masks is not None and dissociable_masks[i].item() > 0):
                # 預測pKa值
                pka_pred = self.pka_regressor(node_embeddings[i].unsqueeze(0))
                all_reg_predictions.append((i, pka_pred.item()))
                
                # 如果有真實pKa值
                if pka_values is not None and i < len(pka_values) and not torch.isnan(pka_values[i]):
                    reg_loss = self.criterion_reg(pka_pred, pka_values[i].unsqueeze(0).unsqueeze(0))
                    total_reg_loss += reg_loss
                    
                    # 更新節點嵌入，使用門控機制
                    node_embeddings = node_embeddings * (1 - self.rnn_gate(node_embeddings))
        
        # 計算總損失
        total_loss = total_cls_loss + total_reg_loss
        
        # 計算分類準確率
        accuracy = correct_predictions / total_dissociable_atoms if total_dissociable_atoms > 0 else 0
        
        return all_cls_predictions, all_reg_predictions, (total_loss, total_cls_loss, total_reg_loss, accuracy)

    def predict(self, batch):
        # 預測模式，與forward類似但不計算損失
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        batch_idx = batch.batch if hasattr(batch, 'batch') else None
        
        # 處理圖，獲取節點嵌入
        node_embeddings = self.forward_graph(x, edge_index, batch_idx, edge_attr)
        
        # 記錄預測結果
        dissociable_nodes = []
        pka_predictions = []
        
        # 遍歷每個節點
        for i in range(len(x)):
            # 進行分類任務 - 判斷是否可解離
            cls_pred = self.dissociable_classifier(node_embeddings[i].unsqueeze(0))
            cls_pred_label = torch.argmax(cls_pred, dim=1).item()
            
            # 如果預測為可解離，則預測pKa值
            if cls_pred_label == 1:
                dissociable_nodes.append(i)
                pka_pred = self.pka_regressor(node_embeddings[i].unsqueeze(0))
                pka_predictions.append(pka_pred.item())
                
                # 更新節點嵌入，使用門控機制
                node_embeddings = node_embeddings * (1 - self.rnn_gate(node_embeddings))
        
        return dissociable_nodes, pka_predictions 