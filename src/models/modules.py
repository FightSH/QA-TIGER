import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2] # 项目根目录，假设modules.py在src/models/下
sys.path.append(ROOT.as_posix()) # 将项目根目录添加到Python路径，以便导入其他模块

from torch import Tensor
from typing import List, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# 简单的线性投影层
class Projection(nn.Module):
    # 投影模块，用于将输入特征投影到指定维度
    def __init__(self,
                 inp_dim: int = 512, # 输入特征维度
                 d_model: int = 512, # 输出特征维度 (模型主维度)
    ):
        """
        A simple linear projection layer.

        Args:
            inp_dim (int): Input dimension.
            d_model (int): Output dimension (model dimension).
        """
        super(Projection, self).__init__()
        self.proj = nn.Linear(inp_dim, d_model) # 线性投影层
        
    def forward(self, inp: Tensor) -> Tensor:
        """
        Forward pass of the projection layer.

        Args:
            inp (Tensor): Input tensor.

        Returns:
            Tensor: Projected tensor.
        """
        # 前向传播
        # inp: 输入张量
        # return: 投影后的张量
        return self.proj(inp)


# 音频-视频交叉注意力模块 (不包含问题信息)
class AVCrossAttn(nn.Module):
    # 音频-视频交叉注意力模块
    def __init__(self,
                 d_model: int = 512, # 模型维度
                 nhead: int = 8,     # 注意力头数
                 dropout: float = 0.1 # dropout概率
    ):
        """
        Audio-Visual Cross-Attention module.

        Args:
            d_model (int): Model dimension.
            nhead (int): Number of attention heads.
            dropout (float): Dropout rate.
        """
        super(AVCrossAttn, self).__init__()

        # 定义多头注意力层
        self.crs_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout) # 交叉注意力层
        self.slf_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout) # 自注意力层
        # 定义前馈网络层 (FFN)
        self.linear1 = nn.Linear(d_model, d_model) # 线性层1
        self.linear2 = nn.Linear(d_model, d_model) # 线性层2
        self.dropout = nn.Dropout(dropout) # dropout层
        # 定义层归一化 (LayerNorm)
        self.norm1 = nn.LayerNorm(d_model) # LayerNorm层1
        self.norm2 = nn.LayerNorm(d_model) # LayerNorm层2
        self.attn_mask = None # 注意力掩码，当前未使用
        
        # 初始化FFN的权重
        nn.init.kaiming_normal_(self.linear1.weight) # kaiming正态初始化权重
        nn.init.constant_(self.linear1.bias, 0)      # bias初始化为0
        nn.init.kaiming_normal_(self.linear2.weight) # kaiming正态初始化权重
        nn.init.constant_(self.linear2.bias, 0)      # bias初始化为0
    
    # 子前向传播模块，执行一个Transformer编码器层的操作
    def sub_forward(self, 
                src_q: Tensor, # 查询序列 (Query)
                src_v: Tensor, # 值序列 (Value) 和键序列 (Key) - 在交叉注意力中
                query: Optional[Tensor] = None, # 额外的查询输入 (当前未使用)
    ) -> Tensor:
        """
        A sub-forward pass for cross-attention and self-attention.

        Args:
            src_q (Tensor): Query tensor.
            src_v (Tensor): Value tensor for cross-attention.
            query (Optional[Tensor]): Optional query tensor (not used in this method).

        Returns:
            Tensor: Output tensor after attention and feed-forward layers.
        """

        # PyTorch的MultiheadAttention期望输入形状为 (序列长度, 批量大小, 特征维度)
        # 因此需要进行permute操作
        src_q = src_q.permute(1, 0, 2) # [T, B, D]
        src_v = src_v.permute(1, 0, 2) # [T, B, D]
        
        # 计算自注意力 (src_q作为Q, K, V)
        slf_attn = self.slf_attn(src_q, src_q, src_q)[0]
        # 计算交叉注意力 (src_q作为Q, src_v作为K和V)
        crs_attn = self.crs_attn(src_q, src_v, src_v)[0]
        # 残差连接：原始输入 + 自注意力输出 + 交叉注意力输出
        src_q = src_q + \
                self.dropout(slf_attn) + \
                self.dropout(crs_attn)
        src_q = self.norm1(src_q) # 第一个层归一化
        
        # 前馈网络 (FFN)
        # 残差连接：注意力层输出 + FFN输出
        src_q = src_q + \
                self.dropout(self.linear2(self.dropout(F.relu(self.linear1(src_q)))))
        src_q = self.norm2(src_q) # 第二个层归一化
        return src_q.permute(1, 0, 2) # 恢复原始形状 [B, T, D]
    
    # 主前向传播函数
    def forward(self, 
                src_q: Tensor, # 第一个输入序列 (例如音频)
                src_v: Tensor, # 第二个输入序列 (例如视频)
                query: Optional[Tensor] = None, # 额外的查询 (当前未使用)
                visualize: bool = False # 是否返回注意力权重等用于可视化 (当前未实现)
    ) -> List[Tensor]: # 返回两个增强后的序列

        # 对称地进行交叉注意力：src_q关注src_v，src_v关注src_q
        src1 = self.sub_forward(src_q, src_v) # src_q被src_v增强
        src2 = self.sub_forward(src_v, src_q) # src_v被src_q增强
        
        if visualize:
            # 如果需要可视化，当前实现返回None作为权重
            return src1, src2, None 
        return src1, src2


# 音频-视频-问题交叉注意力模块
class AVQCrossAttn(nn.Module):
    # 音频-视频-问题交叉注意力模块
    def __init__(self,
                 d_model: int = 512, # 模型维度
                 nhead: int = 8,     # 注意力头数
                 dropout: float = 0.1 # dropout概率
    ):
        """
        Audio-Visual-Question Cross-Attention module.

        Args:
            d_model (int): Model dimension.
            nhead (int): Number of attention heads.
            dropout (float): Dropout rate.
        """
        super(AVQCrossAttn, self).__init__()

        # 定义三种注意力机制
        self.qst_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout) # 问题引导的注意力
        self.crs_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout) # 模态间的交叉注意力
        self.slf_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout) # 模态内的自注意力
        # 定义FFN层
        self.linear1 = nn.Linear(d_model, d_model) # 线性层1
        self.linear2 = nn.Linear(d_model, d_model) # 线性层2
        self.dropout = nn.Dropout(dropout) # dropout层
        # 定义层归一化
        self.norm1 = nn.LayerNorm(d_model) # LayerNorm层1
        self.norm2 = nn.LayerNorm(d_model) # LayerNorm层2
        
        # 初始化FFN权重
        nn.init.kaiming_normal_(self.linear1.weight) # kaiming正态初始化权重
        nn.init.constant_(self.linear1.bias, 0)      # bias初始化为0
        nn.init.kaiming_normal_(self.linear2.weight) # kaiming正态初始化权重
        nn.init.constant_(self.linear2.bias, 0)      # bias初始化为0
    
    # 子前向传播模块，执行一个包含问题引导的Transformer编码器层操作
    def sub_forward(self, 
                src_q: Tensor, # 当前模态的查询序列 (例如音频或视频)
                src_v: Tensor, # 另一模态的键/值序列 (例如视频或音频)
                query: Tensor, # 问题特征序列 (作为额外的上下文)
                visualize: bool = False # 是否返回注意力权重
    ) -> Tensor:
        """
        A sub-forward pass for cross-attention, self-attention, and question-attention.

        Args:
            src_q (Tensor): Query tensor (e.g., audio or visual features).
            src_v (Tensor): Value tensor for cross-attention (e.g., visual or audio features).
            query (Tensor): Question tensor.
            visualize (bool): If True, enables visualization (not fully implemented here).

        Returns:
            Tensor: Output tensor after attention and feed-forward layers.
            Tensor: Attention weights from question attention.
        """

        # 调整输入形状以适应MultiheadAttention: (序列长度, 批量大小, 特征维度)
        src_q = src_q.permute(1, 0, 2) # [T, B, D]
        src_v = src_v.permute(1, 0, 2) # [T, B, D]
        query = query.permute(1, 0, 2) # [SeqLen_Q, B, D]
        
        # 计算问题引导的注意力 (当前模态src_q关注问题query)
        qst_attn, weight = self.qst_attn(src_q, query, query) # weight是注意力权重
        # 计算自注意力 (当前模态src_q关注自身)
        slf_attn = self.slf_attn(src_q, src_q, src_q)[0]
        # 计算交叉注意力 (当前模态src_q关注另一模态src_v)
        crs_attn = self.crs_attn(src_q, src_v, src_v)[0]
        # 残差连接：原始输入 + 自注意力 + 交叉注意力 + 问题注意力
        src_q = src_q + \
                self.dropout(slf_attn) + \
                self.dropout(crs_attn) + \
                self.dropout(qst_attn)
        src_q = self.norm1(src_q) # 第一个层归一化
        
        # 前馈网络 (FFN)
        src_q = src_q + \
                self.dropout(self.linear2(self.dropout(F.relu(self.linear1(src_q)))))
        src_q = self.norm2(src_q) # 第二个层归一化
        # 恢复原始形状并返回增强后的序列和问题注意力权重
        return src_q.permute(1, 0, 2), weight 
    
    # 主前向传播函数
    def forward(self, 
                src_q: Tensor, # 第一个模态序列 (例如音频)
                src_v: Tensor, # 第二个模态序列 (例如视频)
                query: Tensor, # 问题特征序列
                visualize: bool = False # 是否返回注意力权重
    ) -> List[Tensor]: # 返回两个增强后的模态序列和对应的注意力权重列表

        # 对称地进行问题引导的交叉注意力
        src1, a_weight = self.sub_forward(src_q, src_v, query, visualize) # src_q被src_v和query增强
        src2, v_weight = self.sub_forward(src_v, src_q, query, visualize) # src_v被src_q和query增强
        
        if visualize:
            # 如果需要可视化，返回增强后的序列和注意力权重
            return src1, src2, [a_weight, v_weight]
        return src1, src2 # 否则只返回增强后的序列


# 问题定位模块 (Question Grounding)
class QstGrounding(nn.Module):
    # 问题引导的特征融合模块 (Question Grounding)
    def __init__(self, 
                 d_model: int = 512, # 模型维度
                 nhead: int = 8,     # 注意力头数
                 dropout: float = 0.1):
        """
        Question Grounding module. Grounds the question to the input data (audio/visual).

        Args:
            d_model (int): Model dimension.
            nhead (int): Number of attention heads.
            dropout (float): Dropout rate.
        """
        super(QstGrounding, self).__init__()
        
        self.act = nn.ReLU() # ReLU激活函数
        self.norm = nn.LayerNorm(d_model) # LayerNorm层
        # 注意力层：问题特征作为Query，融合后的多模态特征作为Key和Value
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=0.1)
        # MLP层，用于进一步处理注意力输出
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, d_model)
        )
        self.dropout = nn.Dropout(dropout) # dropout层
        self.mlp.apply(self._init_weights) # 初始化MLP权重

    # MLP权重初始化方法
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight) # kaiming正态初始化权重
            if m.bias is not None:
                nn.init.constant_(m.bias, 0) # bias初始化为0
        
    # 前向传播函数
    def forward(self,
                qst: Tensor, # 问题特征 [B, D_qst] (通常是全局问题表征)
                data: Union[Tensor, List[Tensor]], # 待融合的多模态特征，可以是单个Tensor或Tensor列表
    ) -> Tensor: # 返回融合后的特征 [B, D]
        """
        Forward pass of the QstGrounding module.

        Args:
            qst (Tensor): Question tensor.
            data (Union[Tensor, List[Tensor]]): Input data, can be a single tensor or a list of tensors.

        Returns:
            Tensor: Grounded feature tensor.
        """
        # 处理输入数据data，确保其形状为 (序列长度, 批量大小, 特征维度)
        if isinstance(data, list):
            # 如果data是列表 (例如包含多个全局模态表征)，则先permute再拼接
            data = [d.permute(1, 0, 2) if d.ndim == 3 else d.unsqueeze(0) for d in data] # d可能是[B,D]或[B,T,D]
            data = torch.cat(data, dim=0) # 拼接成 [N_modalities*T, B, D] 或 [N_modalities, B, D]
        else:
            # 如果data是单个Tensor
            data = data.permute(1, 0, 2) if data.ndim == 3 else data.unsqueeze(0) # [T, B, D] or [1, B, D]
        
        qst = qst.unsqueeze(0) # 将问题特征扩展为 [1, B, D] 作为注意力查询
        # 计算注意力：问题查询data中的信息
        attn_output = self.attn(qst, data, data)[0].squeeze(0) # 输出 [B, D]
        # 特征融合：data的平均值 + MLP处理后的注意力输出 (残差连接思想)
        # data.mean(dim=0) 计算data在序列/模态维度上的平均特征
        feat = data.mean(dim=0) + self.dropout(self.mlp(attn_output))
        feat = self.norm(feat) # 层归一化
        return feat


# 时序混合专家模块 (Temporal Mixture of Experts)
class TempMoE(nn.Module):
    # 时间混合专家模块 (Temporal Mixture of Experts)
    def __init__(self, 
                 d_model: int = 512,    # 模型维度
                 nhead: int = 8,        # 注意力头数 (用于qst_attn)
                 topK: int = 5,         # 选择top K个专家
                 n_experts: int = 10,   # 专家总数
                 sigma: int = 9,        # 高斯权重生成时的sigma参数
                 dropout: float = 0.1,  # dropout概率
                 vis_branch: bool = False, # 是否为视觉分支（影响LayerNorm的使用）
    ):
        """
        Temporal Mixture of Experts module.

        Args:
            d_model (int): Model dimension.
            nhead (int): Number of attention heads.
            topK (int): Number of top experts to select.
            n_experts (int): Total number of experts.
            sigma (int): Sigma value for Gaussian generation.
            dropout (float): Dropout rate.
            vis_branch (bool): Whether to use separate normalization for visual branch.
        """
        super(TempMoE, self).__init__()

        self.sigma = sigma
        self.topK = topK
        self.n_experts = n_experts
        
        if vis_branch:
            # 如果是视觉分支，分别对音频和视频输出进行LayerNorm
            self.anorm = nn.LayerNorm(d_model)
            self.vnorm = nn.LayerNorm(d_model)
        else:
            # 否则，对主输出进行LayerNorm
            self.norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout) # dropout层
        # 问题引导的注意力层，用于生成时序权重
        self.qst_attn = nn.MultiheadAttention(d_model, nhead, dropout=0.1)
        # 预测高斯分布参数（中心和宽度）的线性层
        self.gauss_pred = nn.Sequential(
            nn.Linear(d_model, 2 * n_experts) # 每个专家预测2个参数 (中心偏移, 宽度调整)
        )
        # 路由网络，用于选择专家
        self.router = nn.Sequential(
            nn.Linear(d_model, n_experts) # 输出每个专家的路由权重
        )
        # 专家网络列表，每个专家是一个简单的MLP
        self.experts = nn.ModuleList([
            nn.Sequential(*[
                nn.Linear(d_model, int(d_model // 2)),
                nn.ReLU(),
                nn.Linear(int(d_model // 2), d_model)
            ])
            for _ in range(n_experts)
        ])
        self.experts.apply(self._init_weights) # 初始化专家网络权重

        # 定义高斯中心点的基础位置 (均匀分布在0-1之间，有边距)
        self.margin = (1 / (n_experts * 2)) 
        # 生成n_experts个在[margin, 1-margin]区间内均匀分布的中心点
        self.center = torch.linspace(self.margin, 1-self.margin, self.n_experts)
        self.center.requires_grad_(False) # 中心点不参与梯度更新
        self.router.apply(self._init_weights) # 初始化路由网络权重
        self.gauss_pred.apply(self._init_weights) # 初始化高斯参数预测层权重
        
    # 权重初始化方法
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight) # kaiming正态初始化权重
            if m.bias is not None:
                nn.init.constant_(m.bias, 0) # bias初始化为0

    # 生成高斯时序权重
    def generate_gaussian(self,
                          pred: torch.Tensor,      # 预测的高斯参数 (中心偏移, 宽度调整) [B, N_EXPERTS, 2]
                          topk_inds: torch.Tensor, # Top-K专家的索引 [B, TOPK]
                          T: int = 60              # 时序长度
    ) -> Tensor: # 返回生成的时序权重 [B, TOPK, T]
        """
        Generates Gaussian weights based on predictions and top-k indices.

        Args:
            pred (torch.Tensor): Predictions for Gaussian centers and widths.
            topk_inds (torch.Tensor): Indices of the top-k experts.
            T (int): Temporal length.

        Returns:
            Tensor: Generated Gaussian weights.
        """
        # 参考: https://github.com/minghangz/cpl
        weights = []
        # 获取预设的中心点，并加上预测的偏移量
        centers = self.center.unsqueeze(0).repeat(pred.size(0), 1).to(pred.device)
        centers = centers + pred[:, :, 0] # pred[:, :, 0] 是中心偏移
        centers = torch.gather(centers, 1, topk_inds) # 根据topk_inds选择对应专家的中心
        # 获取预测的宽度调整量，并选择对应专家的宽度
        widths = torch.gather(pred[:, :, 1], 1, topk_inds) # pred[:, :, 1] 是宽度调整
        
        #为每个选中的Top-K专家生成高斯权重
        for i in range(self.topK):
            center_i = centers[:, i] # 当前专家的中心 [B]
            width_i = widths[:, i]   # 当前专家的宽度调整 [B]
            
            # 生成时间轴 [0, 1] 上的T个点
            time_axis = torch.linspace(0, 1, T)
            time_axis = time_axis.view(1, -1).expand(center_i.size(0), -1).to(center_i.device) # [B, T]
            # 限制中心点在[0,1]范围，宽度调整后除以sigma得到实际高斯核宽度
            center_i = torch.clamp(center_i.unsqueeze(-1), min=0, max=1) # [B, 1]
            width_i = torch.clamp(width_i.unsqueeze(-1), min=0.09) / self.sigma # [B, 1], min=0.09防止宽度过小
            
            # 计算高斯权重 (未归一化的高斯函数值)
            # w_const = 0.3989422804014327 # 1/sqrt(2*pi)
            # weight = w_const/width_i * torch.exp(-(time_axis-center_i)**2/(2*width_i**2))
            # 简化版，不带常数系数，因为后续会归一化
            weight = torch.exp(-(time_axis-center_i)**2/(2*width_i**2)) # [B, T]
            # 对每个batch内的权重进行归一化 (除以最大值，使得峰值为1)
            weights.append(
                weight/weight.max(dim=-1, keepdim=True)[0]
            )
        return torch.stack(weights, dim=1) # [B, TOPK, T]
    
    # 计算加权后的专家输出
    def get_output(self,
                   experts_logits: Tensor, # 所有专家的输出 [T, B, N_EXPERTS, C] 或 [B*T, N_EXPERTS, C] (取决于reshape方式)
                   gauss_weight: Tensor,   # 生成的高斯时序权重 [B, TOPK, T]
                   topk_inds: Tensor,      # Top-K专家的索引 [B, TOPK]
                   topk_probs: Tensor,     # Top-K专家的路由概率 [B, TOPK]
                   shape: tuple,           # 原始数据形状 (B, T, C)
    ) -> Tensor: # 返回最终的聚合特征 [B, 1, C]
        """
        Combines expert outputs using Gaussian weights and top-k probabilities.

        Args:
            experts_logits (Tensor): Logits from all experts.
            gauss_weight (Tensor): Gaussian weights.
            topk_inds (Tensor): Indices of the top-k experts.
            topk_probs (Tensor): Probabilities of the top-k experts.
            shape (tuple): Shape of the input data (B, T, C).

        Returns:
            Tensor: Combined output tensor.
        """
        B, T, C = shape
        
        # experts_logits 原始是 [T, B, N_EXPERTS, C]
        # 调整并收集Top-K专家的输出
        # permute -> [B, T, N_EXPERTS, C]
        # gather -> [B, T, TOPK, C]
        experts_logits_gathered = torch.gather(
            experts_logits.permute(1, 0, 2, 3), # [B, T, N_EXPERTS, C]
            2, # 沿着N_EXPERTS维度收集
            topk_inds.unsqueeze(1).unsqueeze(-1).repeat(1, T, 1, C) # 扩展topk_inds以匹配维度 [B, T, TOPK, C]
        ) # 输出 [B, T, TOPK, C]

        # 使用高斯时序权重对每个Top-K专家的输出进行加权求和
        # gauss_weight: [B, TOPK, T]
        # experts_logits_gathered: [B, T, TOPK, C]
        # 期望输出: [B, TOPK, C] (每个Top-K专家一个聚合后的特征)
        output_per_expert = []
        for k in range(self.topK):
            # gauss_weight[:, k, :] -> [B, T] (第k个专家的时序权重)
            # experts_logits_gathered[:, :, k, :] -> [B, T, C] (第k个专家的时序特征)
            # ( [B, 1, T] @ [B, T, C] ) -> [B, 1, C]
            weighted_sum = torch.bmm(gauss_weight[:, k, :].unsqueeze(1), experts_logits_gathered[:, :, k, :])
            output_per_expert.append(weighted_sum)
        
        output = torch.cat(output_per_expert, dim=1) # [B, TOPK, C]
        # 使用路由概率对Top-K专家的加权输出进行最终聚合
        # topk_probs: [B, TOPK]
        # ( [B, 1, TOPK] @ [B, TOPK, C] ) -> [B, 1, C]
        output = torch.bmm(topk_probs.unsqueeze(1), output) 
        return output
    
    # 主前向传播函数
    def forward(self,
                qst: Tensor,                       # 问题特征 [B, D]
                data: Tensor,                      # 主模态数据 (音频或视频) [B, T, D]
                sub_data: Optional[Tensor] = None, # 可选的子模态数据 (例如patch特征) [[B,T,D],[B,T,D]] for audio-patch, video-patch
    ) -> Union[Tensor, List[Tensor]]: # 返回聚合后的特征或特征列表
        """
        Forward pass of the TempMoE module.

        Args:
            qst (Tensor): Question tensor.
            data (Tensor): Main input data (Audio or Video).
            sub_data (Optional[Tensor]): Optional sub-data (e.g., patches for audio and video).

        Returns:
            Union[Tensor, List[Tensor]]: If sub_data is provided, returns a list of two tensors (audio and video outputs).
                                         Otherwise, returns a single tensor (main output).
        """
        B, T, C = data.size()
        data_permuted = data.permute(1, 0, 2) # [T, B, C]
        
        qst_unsqueezed = qst.unsqueeze(0) # [1, B, D]
        # 问题引导的注意力，生成时序上下文向量temp_w
        temp_w = self.qst_attn(qst_unsqueezed, data_permuted, data_permuted)[0] # [1, B, C]
        temp_w = temp_w.squeeze(0) # [B, C]
        
        # 路由网络：根据temp_w计算每个专家的选择概率
        router_logits = self.router(temp_w) # [B, N_EXPERTS]
        router_probs = F.softmax(router_logits, dim=-1) # [B, N_EXPERTS]
        # 选择Top-K专家及其概率
        topk_probs, topk_inds = torch.topk(router_probs, self.topK, dim=-1) # [B, TOPK], [B, TOPK]
        # 归一化Top-K概率，使其和为1
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True) # [B, TOPK]
        
        # 预测高斯参数 (中心偏移和宽度调整)
        gauss_cw = self.gauss_pred(temp_w) # [B, 2*N_EXPERTS]
        gauss_cw = gauss_cw.view(B, self.n_experts, 2) # [B, N_EXPERTS, 2]
        # 中心偏移使用tanh激活，并乘以边距进行缩放
        gauss_cw[:, :, 0] = torch.tanh(gauss_cw[:, :, 0]) * self.margin     
        # 宽度调整使用sigmoid激活
        gauss_cw[:, :, 1] = torch.sigmoid(gauss_cw[:, :, 1])
        # 生成高斯时序权重
        gauss_weight = self.generate_gaussian(gauss_cw, topk_inds=topk_inds, T=T) # [B, TOPK, T]
        
        # 如果是视觉分支 (vis_branch=True)，处理主数据(视频)和子数据(patch)
        if sub_data is not None: 
            # sub_data 是一个列表，包含与音频对齐的patch和与视频对齐的patch
            # 这里的data是video特征，sub_data是patch特征 (与net.py中vt_aggregator的调用对应)
            # data (video) + sub_data[0] (audio_patch) -> 融合特征1 (ap_global)
            # data (video) + sub_data[1] (video_patch) -> 融合特征2 (vp_global)
            # 注意：在net.py中，vt_aggregator的输入是 (quest, video, patch)，patch是经过selecter的
            # 而PatchSelecter的输出是 [audio_aligned_patch, video_aligned_patch]
            # 所以这里的sub_data[0]对应audio_aligned_patch, sub_data[1]对应video_aligned_patch

            # 处理与音频对齐的patch (ap_global的计算路径)
            # data (video) 与 audio_aligned_patch 融合
            # 论文中可能是 video 与 audio-grounded patch, audio 与 video-grounded patch
            # 这里 sub_data[0] 对应 PatchSelecter 输出的第一个元素 (audio-aligned patch)
            # data (video) + audio_aligned_patch
            fused_data_ap = data_permuted + sub_data[0].permute(1,0,2) # [T,B,C] + [T,B,C]
            ap_expert_outs = torch.stack([exprt(fused_data_ap) for exprt in self.experts], dim=2) # [T, B, N_EXPERTS, C]
            ap_global = self.get_output(ap_expert_outs, gauss_weight, topk_inds, topk_probs, (B, T, C))  # [B, 1, C]
            
            # 处理与视频对齐的patch (vp_global的计算路径)
            # data (video) 与 video_aligned_patch 融合
            fused_data_vp = data_permuted + sub_data[1].permute(1,0,2) # [T,B,C] + [T,B,C]
            vp_expert_outs = torch.stack([exprt(fused_data_vp) for exprt in self.experts], dim=2) # [T, B, N_EXPERTS, C]
            vp_global = self.get_output(vp_expert_outs, gauss_weight, topk_inds, topk_probs, (B, T, C))  # [B, 1, C]
            return self.anorm(ap_global), self.vnorm(vp_global) # 返回归一化后的两个全局特征
        else:
            # 如果不是视觉分支 (例如纯音频分支)，只处理主数据data
            main_expert_outs = torch.stack([exprt(data_permuted) for exprt in self.experts], dim=2) # [T, B, N_EXPERTS, C]
            main_global = self.get_output(main_expert_outs, gauss_weight, topk_inds, topk_probs, (B, T, C)) # [B, 1, C]
            return self.norm(main_global) # 返回归一化后的单个全局特征


# Patch选择器模块
class PatchSelecter(nn.Module):
    # Patch选择器模块，用于融合patch级别特征和全局音视频特征
    def __init__(self,
                 d_model: int = 512, # 模型维度
                 nhead: int = 8,     # 注意力头数
                 dropout: float = 0.1):
        """
        Patch Selector module. Selects relevant patches based on audio and video features.

        Args:
            d_model (int): Model dimension.
            nhead (int): Number of attention heads.
            dropout (float): Dropout rate.
        """
        super(PatchSelecter, self).__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.dropout_rate = dropout # dropout概率 (变量名修正)
        
        # 定义层归一化
        self.vnorm = nn.LayerNorm(d_model) # 视频输出的LayerNorm
        self.anorm = nn.LayerNorm(d_model) # 音频输出的LayerNorm
        self.dropout = nn.Dropout(self.dropout_rate) # dropout层 (使用修正后的变量名)
        # 定义注意力机制
        self.slf_attn = nn.MultiheadAttention(d_model, nhead, dropout=self.dropout_rate) # patch自注意力
        self.crs_attn = nn.MultiheadAttention(d_model, nhead, dropout=self.dropout_rate) # 全局特征对patch的交叉注意力
        # 定义MLP层
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, d_model)
        )
        self.mlp.apply(self._init_weights) # 初始化MLP权重
        
    # MLP权重初始化方法
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight) # kaiming正态初始化权重
            if m.bias is not None:
                nn.init.constant_(m.bias, 0) # bias初始化为0
    
    # 前向传播函数
    def forward(self,
                patch: Tensor, # patch特征 (B, T, P, D)，P是patch数量
                audio: Tensor, # 全局音频特征 (B, T, D)
                video: Tensor, # 全局视频特征 (B, T, D)
    ) -> List[Tensor]: 
        # 前向传播
        B, T, P, D = patch.size()
        # 将音频和视频特征reshape以匹配patch的处理流程
        audio_reshaped = audio.reshape(B*T, 1, D) # (B*T, 1, D)
        video_reshaped = video.reshape(B*T, 1, D) # (B*T, 1, D)
        patch_reshaped = patch.reshape(B*T, P, D) # (B*T, P, D)
        
        # 调整维度以适应MultiheadAttention
        video_permuted = video_reshaped.permute(1, 0, 2)   # (1, B*T, D)
        audio_permuted = audio_reshaped.permute(1, 0, 2)   # (1, B*T, D)
        patch_permuted = patch_reshaped.permute(1, 0, 2)   # (P, B*T, D)

        # patch自注意力
        patch_slf_attended = patch_permuted + self.slf_attn(patch_permuted, patch_permuted, patch_permuted)[0] # (P, B*T, D)
        
        # 将视频和音频特征拼接作为交叉注意力的查询
        av_data_concat = torch.cat([video_permuted, audio_permuted], dim=0) # (2, B*T, D)
        # 全局音视频特征对patch进行交叉注意力
        # query: av_data_concat, key: patch_slf_attended, value: patch_slf_attended
        attn_output = self.crs_attn(av_data_concat, patch_slf_attended, patch_slf_attended)[0] # (2, B*T, D)
        attn_output_permuted = attn_output.permute(1, 0, 2) # (B*T, 2, D)
        
        # MLP处理
        mlp_output = self.mlp(self.dropout(attn_output_permuted)) # (B*T, 2, D)

        # 分离视频和音频相关的patch特征
        v_patch_feat, a_patch_feat = torch.chunk(mlp_output, 2, dim=1) # 各为 (B*T, 1, D)
        
        # Reshape回原始的时间维度
        v_patch_feat_reshaped = v_patch_feat.reshape(B, T, D) # (B, T, D)
        a_patch_feat_reshaped = a_patch_feat.reshape(B, T, D) # (B, T, D)
        
        # LayerNorm并返回
        # 注意：原代码中permute(1,0,2)后直接norm再permute(1,0,2)回来的操作，
        # 似乎是为了在 (T, B, D) 的维度上进行norm，但LayerNorm通常在最后一个维度上操作。
        # 这里保持原样，但标记一下。如果LayerNorm期望在特征维度D上操作，则不需要permute。
        # 假设这里的意图是保持 (B, T, D) 的输出格式。
        return [
            self.anorm(a_patch_feat_reshaped), # (B, T, D)
            self.vnorm(v_patch_feat_reshaped), # (B, T, D)
        ]


# TSPM (Temporal Segment Proposal Module) Top-K选择模块的参考实现
# 这个类似乎是参考了TSPM项目的一个特定模块，用于根据问题选择Top-K时序片段
# 在当前的QA-TIGER模型中，这个类并没有被直接实例化和使用，但提供了另一种时序选择的思路
class TSPM_topKSelection(nn.Module):
    """
    Top-K Temporal Segment Selection module from TSPM.
    Selects top-K temporal segments based on question-visual attention.
    """
    def __init__(self, topK: int = 10):
        """
        Args:
            topK (int): Number of top temporal segments to select.
        """
        super(TSPM_topKSelection, self).__init__()
        
        self.topK = topK
        # 问题-查询注意力机制，用于计算问题与视频片段的相似度
        self.attn_qst_query = nn.MultiheadAttention(512, 4, dropout=0.1) # d_model=512, nhead=4
        # 用于处理注意力输出的线性层和激活函数
        self.qst_query_linear1 = nn.Linear(512, 512)
        self.qst_query_relu = nn.ReLU()
        self.qst_query_dropout1 = nn.Dropout(0.1)
        self.qst_query_linear2 = nn.Linear(512, 512)
        self.qst_query_dropout2 = nn.Dropout(0.1)
        self.qst_query_visual_norm = nn.LayerNorm(512) # 层归一化

    def QstQueryClipAttn(self, query_feat, kv_feat):
        """
        Performs attention between question query and key-value features (visual).

        Args:
            query_feat (Tensor): Question query features.
            kv_feat (Tensor): Key-value features (e.g., visual clip features).

        Returns:
            Tensor: Attention output.
            Tensor: Attention weights.
        """
        kv_feat = kv_feat.permute(1, 0, 2)
        query_feat = query_feat.unsqueeze(0) # [1, B, D]
        # 问题特征作为Query，视频片段特征作为Key和Value
        attn_feat, temp_weights = self.attn_qst_query(query_feat, kv_feat, kv_feat, 
                                                      attn_mask=None, key_padding_mask=None)
        # attn_feat: [1, B, D], temp_weights: [B, 1, T] (问题对每帧的注意力权重)
        attn_feat = attn_feat.squeeze(0) # [B, D]
        
        # FFN处理
        src = self.qst_query_linear1(attn_feat)
        src = self.qst_query_relu(src)
        src = self.qst_query_dropout1(src)
        src = self.qst_query_linear2(src)
        src = self.qst_query_dropout2(src)

        # 残差连接和层归一化
        attn = attn_feat + src
        attn = self.qst_query_visual_norm(attn)

        return attn, temp_weights # 返回聚合后的视觉特征和时序注意力权重

    def SelectTopK(self, temp_weights, audio_input, visual_input, patch_inputs, B, C):
        '''
        Selects Top-k temporal segments from audio, visual, and patch inputs
        based on temporal attention weights.

        Args:
            temp_weights (Tensor): Temporal attention weights.
            audio_input (Tensor): Audio input features.
            visual_input (Tensor): Visual input features (not directly used for output selection but part of original signature).
            patch_inputs (List[Tensor]): List of patch features (audio_patches, video_patches).
            B (int): Batch size.
            C (int): Feature dimension.

        Returns:
            Tensor: Selected top-K audio features.
            Tuple[Tensor, Tensor]: Selected top-K audio patch features and video patch features.
        '''

        # return temporal indices
        sort_index = torch.argsort(temp_weights, dim=-1)        # [B, 1, T], 升序排序
        top_k_index = sort_index[:, :, -self.topK:]             # [B, 1, Top_K], 取最后K个，即权重最高的K个
        top_k_index_sort, _ = torch.sort(top_k_index)     # [B, 1, Top_K]
        top_k_index_sort_np = top_k_index_sort.cpu().numpy() # 转为numpy方便索引

        # 初始化输出Tensor
        output_audio = torch.zeros(B, self.topK, C).to(audio_input.device)
        out_a_patches = torch.zeros(B, self.topK, C).to(audio_input.device)
        out_v_patches = torch.zeros(B, self.topK, C).to(audio_input.device)
        
        # 根据Top-K索引收集对应的音频和patch特征
        for batch_idx in range(B):
            idx = 0
            for temp_idx in top_k_index_sort_np.tolist()[batch_idx][0]:
                output_audio[batch_idx, idx, :] = audio_input[batch_idx, temp_idx, :]
                out_a_patches[batch_idx, idx, :] = patch_inputs[0][batch_idx, temp_idx, :] # 音频对齐的patch
                out_v_patches[batch_idx, idx, :] = patch_inputs[1][batch_idx, temp_idx, :] # 视频对齐的patch
                idx = idx + 1
        return output_audio, (out_a_patches, out_v_patches) # 返回选择后的音频和patch特征

    # TSPM_topKSelection模块的前向传播
    def forward(self, audio_input, visual_input, patch_inputs, qst_input):
        """
        Forward pass of the TSPM_topKSelection module.

        Args:
            audio_input (Tensor): Audio input features.
            visual_input (Tensor): Visual input features.
            patch_inputs (List[Tensor]): List of patch features.
            qst_input (Tensor): Question input features.

        Returns:
            Tensor: Selected top-K audio features.
            Tuple[Tensor, Tensor]: Selected top-K patch features (audio_patches, video_patches).
        """

        B, T, C = audio_input.size()
        # 1. 计算问题对视频帧的注意力权重
        # temp_clip_attn_feat: (B, C), temp_weights: (B, 1, T)
        _, temp_weights = self.QstQueryClipAttn(qst_input, visual_input)
        # 2. 根据权重选择Top-K时序片段的音频和patch特征
        # output_audio: (B, TopK, C)
        # output_patches: ((B, TopK, C), (B, TopK, C))
        output_audio, output_patches = self.SelectTopK(temp_weights, audio_input, visual_input, patch_inputs, B, C)
        return output_audio, output_patches


