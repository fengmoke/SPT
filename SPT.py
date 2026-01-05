import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class spt(nn.Module):
    def __init__(self, cfg,in_channels, out_channels, kernel_sizes, routing_type='soft',top_k=2):
        super(spt, self).__init__()
        self.top_k = top_k
        self.cfg = cfg
        self.kernel_sizes = kernel_sizes
        self.routing_type = routing_type
        self.unified_padding = (max(kernel_sizes) // 2, 0)  # 所有卷积层使用相同padding
        self.save_path = os.path.join(cfg.save_path, cfg.data, cfg.finetune_type)
        if cfg.shot is not None:
            self.save_path = os.path.join(self.save_path, cfg.shot)
        
        # 初始化卷积层
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=(k, 1), padding=(k//2, 0)) for k in kernel_sizes
        ])
        
        # 注意力模块（门控机制）
        if self.cfg.attention:
            self.attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, len(kernel_sizes), 1),
                nn.Softmax(dim=1)
            )
        if self.cfg.ratio:
            self.attention_ratio = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, 1, 1),
                nn.Sigmoid()
            )
        
        # 重参数化卷积核（仅在推理时使用）
        self.reparameterized_weight = None
        self.attention_cached = None
    
    def forward(self, x):
        # 生成注意力权重
        if self.cfg.attention:
            attention_weights = self.attention(x)
            # self.set_cached_attention(attention_weights[0])
            self.set_cached_attention(torch.mean(attention_weights, dim=0))
            attention_weights = attention_weights.view(attention_weights.size(0), -1, attention_weights.size(2), attention_weights.size(3))
        if self.cfg.ratio:
            ratio = self.attention_ratio(x)
        # print(1 - ratio)
        if self.training:  # 训练模式
            
            # 根据路由类型进行加权求和
            if self.routing_type == 'hard':
                # 硬路由：选择 top-k 专家进行卷积和加权求和
                _, top_k_indices = attention_weights.view(attention_weights.size(0), -1).topk(self.top_k, dim=1)
                weighted_sum = sum(self.convs[i](x) * attention_weights[:, i:i+1, :, :] for i in top_k_indices[0])
            else:
                # 软路由：所有卷积核的加权和
                if self.cfg.attention:
                    weighted_sum = sum(conv(x) * attention_weights[:, i:i+1, :, :] for i, conv in enumerate(self.convs))
                else:
                    weighted_sum = sum(conv(x) for conv in self.convs)
            if self.cfg.ratio:
                return weighted_sum * (1 - ratio) + x * ratio
            else:
                return weighted_sum + x
        else:  # 推理模式
            if self.reparameterized_weight is None:
                # 在推理模式下，合并卷积核
                self.reparameterized_weight = self._reparameterize_weights()

            # 使用合并后的卷积核进行推理
            if self.cfg.ratio:
                return F.conv2d(x, self.reparameterized_weight, padding=self.unified_padding) * (1 - ratio) + x * ratio
            else:
                return F.conv2d(x, self.reparameterized_weight, padding=self.unified_padding) + x
    def _reparameterize_weights(self):
        """
        将训练过程中计算的注意力权重应用到每个卷积核，合并卷积核为一个卷积核。
        """
        # 创建一个与第一个卷积层权重相同形状的合并卷积核
        # if self.attention_cached is None:
        #     # 如果没有缓存的注意力权重，加载预训练的权重
        #     self.attention_cached = torch.load(os.path.join(self.save_path, 'attention_weights.pth'))
        merged_weight = torch.zeros_like(self.convs[0].weight)

        # 在训练阶段保存的注意力权重用于合并卷积核
        # 使用每个卷积核的权重乘以注意力权重
        # 添加维度对齐处理
        for i, conv in enumerate(self.convs):
            # 获取当前卷积核对应的注意力权重
            
            # 对每个卷积核进行注意力加权
            if self.cfg.attention:
                attn_weight = self.attention_cached[i].view(1, 1, -1, 1)
                weighted_conv = conv.weight * attn_weight
            else:
                weighted_conv = conv.weight
            
            # 填充不同尺寸卷积核到最大尺寸
            pad_size = abs((merged_weight.size(2) - conv.weight.size(2))) // 2
            merged_weight = F.pad(merged_weight, (0, 0, pad_size, pad_size))
            
            merged_weight += weighted_conv

        return merged_weight

    def set_cached_attention(self, attention_weights):
        """
        在训练过程中调用此方法保存注意力权重，以供推理时使用。
        """
        self.attention_cached = attention_weights.detach()  # Detach to ensure no gradients are computed for the cached weights
        # torch.save(self.attention_cached, os.path.join(self.save_path, 'attention_weights.pth'))
        
