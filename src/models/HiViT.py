import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from .aggregation_module import TokenWise_TokenReducer, TokenWise_TokenReducer_MoE
from .models_vit import VisionTransformer, vit_base_patch16, vit_large_patch16

def init_weights(m):
    """
    初始化权重的通用函数：
    - 对 nn.Linear 层使用 Xavier uniform 初始化
    - 对 bias 使用 0 初始化
    """
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)

# 可变长的MLP
class FlexibleMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, drop_rates=None, activation_fn=nn.ReLU):
        """
        参数说明：
        - input_size: 输入特征维度，如 768
        - hidden_sizes: 隐藏层大小列表，如 [512, 256]
        - num_classes: 输出维度，如 1（二分类）
        - drop_rates: 每层的 Dropout 概率列表，如 [0.1, 0.1]（长度必须与 hidden_sizes 相同）
        - activation_fn: 激活函数类（默认是 nn.ReLU，可传 nn.LeakyReLU）
        """
        super(FlexibleMLP, self).__init__()

        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.activations = nn.ModuleList()

        if drop_rates is None:
            drop_rates = [0.0] * len(hidden_sizes)
        assert len(drop_rates) == len(hidden_sizes), "drop_rates 和 hidden_sizes 长度不一致"

        prev_size = input_size
        for hidden_size, drop_rate in zip(hidden_sizes, drop_rates):
            self.layers.append(nn.Linear(prev_size, hidden_size))
            self.dropouts.append(nn.Dropout(drop_rate))
            self.activations.append(activation_fn())  # 动态创建激活函数实例
            prev_size = hidden_size

        self.output_layer = nn.Linear(prev_size, num_classes)
        self.apply(init_weights)

    def forward(self, x):
        for fc, drop, act in zip(self.layers, self.dropouts, self.activations):
            x = fc(x)
            x = drop(x)
            x = act(x)
        return self.output_layer(x)

class HiViT_PT(nn.Module):
    pass

class HiViT_FT(nn.Module):
    def __init__(self,
                 encoder_embed_dim=768,
                 use_hierarchical=True,
                 moe=False
                 ):
        super().__init__()


        self.use_hierarchical = use_hierarchical
        self.moe = moe
        print(f"HiViT_FT: use_hierarchical={use_hierarchical}, moe={moe}, encoder_embed_dim={encoder_embed_dim}")

        if encoder_embed_dim == 768:
            self.visual_encoder = vit_base_patch16()  # 输出 [B, N+1, D]
        else:
            self.visual_encoder = vit_large_patch16()  # 输出 [B, N+1, D]

        self.VisualTokenReducer_12 = TokenWise_TokenReducer(input_dim=encoder_embed_dim,hidden_dim=encoder_embed_dim//4 if encoder_embed_dim==1024 else 128)  # 用于最后一层的token聚合

        if self.use_hierarchical:
            self.VisualTokenReducer_3 = TokenWise_TokenReducer(input_dim=encoder_embed_dim,hidden_dim=encoder_embed_dim//4 if encoder_embed_dim==1024 else 128)
            self.VisualTokenReducer_6 = TokenWise_TokenReducer(input_dim=encoder_embed_dim,hidden_dim=encoder_embed_dim//4 if encoder_embed_dim==1024 else 128)
            self.VisualTokenReducer_9 = TokenWise_TokenReducer(input_dim=encoder_embed_dim,hidden_dim=encoder_embed_dim//4 if encoder_embed_dim==1024 else 128)
            if self.moe:
                self.VisualHierarchicalReducer = TokenWise_TokenReducer_MoE(input_dim=encoder_embed_dim, hidden_dim=encoder_embed_dim//4 if encoder_embed_dim==1024 else 128, num_tokens=4, num_experts=4)
                self.classifier = FlexibleMLP(input_size=encoder_embed_dim,
                                            hidden_sizes=[2 * encoder_embed_dim,
                                                            1 * encoder_embed_dim],
                                            num_classes=2,
                                            drop_rates=[0.1, 0.2])
            else:
                self.classifier = FlexibleMLP(input_size=4 * encoder_embed_dim,
                                            hidden_sizes=[2 * encoder_embed_dim,
                                                            1 * encoder_embed_dim],
                                            num_classes=2,
                                            drop_rates=[0.1, 0.2])
        else:
            self.classifier = FlexibleMLP(input_size=encoder_embed_dim,
                                        hidden_sizes=[2 * encoder_embed_dim,
                                                        1 * encoder_embed_dim],
                                        num_classes=2,
                                        drop_rates=[0.1, 0.2])


    def remove_cls_token(self, x):
        # x: [B, N+1, D]
        return x[:, 1:, :]  # [B, N, D]
        
    def forward(self, img_batch):
        # img_batch: [B, 3, H, W]
        # 这里假设 visual_encoder 已经定义并且输出 [B, N+1, D]
        if self.use_hierarchical:
            visual_features_3, visual_features_6, visual_features_9, visual_features_12 = self.visual_encoder.forward_features(img_batch,use_hierarchical=self.use_hierarchical)  # 每个都是 [B, N+1, D]

            # 去掉 cls token
            visual_features_3 = self.remove_cls_token(visual_features_3)  # [B, N, D]
            visual_features_6 = self.remove_cls_token(visual_features_6)  # [B, N, D]
            visual_features_9 = self.remove_cls_token(visual_features_9)  # [B, N, D]
            visual_features_12 = self.remove_cls_token(visual_features_12)  # [B, N, D]

            # 每层单独聚合
            visual_agg_3 = self.VisualTokenReducer_3(visual_features_3)  # [B, D]
            visual_agg_6 = self.VisualTokenReducer_6(visual_features_6)  # [B, D]
            visual_agg_9 = self.VisualTokenReducer_9(visual_features_9)  # [B, D]
            visual_agg_12 = self.VisualTokenReducer_12(visual_features_12)  # [B, D]

            # 最后再进行层次聚合
            if self.moe:
                visual_agg = torch.stack([visual_agg_3, visual_agg_6, visual_agg_9, visual_agg_12], dim=1) # [B, 4, D]
                visual_agg = self.VisualHierarchicalReducer(visual_agg)  # [B, D]
            else:
                visual_agg = torch.concat([visual_agg_3, visual_agg_6, visual_agg_9, visual_agg_12], dim=1) # [B, 4*D]
            
        else:
            visual_features = self.visual_encoder.forward_features(img_batch)  # [B, N+1, D]
            visual_features = self.remove_cls_token(visual_features)  # [B, N, D]
            visual_agg = self.VisualTokenReducer_12(visual_features)  # [B, D]

        output = self.classifier(visual_agg)  # [B, 1]

        if self.moe:
            return output, visual_agg
        else:
            return output, visual_agg_3, visual_agg_6, visual_agg_9, visual_agg_12
            
