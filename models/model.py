# models/model.py

import torch
import torch.nn as nn
from torch_geometric.nn import knn_graph, knn_interpolate, fps
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax as pyg_softmax
import math

class VirtualNode(nn.Module):
    """
    این ماژول گره مجازی را پیاده‌سازی می‌کند که طبق مقاله، برای агрегирование
    و توزیع اطلاعات سراسری در گراف به کار می‌رود.
    پیاده‌سازی فعلی یک نسخه ساده و مؤثر با استفاده از میانگین‌گیری است.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.aggregate = nn.Linear(hidden_dim, hidden_dim)
        self.distribute = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        if x.size(0) == 0:
            return x
        # агрегирование اطلاعات سراسری با میانگین‌گیری
        global_context = x.mean(dim=0, keepdim=True)
        global_context = self.aggregate(global_context)
        global_context = self.norm(global_context)
        # توزیع اطلاعات سراسری به تمام گره‌ها
        return x + self.distribute(global_context)

class AdaptiveGraphTransformerBlock(MessagePassing):
    """
    بلوک اصلی Adaptive Graph Transformer (AGT) مطابق با بخش 3.2 مقاله.
    این بلوک ویژگی‌های وزنی (Weighted Features) را بر اساس تفاوت ویژگی‌ها و موقعیت
    محاسبه کرده و از مکانیزم توجه گرافی (Graph Attention) برای به‌روزرسانی گره‌ها استفاده می‌کند.
    """
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__(aggr='add', flow='source_to_target')
        
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.mlp_feature = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.LayerNorm(out_channels)
        )
        # این MLP ویژگی وزنی W_ij را بر اساس تفاوت ویژگی (Δf) و تفاوت موقعیت (Δp) می‌سازد
        self.mlp_weighted_feature = nn.Sequential(
            nn.Linear(out_channels + 3, out_channels),
            nn.ReLU(),
            nn.LayerNorm(out_channels) 
        )
        self.mlp_q = nn.Linear(out_channels, out_channels)
        self.mlp_k = nn.Linear(out_channels, out_channels)
        
        # Position Embedding بر اساس تفاوت موقعیت نسبی (Δp_ij)
        self.pos_embedding = nn.Sequential(
            nn.Linear(3, out_channels),
            nn.ReLU(),
            nn.LayerNorm(out_channels)
        )
        self.final_norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(p=dropout)

        # اتصال کوتاه (Residual Connection) برای جلوگیری از محو شدن گرادیان
        if in_channels != out_channels:
            self.residual = nn.Linear(in_channels, out_channels)
        else:
            self.residual = nn.Identity()
            
    def forward(self, x, pos, edge_index):
        edge_index = edge_index.long()
        features = self.mlp_feature(x)
        
        # فراخوانی propagate که به صورت داخلی message و aggregate را اجرا می‌کند
        updated_features = self.propagate(edge_index, x=features, pos=pos)
        
        # اعمال اتصال کوتاه و نرمال‌سازی نهایی
        output = self.final_norm(updated_features + self.residual(x))
        output = self.dropout(output)
        return output

    def message(self, x_i, x_j, pos_i, pos_j, index):
        # محاسبه Δf و Δp طبق مقاله
        delta_f = x_i - x_j
        delta_p = pos_i - pos_j
        
        # Eq. (2) در مقاله: محاسبه ویژگی وزنی W_ij
        concatenated_deltas = torch.cat([delta_f, delta_p], dim=-1)
        W_ij = self.mlp_weighted_feature(concatenated_deltas)
        
        # Eq. (4) در مقاله: مکانیزم توجه گرافی
        query_base = self.mlp_q(x_i)
        pos_emb = self.pos_embedding(delta_p)
        query = query_base + pos_emb
        key = self.mlp_k(W_ij)
        value = W_ij
        
        attention_score = (query * key).sum(dim=-1) / math.sqrt(self.out_channels)
        attention_weights = pyg_softmax(attention_score, index)
        
        # اعمال وزن‌های توجه به مقادیر (value)
        return attention_weights.unsqueeze(-1) * value

class Stage(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout_param=0.1):
        super(Stage, self).__init__()
        layers = []
        for i in range(num_layers):
            current_input_dim = input_dim if i == 0 else hidden_dim
            layers.append(AdaptiveGraphTransformerBlock(current_input_dim, hidden_dim, dropout_param))
        self.layers = nn.ModuleList(layers)

    def forward(self, x, pos, edge_index):
        if x.size(0) == 0:
            return x
        for layer in self.layers:
            x = layer(x, pos, edge_index)
        return x

class Encoder(nn.Module):
    def __init__(self, input_dim, stages_config, knn_param, dropout_param=0.1):
        super(Encoder, self).__init__()
        
        self.knn_param = knn_param
        self.stages = nn.ModuleList()
        self.virtual_nodes = nn.ModuleList()
        self.downsampling_ratios = []

        current_dim = input_dim
        for idx, stage_cfg in enumerate(stages_config):
            # مرحله اول فقط یک MLP ساده است، طبق مقاله
            if idx == 0:
                self.stages.append(
                    nn.Sequential(
                        nn.Linear(current_dim, stage_cfg['hidden_dim']),
                        nn.ReLU(),
                        nn.LayerNorm(stage_cfg['hidden_dim']), # بهبود: افزودن LayerNorm
                    )
                )
            else:
                self.stages.append(
                    Stage(input_dim=current_dim,
                          hidden_dim=stage_cfg['hidden_dim'],
                          num_layers=stage_cfg['num_layers'],
                          dropout_param=dropout_param)
                )

            self.virtual_nodes.append(VirtualNode(stage_cfg['hidden_dim']))
            self.downsampling_ratios.append(stage_cfg.get('downsample_ratio', None))
            current_dim = stage_cfg['hidden_dim']

    def forward(self, x, pos, labels, batch):
        features_list = [x]
        positions_list = [pos]
        labels_list = [labels]
        batches_list = [batch]

        current_x, current_pos, current_labels, current_batch = x, pos, labels, batch

        for stage_idx, (stage, virtual_node, ratio) in enumerate(zip(self.stages, self.virtual_nodes, self.downsampling_ratios)):
            if stage_idx > 0:
                current_x, current_pos, current_labels, current_batch = self._downsample(
                    current_x, current_pos, current_labels, current_batch, ratio
                )

            if current_x.size(0) == 0: break
            
            if isinstance(stage, nn.Sequential):
                current_x = stage(current_x)
            else:
                k_safe = min(self.knn_param, current_x.size(0) - 1)
                if k_safe > 0:
                    edge_index = knn_graph(current_pos, k=k_safe, batch=current_batch, loop=False)
                    current_x = stage(current_x, current_pos, edge_index)
            
            current_x = virtual_node(current_x)
            
            features_list.append(current_x)
            positions_list.append(current_pos)
            labels_list.append(current_labels)
            batches_list.append(current_batch)
            
        return features_list, positions_list, labels_list, batches_list
    
    def _downsample(self, x, pos, labels, batch, ratio):
        if ratio is None or ratio >= 1.0 or x.size(0) == 0:
            return x, pos, labels, batch
        
        mask = fps(pos, batch, ratio=ratio)
        return x[mask], pos[mask], labels[mask], batch[mask] if batch is not None else None

# بهبود: رفع کامل خطای CUDA با جایگزینی پیاده‌سازی دستی با knn_interpolate
class InterpolationStage(nn.Module):
    def __init__(self, coarse_dim, fine_dim, out_dim, knn_param, dropout_param=0.1):
        super(InterpolationStage, self).__init__()
        
        self.knn_param = knn_param
        self.mlp = nn.Sequential(
            nn.Linear(coarse_dim + fine_dim, out_dim),
            nn.ReLU(),
            nn.LayerNorm(out_dim),
            nn.Dropout(p=dropout_param)
        )

    def forward(self, coarse_features, coarse_pos, fine_features, fine_pos):
        if coarse_pos.size(0) == 0:
            # اگر ورودی لایه درشت خالی باشد، نمی‌توان درون‌یابی کرد.
            # در این حالت، ویژگی‌های لایه ریز را با یک MLP ساده پردازش می‌کنیم.
            return self.mlp(torch.cat([torch.zeros_like(fine_features), fine_features], dim=1))

        # استفاده از knn_interpolate برای درون‌یابی امن و بهینه
        # این تابع برای هر نقطه در fine_pos، سه همسایه نزدیک در coarse_pos پیدا می‌کند،
        # ویژگی‌های آنها را بر اساس فاصله وزن‌دهی کرده و به نقطه fine_pos منتقل می‌کند.
        k_safe = min(self.knn_param, coarse_pos.size(0))
        interpolated_features = knn_interpolate(
            coarse_features, coarse_pos, fine_pos, k=k_safe
        )

        # ترکیب ویژگی‌های درون‌یابی شده با ویژگی‌های اصلی از طریق skip-connection
        combined = torch.cat([interpolated_features, fine_features], dim=1)
        return self.mlp(combined)

class Decoder(nn.Module):
    def __init__(self, main_output_dim, stages_config, knn_param, dropout_param=0.1):
        super(Decoder, self).__init__()
        
        self.stages = nn.ModuleList()
        num_encoder_stages = len(stages_config)

        # دیکودر یک مرحله کمتر از انکودر دارد
        for i in range(num_encoder_stages - 1):
            coarse_stage_cfg = stages_config[-(i + 1)] 
            fine_stage_cfg = stages_config[-(i + 2)]
            
            self.stages.append(
                InterpolationStage(
                    coarse_dim=coarse_stage_cfg['hidden_dim'],
                    fine_dim=fine_stage_cfg['hidden_dim'],
                    out_dim=fine_stage_cfg['hidden_dim'],
                    knn_param=knn_param,
                    dropout_param=dropout_param
                )
            )
            
        self.final_mlp = nn.Sequential(
            nn.Linear(stages_config[0]['hidden_dim'], stages_config[0]['hidden_dim']),
            nn.ReLU(),
            nn.Linear(stages_config[0]['hidden_dim'], main_output_dim)
        )

    def forward(self, encoder_features, positions, sampled_labels, batches):
        if not encoder_features:
            return torch.empty(0), torch.empty(0)

        # شروع از خروجی آخرین (درشت‌ترین) لایه انکودر
        x = encoder_features.pop()
        pos = positions.pop()

        for stage in self.stages:
            # ویژگی‌ها و موقعیت‌های لایه ریزتر (اتصال کوتاه)
            skip_features = encoder_features.pop()
            skip_pos = positions.pop()
            
            x = stage(x, pos, skip_features, skip_pos)
            pos = skip_pos # موقعیت فعلی به موقعیت لایه ریزتر به‌روز می‌شود

        final_labels = sampled_labels[0]
        return self.final_mlp(x), final_labels

class ASGFormer(nn.Module):
    def __init__(self, feature_dim, main_input_dim, main_output_dim, stages_config, knn_param, dropout_param=0.1):
        super(ASGFormer, self).__init__()
        
        self.x_mlp = nn.Sequential(
            nn.Linear(feature_dim, main_input_dim),
            nn.ReLU(),
            nn.LayerNorm(main_input_dim)
        )
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, main_input_dim),
            nn.ReLU(),
            nn.LayerNorm(main_input_dim)
        )
        self.encoder = Encoder(
            input_dim=main_input_dim,
            stages_config=stages_config,
            knn_param=knn_param,
            dropout_param=dropout_param
        )
        self.decoder = Decoder(
            main_output_dim=main_output_dim,
            stages_config=stages_config,
            knn_param=knn_param,
            dropout_param=dropout_param
        )
        self._initialize_weights()

    def forward(self, data):
        # ✅ اصلاح: ورودی مدل اکنون آبجکت data از PyG است
        x, pos, labels, batch = data.x, data.pos, data.y, data.batch

        # ✅ اصلاح: x شامل ۹ ویژگی و pos شامل ۳ ویژگی است
        x_emb = self.x_mlp(x)
        pos_emb = self.pos_mlp(pos)
        combined_features = x_emb + pos_emb 
        
        encoder_features, positions, sampled_labels, batches = self.encoder(combined_features, pos, labels, batch)
        
        logits, _ = self.decoder(encoder_features, positions, sampled_labels, batches)
        
        # ✅ اصلاح: خروجی labels باید labels اصلی باشد
        return logits, labels
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)