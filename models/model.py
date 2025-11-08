# models/model.py

import torch
import torch.nn as nn
import math
from torch_geometric.nn import EdgeConv,MessagePassing
from torch_geometric.nn import knn_interpolate, knn_graph, fps, knn
from torch_geometric.utils import softmax as pyg_softmax
from utils.neighbor_finder import find_neighbors


class VirtualNode(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.aggregate = nn.Linear(hidden_dim, hidden_dim)
        self.distribute = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        if x.size(0) == 0:
            return x
        global_context = x.mean(dim=0, keepdim=True)
        global_context = self.aggregate(global_context)
        global_context = self.norm(global_context)
        return x + self.distribute(global_context)

class AdaptiveGraphTransformerBlock(MessagePassing):
    def __init__(self, in_channels, out_channels, num_heads, dropout=0.1):
        super().__init__(aggr='add', flow='source_to_target')
        
        if out_channels % num_heads != 0:
            raise ValueError(f"out_channels ({out_channels}) must be divisible by num_heads ({num_heads})")
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads

        self.to_q = nn.Linear(in_channels, out_channels)
        self.to_k = nn.Linear(in_channels, out_channels)
        self.to_v = nn.Linear(in_channels, out_channels)

        self.pos_embedding = nn.Sequential(
            nn.Linear(3, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.LayerNorm(out_channels)
        )

        self.to_out = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        
        self.ffn = nn.Sequential(
            nn.Linear(out_channels, out_channels * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels * 4, out_channels)
        )

        if in_channels != out_channels:
            self.residual = nn.Linear(in_channels, out_channels)
        else:
            self.residual = nn.Identity()
            
    def forward(self, x, pos, edge_index):
        x_res_1 = self.residual(x) 

        message_output = self.propagate(edge_index, x=x, pos=pos) 

        x = self.norm1(message_output + x_res_1)

        x_res_2 = x
        x_ffn = self.ffn(x)
        x = self.norm2(x_ffn + x_res_2)
        
        return x

    def message(self, x_i, x_j, pos_i, pos_j, index):
        query = self.to_q(x_i) 
        key   = self.to_k(x_j) 
        value = self.to_v(x_j) 

        pos_enc = self.pos_embedding(pos_i - pos_j) 

        key = key + pos_enc

        query = query.view(-1, self.num_heads, self.head_dim)
        key   = key.view(-1, self.num_heads, self.head_dim)
        value = value.view(-1, self.num_heads, self.head_dim)

        attention_score = (query * key).sum(dim=-1) / math.sqrt(self.head_dim) 

        attention_weights = pyg_softmax(attention_score, index) 

        message_output = attention_weights.unsqueeze(-1) * value

        return message_output.view(-1, self.out_channels)
    
    def update(self, aggr_out, x):
        return self.to_out(aggr_out)
   

class Stage(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, dropout_param=0.1):
        super(Stage, self).__init__()
        layers = []
        for i in range(num_layers):
            current_input_dim = input_dim if i == 0 else hidden_dim
            layers.append(AdaptiveGraphTransformerBlock(current_input_dim, hidden_dim, num_heads, dropout_param))
        self.layers = nn.ModuleList(layers)

    def forward(self, x, pos, edge_index):
        if x.size(0) == 0:
            return x
        for layer in self.layers:
            x = layer(x, pos, edge_index)
        return x

class Encoder(nn.Module):
    def __init__(self, input_dim, stages_config, knn_param, num_heads, 
                 neighbor_finder, search_radius, dropout_param=0.1):
        super(Encoder, self).__init__()
        
        self.knn_param = knn_param
        self.num_heads = num_heads
        self.neighbor_finder = neighbor_finder 
        self.search_radius = search_radius 

        self.stages = nn.ModuleList()
        self.virtual_nodes = nn.ModuleList()
        self.downsampling_ratios = []

        current_dim = input_dim
        for idx, stage_cfg in enumerate(stages_config):
            if idx == 0:
                self.stages.append(
                    nn.Sequential(
                        nn.Linear(current_dim, stage_cfg['hidden_dim']),
                        nn.ReLU(),
                        nn.LayerNorm(stage_cfg['hidden_dim']), 
                    )
                )
            else:
                self.stages.append(
                    Stage(input_dim=current_dim,
                          hidden_dim=stage_cfg['hidden_dim'],
                          num_layers=stage_cfg['num_layers'],
                          num_heads=self.num_heads, 
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
            
            edge_index = None
            if not isinstance(stage, nn.Sequential) and current_x.size(0) > 1:
                 edge_index = find_neighbors(
                     current_pos, 
                     batch=current_batch,
                     method=self.neighbor_finder,
                     k=self.knn_param,
                     r=self.search_radius
                 )
            if isinstance(stage, nn.Sequential):
                current_x = stage(current_x)
            elif edge_index is not None and edge_index.size(1) > 0: 
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

class InterpolationStage(nn.Module):
    def __init__(self, coarse_dim, fine_dim, out_dim, interpolation_k, dropout_param=0.1):
        super(InterpolationStage, self).__init__()
        
        self.interpolation_k = interpolation_k
        self.mlp = nn.Sequential(
            nn.Linear(coarse_dim + fine_dim, out_dim),
            nn.ReLU(),
            nn.LayerNorm(out_dim),
            nn.Dropout(p=dropout_param)
        )

    def forward(self, coarse_features, coarse_pos, fine_features, fine_pos):
        if coarse_pos.size(0) == 0:
            return self.mlp(torch.cat([torch.zeros_like(fine_features), fine_features], dim=1))

        k_safe = min(self.interpolation_k, coarse_pos.size(0))
        interpolated_features = knn_interpolate(
            coarse_features, coarse_pos, fine_pos, k=k_safe
        )

        combined = torch.cat([interpolated_features, fine_features], dim=1)
        return self.mlp(combined)

class Decoder(nn.Module):
    def __init__(self, main_output_dim, stages_config, interpolation_k, dropout_param=0.1):
        super(Decoder, self).__init__()
        
        self.stages = nn.ModuleList()
        num_encoder_stages = len(stages_config)

        for i in range(num_encoder_stages - 1):
            coarse_stage_cfg = stages_config[-(i + 1)] 
            fine_stage_cfg = stages_config[-(i + 2)]
            
            self.stages.append(
                InterpolationStage(
                    coarse_dim=coarse_stage_cfg['hidden_dim'],
                    fine_dim=fine_stage_cfg['hidden_dim'],
                    out_dim=fine_stage_cfg['hidden_dim'],
                    interpolation_k=interpolation_k,
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

        x = encoder_features.pop()
        pos = positions.pop()

        for stage in self.stages:
            skip_features = encoder_features.pop()
            skip_pos = positions.pop()
            
            x = stage(x, pos, skip_features, skip_pos)
            pos = skip_pos 

        final_labels = sampled_labels[0]
        return self.final_mlp(x), final_labels

class ASGFormer(nn.Module):
    def __init__(self, feature_dim, main_input_dim, main_output_dim, stages_config, 
                 knn_param, num_heads, neighbor_finder, search_radius, interpolation_k, 
                 dropout_param=0.1, kpconv_radius=0.1, kpconv_kernel_size=15):

        super(ASGFormer, self).__init__()
        
        self.knn_param = knn_param
        self.neighbor_finder = neighbor_finder
        self.search_radius = search_radius

        edgeconv_output_dim = 64 
        
        initial_encoder_nn = nn.Sequential(
            nn.Linear(2 * feature_dim, edgeconv_output_dim), 
            nn.ReLU(),
            nn.LayerNorm(edgeconv_output_dim)
        )

        print(f"Initializing EdgeConv layer with input MLP: 2*({feature_dim}+3) -> {edgeconv_output_dim}")
        self.initial_encoder_conv = EdgeConv(nn=initial_encoder_nn, aggr='max')
        self.initial_encoder_norm = nn.LayerNorm(edgeconv_output_dim)
        
        kpconv_output_dim = 64
        self.kpconv_norm = nn.LayerNorm(kpconv_output_dim)

        print(f"Initializing Embedding MLPs: x_mlp input={edgeconv_output_dim}, pos_mlp input=3, output={main_input_dim}")
        self.x_mlp = nn.Sequential(
            nn.Linear(edgeconv_output_dim, main_input_dim),
            nn.ReLU(),
            nn.LayerNorm(main_input_dim)
        )        
        
        print(f"Initializing Main Encoder with input_dim={main_input_dim}")
        self.encoder = Encoder(
            input_dim=main_input_dim, 
            stages_config=stages_config,
            knn_param=knn_param,
            num_heads=num_heads, 
            neighbor_finder=neighbor_finder, 
            search_radius=search_radius,
            dropout_param=dropout_param
        )

        print("Initializing Main Decoder...")
        self.decoder = Decoder(
            main_output_dim=main_output_dim,
            stages_config=stages_config,
            interpolation_k=interpolation_k,
            dropout_param=dropout_param
        )

        self._initialize_weights()
        print("Model Initialization Complete.")

    def forward(self, data):
        x_initial, pos, labels, batch = data.x, data.pos, data.y, data.batch

        edge_index = find_neighbors(
            pos, 
            batch=batch,
            method=self.neighbor_finder,
            k=self.knn_param,
            r=self.search_radius
        )

        x_encoded = self.initial_encoder_conv(x=x_initial, edge_index=edge_index)
        x_encoded = self.initial_encoder_norm(x_encoded)

        combined_features = self.x_mlp(x_encoded) 

        encoder_features, positions, sampled_labels, batches = self.encoder(combined_features, pos, labels, batch)

        logits, final_labels_from_decoder = self.decoder(encoder_features, positions, sampled_labels, batches)

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
            