import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import knn, knn_graph
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import MessagePassing # برای پیاده‌سازی کارآمد Attention
import torch_geometric.utils as pyg_utils


class VirtualNode(nn.Module):
    """
    ماژول گره مجازی (VNGO) برای تجمیع و توزیع زمینه جهانی.
    به جای جمع، از میانگین استفاده می‌شود تا در برابر تغییرات اندازه N پایدارتر باشد.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.aggregate = nn.Linear(hidden_dim, hidden_dim)
        self.distribute = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # تجمیع زمینه جهانی: میانگین ویژگی‌ها روی تمام نقاط ->
        global_context = x.mean(dim=0, keepdim=True)
        global_context = self.aggregate(global_context)
        global_context = self.norm(global_context)
        
        # توزیع زمینه جهانی: Context به x اضافه می‌شود (از طریق Broadcasting)
        distributed_context = self.distribute(global_context)
        return x + distributed_context


class AdaptiveGraphAttention(MessagePassing):
    """
    پیاده‌سازی بهینه AGT Attention با استفاده از MessagePassing PyG (پیچیدگی O(N*K)).
    این جایگزین پیاده‌سازی O(N^2) قبلی است.
    """
    def __init__(self, hidden_dim, dropout_param=0.1, heads=1):
        super().__init__(aggr='add', flow='source_to_target')
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.d_k = hidden_dim // heads # ابعاد هر سر توجه
        
        if self.d_k * self.heads!= hidden_dim:
             raise ValueError("hidden_dim must be divisible by heads")

        # لایه‌های خطی برای Q, K, V
        self.lin_q = nn.Linear(hidden_dim, hidden_dim)
        self.lin_k = nn.Linear(hidden_dim, hidden_dim)
        self.lin_v = nn.Linear(hidden_dim, hidden_dim)

        # MLP برای جاسازی موقعیت نسبی (Delta P) - Attention Bias
        # این ماژول بایاس موقعیتی را محاسبه می‌کند.
        self.position_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.dropout = nn.Dropout(p=dropout_param)
        self.norm = nn.LayerNorm(hidden_dim)
        self.scale = math.sqrt(self.d_k)

    def forward(self, x, edge_index, pos):
        # 1. محاسبه Q, K, V
        Q = self.lin_q(x).view(-1, self.heads, self.d_k)
        K = self.lin_k(x).view(-1, self.heads, self.d_k)
        V = self.lin_v(x).view(-1, self.heads, self.d_k)

        # 2. محاسبه جاسازی موقعیت نسبی (Delta P)
        # row: گره هدف (i), col: گره مبدا (j)
        row, col = edge_index
        rel_pos = pos[row] - pos[col] # [E, 3] -> p_i - p_j
        
        # جاسازی موقعیت نسبی (Attention Bias)
        pos_emb = self.position_mlp(rel_pos).view(-1, self.heads, self.d_k) # [E, heads, d_k]

        # 3. شروع Message Passing
        # انتقال Q، K، V و pos_emb به متد message
        # Q_i در message با Q[row]، K_j با K[col] و... متناظر است.
        out = self.propagate(edge_index, Q=Q, K=K, V=V, pos_emb=pos_emb)
        
        # 4. اتصال باقی‌مانده و نرمال‌سازی
        out = out.view(-1, self.hidden_dim)
        # طبق Eq. (5) مقاله: T_ij = Norm(Attn + phi F_i)
        output = self.norm(out + x) 
        
        # توجه: وزن‌های توجه تنک نهایی (E x heads) به دلیل ماهیت MessagePassing برگردانده نمی‌شوند
        # مگر اینکه به طور خاص در متد message ذخیره شوند.
        return output, None 

    def message(self, Q_i, K_j, V_j, pos_emb):
        # Q_i: Query گره هدف (i), K_j: Key گره مبدا (j), V_j: Value گره مبدا (j)
        
        # ترکیب بایاس موقعیتی با Value (مشابه Point Transformer)
        # این عمل W_ij (در فرمول AGT) را دربرمی‌گیرد که شامل Delta p_ij است.
        V_j_adapted = V_j + pos_emb # [E, heads, d_k]

        # محاسبه امتیاز توجه (Attention Score) برای هر یال (edge)
        # آلفا [E, heads] است
        alpha = (Q_i * K_j).sum(dim=-1) / self.scale 

        # اعمال Softmax در طول همسایگان (row)
        # pyg_utils.softmax اطمینان می‌دهد که Softmax روی هر مجموعه همسایه (گره هدف row) اعمال شود.
        alpha = pyg_utils.softmax(alpha, self.index) # self.index = row

        # ضرب در Value ترکیبی و Dropout
        return self.dropout(alpha).unsqueeze(-1) * V_j_adapted


class AGTBlock(nn.Module):
    """
    بلوک هسته‌ای Adaptive Graph Transformer (AGT)
    """
    def __init__(self, input_dim, output_dim, dropout_param=0.1):
        super(AGTBlock, self).__init__()
        # MLP اولیه
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_param),
            nn.Linear(output_dim, output_dim)
        )
        # استفاده از لایه بهینه شده
        self.graph_attention = AdaptiveGraphAttention(output_dim, dropout_param)
        self.residual = nn.Identity() if input_dim == output_dim else nn.Linear(input_dim, output_dim)
        # لایه نرمال‌سازی دوم (بعد از Attn و Residual)
        self.norm_2 = nn.LayerNorm(output_dim) 
        self.dropout = nn.Dropout(p=dropout_param)

    def forward(self, x, edge_index, pos):
        # 1. MLP & Residual (Pre-Norm style)
        residual_in = self.residual(x)
        h = self.mlp(x)
        
        # 2. Adaptive Graph Attention (O(N*K))
        h_attn, _ = self.graph_attention(h, edge_index, pos)
        h_attn = self.dropout(h_attn)
        
        # 3. ترکیب و Norm نهایی
        output = self.norm_2(h_attn + residual_in)
        return output, None # توجه: None به جای attention_weights بازگردانده می‌شود.


class Stage(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout_param=0.1):
        super(Stage, self).__init__()
        
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if dropout_param < 0 or dropout_param >= 1:
            raise ValueError("dropout_param must be in [0, 1)")
        
        layers = []
        for i in range(num_layers):
            current_input_dim = input_dim if i == 0 else hidden_dim
            # استفاده از AGTBlock بهینه شده
            layers.append(AGTBlock(current_input_dim, hidden_dim, dropout_param))
        self.layers = nn.ModuleList(layers)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_param = dropout_param

    def forward(self, x, edge_index, pos):
        self._validate_inputs(x, edge_index, pos)
        attention_weights_all = [] # این لیست اکنون فقط شامل None خواهد بود، اما ساختار حفظ می‌شود.
        
        for layer in self.layers:
            # x, attention_weights = layer(x, edge_index, pos)
            x, attention_weights = layer(x, edge_index, pos)
            attention_weights_all.append(attention_weights)
        
        return x, attention_weights_all

    def _validate_inputs(self, x, edge_index, pos):
        if x.dim()!= 2:
            raise ValueError(f"Features must be 2D tensor, got {x.dim()}D")
        if edge_index.dim()!= 2 or edge_index.size(0)!= 2:
            raise ValueError("edge_index must be shape [2, E]")
        if pos.dim()!= 2 or pos.size(-1)!= 3:
            raise ValueError("pos must be shape [N, 3]")
        if x.size(0)!= pos.size(0):
            raise ValueError("Feature and position count mismatch")
        if x.size(-1)!= self.input_dim and x.size(-1)!= self.hidden_dim:
            # فقط اولین لایه باید input_dim را بپذیرد، لایه‌های بعدی hidden_dim.
            pass

    def extra_repr(self):
        return f"input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, num_layers={self.num_layers}, dropout={self.dropout_param}"


class InterpolationStage(nn.Module):
    """
    مرحله درون‌یابی دیکودر با توجه محلی (KNN-based Attention). (اصلاح شده)
    جهت‌یابی صحیح KNN و تجمیع دقیق با استفاده از index_add_.
    """
    def __init__(self, decoder_dim, encoder_dim, out_dim, knn_param, dropout_param=0.1):
        super(InterpolationStage, self).__init__()
        
        if knn_param <= 0:
            raise ValueError("knn_param must be positive")
        
        self.knn_param = knn_param
        self.decoder_dim = decoder_dim
        self.encoder_dim = encoder_dim
        self.out_dim = out_dim

        self.query_layer = nn.Linear(encoder_dim, decoder_dim)
        self.key_layer = nn.Linear(decoder_dim, decoder_dim)
        self.value_layer = nn.Linear(decoder_dim, decoder_dim)

        self.combination_mlp = nn.Sequential(
            nn.Linear(decoder_dim + encoder_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_param),
            nn.Linear(out_dim, out_dim)
        )
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(p=dropout_param)

    def forward(self, decoder_features, decoder_pos, encoder_features, encoder_pos, encoder_labels):
        self._validate_inputs(decoder_features, decoder_pos, encoder_features, encoder_pos)
        
        N_fine = encoder_pos.size(0)
        K = self.knn_param

        # 1. جستجوی K همسایه (KNN) - جهت‌یابی صحیح برای Upsampling
        # x=نقاط فاین/کوئری, y=نقاط کوآرس/مرجع
        knn_indices = knn(x=encoder_pos, y=decoder_pos, k=K) # [2, N_fine * K]
        
        neighbor_indices = knn_indices[1] # [N_fine * K] (اندیس‌های کوآرس)

        # 2. استخراج ویژگی‌ها
        neighbor_decoder_features = decoder_features[neighbor_indices] #

        # 3. محاسبه Q, K, V
        # اندیس‌های هدف (فاین) برای کوئری
        target_indices = knn_indices
        query_fine = encoder_features[target_indices] #
        
        query = self.query_layer(query_fine)         #
        keys = self.key_layer(neighbor_decoder_features)  #
        values = self.value_layer(neighbor_decoder_features) #

        # 4. محاسبه امتیاز توجه
        scores = (query * keys).sum(dim=-1) / math.sqrt(self.decoder_dim) 
        
        # 5. Softmax و تجمیع وزن‌دار (Aggregation)
        # Softmax روی K همسایه برای هر نقطه فاین (target_indices)
        weights = pyg_utils.softmax(scores, target_indices) # [N_fine * K]
        
        weighted_values = values * weights.unsqueeze(-1) #

        # تجمیع نهایی (Upsampled): جمع وزن‌دار روی هر N_fine نقطه هدف.
        aggregated_decoder_features = torch.zeros(N_fine, self.decoder_dim, device=weighted_values.device)
        aggregated_decoder_features.index_add_(0, target_indices, weighted_values) #

        # 6. ترکیب ویژگی‌ها (Skip Connection)
        combined_features = torch.cat([aggregated_decoder_features, encoder_features], dim=-1)
        upsampled_features = self.combination_mlp(combined_features)
        
        output = self.norm(upsampled_features)
        output = self.dropout(output)
        
        return output, encoder_pos, encoder_labels

    def _validate_inputs(self, decoder_features, decoder_pos, encoder_features, encoder_pos):
        if decoder_features.dim()!= 2 or encoder_features.dim()!= 2:
            raise ValueError("Features must be 2D tensors")
        if decoder_pos.size(-1)!= 3 or encoder_pos.size(-1)!= 3:
            raise ValueError("Positions must have 3 coordinates")
        if decoder_features.size(0)!= decoder_pos.size(0):
            raise ValueError("Decoder features and positions count mismatch")


class Encoder(nn.Module):
    """
    ماژول Encoder با ساختار هرمی و Downsampling
    """
    def __init__(self, input_dim, stages_config, knn_param, dropout_param=0.1):
        super(Encoder, self).__init__()
        
        if not isinstance(stages_config, list) or len(stages_config) == 0:
            raise ValueError("stages_config must be a non-empty list")
        if knn_param <= 0:
            raise ValueError("knn_param must be positive")

        self.knn_param = knn_param
        self.stages = nn.ModuleList()
        self.virtual_nodes = nn.ModuleList()
        self.downsampling_ratios = []

        current_dim = input_dim
        for idx, stage_cfg in enumerate(stages_config):
            if 'hidden_dim' not in stage_cfg or 'num_layers' not in stage_cfg:
                raise ValueError("Stage config must contain hidden_dim and num_layers")
            
            # لایه MLP اولیه (Stage 1)
            if idx == 0:
                self.stages.append(
                    nn.Sequential(
                        nn.Linear(current_dim, stage_cfg['hidden_dim']),
                        nn.ReLU(),
                        nn.Dropout(p=dropout_param),
                        nn.Linear(stage_cfg['hidden_dim'], stage_cfg['hidden_dim'])
                    )
                )
            # لایه‌های AGT (Stage 2 تا آخر)
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

    def forward(self, x, pos, labels):
        self._validate_inputs(x, pos, labels)
        features = []
        positions = []
        sampled_labels = []
        attention_maps = []

        for stage, virtual_node, ratio in zip(self.stages, self.virtual_nodes, self.downsampling_ratios):
            if ratio is not None:
                # Downsample (استفاده از FPS)
                x, pos, labels, edge_index = self._downsample(x, pos, labels, ratio)
            else:
                # ساخت گراف KNN در بالاترین رزولوشن (N)
                edge_index = knn_graph(pos, k=self.knn_param, loop=False)

            if isinstance(stage, nn.Sequential):
                x = stage(x)
                attention_weights = None
            else:
                # اجرای AGT Stage بهینه شده
                x, attention_weights = stage(x, edge_index, pos)
                attention_maps.append(attention_weights)

            # اعمال Virtual Node (VNGO)
            x = virtual_node(x)
            features.append(x)
            positions.append(pos)
            sampled_labels.append(labels)
            
        return features, positions, sampled_labels, attention_maps

    def _downsample(self, x, pos, labels, ratio):
        #... (بقیه منطق downsample دست‌نخورده باقی می‌ماند)...
        ratio_val = ratio.item() if isinstance(ratio, torch.Tensor) else float(ratio)
        if ratio_val <= 0 or ratio_val > 1:
            raise ValueError(f"Downsample ratio must be in (0, 1], got {ratio_val}")
        
        # نکته بهبود: این FPS می‌تواند در آینده با CFPS یا CBS جایگزین شود.
        mask = pyg_nn.fps(pos, ratio=ratio_val)
        x_sampled = x[mask]
        pos_sampled = pos[mask]
        labels_sampled = labels[mask]
        
        # ساخت گراف KNN روی نقاط نمونه‌برداری شده
        edge_index = knn_graph(pos_sampled, k=self.knn_param, loop=False)
        return x_sampled, pos_sampled, labels_sampled, edge_index

    def _validate_inputs(self, x, pos, labels):
        #... (بقیه اعتبارسنجی‌ها دست‌نخورده باقی می‌ماند)...
        if x.dim()!= 2:
            raise ValueError(f"Features must be 2D tensor, got {x.dim()}D")
        if pos.dim()!= 2 or pos.size(-1)!= 3:
            raise ValueError("Positions must be shape [N, 3]")
        if labels.dim()!= 1:
            raise ValueError("Labels must be 1D tensor")
        if x.size(0)!= pos.size(0) or x.size(0)!= labels.size(0):
            raise ValueError("Inputs must have same number of points")


class Decoder(nn.Module):
    """
    ماژول Decoder با Interpolation Stage بهینه شده
    """
    def __init__(self, main_output_dim, stages_config, knn_param, dropout_param=0.1):
        super(Decoder, self).__init__()
        
        if not isinstance(stages_config, list) or len(stages_config) < 2:
            raise ValueError("stages_config must be a list with at least 2 stages")
        if knn_param <= 0:
            raise ValueError("knn_param must be positive")

        self.knn_param = knn_param
        self.stages = nn.ModuleList()
        self.skip_connections = nn.ModuleList()

        for i in range(len(stages_config)-1):
            encoder_stage = stages_config[-(i+1)]
            prev_stage = stages_config[-(i+2)]
            output_dim = prev_stage['hidden_dim']
            self.stages.append(
                InterpolationStage(
                    decoder_dim=encoder_stage['hidden_dim'],
                    encoder_dim=output_dim,
                    out_dim=output_dim,
                    knn_param=knn_param,
                    dropout_param=dropout_param
                )
            )
            self.skip_connections.append(
                nn.Sequential(
                    nn.Linear(output_dim, output_dim),
                    nn.ReLU(),
                    nn.Dropout(p=dropout_param)
                )
            )
        self.final_mlp = nn.Sequential(
            nn.Linear(stages_config[0]['hidden_dim'], main_output_dim),
            nn.Dropout(p=dropout_param)
        )

    def forward(self, encoder_features, positions, sampled_labels):
        self._validate_inputs(encoder_features, positions, sampled_labels)
        
        # شروع از عمیق‌ترین (کوآرس‌ترین) لایه
        x = encoder_features[-1]
        pos = positions[-1]
        labels = sampled_labels[-1]

        for i, (stage, skip_conn) in enumerate(zip(self.stages, self.skip_connections)):
            # ویژگی‌های Skip از لایه متناظر (N/4, 64)
            skip_features = encoder_features[-(i+2)]
            skip_pos = positions[-(i+2)]
            skip_lbls = sampled_labels[-(i+2)]
            
            # درون‌یابی: upsample از x, pos (کوآرس) به skip_features, skip_pos (فاین)
            x, pos, labels = stage(
                decoder_features=x,
                decoder_pos=pos,
                encoder_features=skip_features,
                encoder_pos=skip_pos,
                encoder_labels=skip_lbls
            )
            # ترکیب با Skip Connection (جمع)
            x = x + skip_conn(skip_features)

        return self.final_mlp(x), labels

    def _validate_inputs(self, encoder_features, positions, sampled_labels):
        #... (بقیه اعتبارسنجی‌ها دست‌نخورده باقی می‌ماند)...
        if not (len(encoder_features) == len(positions) == len(sampled_labels)):
            raise ValueError("Input lists must have same length")
        for i, (feat, pos, lbl) in enumerate(zip(encoder_features, positions, sampled_labels)):
            if feat.dim() != 2 or pos.dim() != 2 or pos.size(-1) != 3 or lbl.dim() != 1:
                raise ValueError(f"Inputs at stage {i} have invalid shape")
            if feat.size(0) != pos.size(0) or feat.size(0) != lbl.size(0):
                raise ValueError(f"Inputs at stage {i} have mismatched sizes")


class ASGFormer(nn.Module):
    """
    ماژول اصلی ASGFormer با ساختار Encoder-Decoder
    """
    def __init__(self, feature_dim, main_input_dim, main_output_dim, stages_config, knn_param, dropout_param=0.1):
        super(ASGFormer, self).__init__()
        
        if not isinstance(stages_config, list) or len(stages_config) < 2:
            raise ValueError("stages_config must be a list with at least 2 stages")
        if knn_param <= 0:
            raise ValueError("knn_param must be positive")

        # جاسازی ویژگی‌های اولیه (X)
        self.x_mlp = nn.Sequential(
            nn.Linear(feature_dim, main_input_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_param),
            nn.Linear(main_input_dim, main_input_dim)
        )
        # جاسازی موقعیت اولیه (P)
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, main_input_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_param),
            nn.Linear(main_input_dim, main_input_dim)
        )
        # ترکیب ویژگی‌ها: F_combined = MLP(X) + MLP(P)
        
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

    def forward(self, x, pos, labels):
        self._validate_inputs(x, pos, labels)
        
        x_emb = self.x_mlp(x)
        pos_emb = self.pos_mlp(pos)
        combined_features = x_emb + pos_emb
        
        # Encoder: استخراج ویژگی‌های سلسله مراتبی
        encoder_features, positions, sampled_labels, _ = self.encoder(combined_features, pos, labels)
        
        # Decoder: بازتولید برچسب‌ها با Up-sampling
        logits, final_labels = self.decoder(encoder_features, positions, sampled_labels)
        return logits, final_labels

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _validate_inputs(self, x, pos, labels):
        #... (بقیه اعتبارسنجی‌ها دست‌نخورده باقی می‌ماند)...
        if x.dim()!= 2:
            raise ValueError(f"Features must be 2D tensor, got {x.dim()}D")
        if pos.dim()!= 2 or pos.size(-1)!= 3:
            raise ValueError("Positions must be shape [N, 3]")
        if labels.dim()!= 1:
            raise ValueError("Labels must be 1D tensor")
        if x.size(0)!= pos.size(0) or x.size(0)!= labels.size(0):
            raise ValueError("Input sizes must match along dimension 0")
