# models/model.py

import torch
import torch.nn as nn
import math
from torch_geometric.nn import EdgeConv,MessagePassing
from torch_geometric.nn import knn_interpolate, knn_graph, fps, knn
from torch_geometric.utils import softmax as pyg_softmax
from utils.neighbor_finder import find_neighbors


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
    def __init__(self, in_channels, out_channels, num_heads, dropout=0.1):
        super().__init__(aggr='add', flow='source_to_target')
        
        if out_channels % num_heads != 0:
            raise ValueError(f"out_channels ({out_channels}) must be divisible by num_heads ({num_heads})")
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads

        # --- لایه‌های پروجکشن Q, K, V ---
        self.to_q = nn.Linear(in_channels, out_channels)
        self.to_k = nn.Linear(in_channels, out_channels)
        self.to_v = nn.Linear(in_channels, out_channels)

        # --- لایه MLP برای انکودینگ موقعیت نسبی (Δp) ---
        self.pos_embedding = nn.Sequential(
            nn.Linear(3, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.LayerNorm(out_channels)
        )

        # لایه خطی نهایی پس از ترکیب سرها
        self.to_out = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        
        # لایه FFN (Feed-Forward Network)
        self.ffn = nn.Sequential(
            nn.Linear(out_channels, out_channels * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels * 4, out_channels)
        )

        # اتصال کوتاه (Residual) اگر ابعاد ورودی و خروجی متفاوت باشند
        # اتصال کوتاه (Residual Connection) برای جلوگیری از محو شدن گرادیان
        if in_channels != out_channels:
            self.residual = nn.Linear(in_channels, out_channels)
        else:
            self.residual = nn.Identity()

            
    def forward(self, x, pos, edge_index):
        # x: [N, in_channels], pos: [N, 3], edge_index: [2, E]
        
        # 1. محاسبه Residual اولیه
        x_res_1 = self.residual(x) # [N, out_channels]

        # 2. انتشار پیام (فراخوانی message و aggregate)
        # x = (x, pos) جفت ویژگی و موقعیت
        message_output = self.propagate(edge_index, x=x, pos=pos) # [N, out_channels]

        # 3. اولین اتصال کوتاه و نرمال‌سازی (Pre-Norm)
        x = self.norm1(message_output + x_res_1)

        # 4. دومین اتصال کوتاه (FFN)
        x_res_2 = x
        x_ffn = self.ffn(x)
        x = self.norm2(x_ffn + x_res_2)
        
        return x
    

    def message(self, x_i, x_j, pos_i, pos_j, index):
        # x_i/x_j: [E, in_channels], pos_i/pos_j: [E, 3]
        # index: [E] (اندیس گره مقصد)

        # 1. محاسبه پروجکشن‌های Q, K, V
        query = self.to_q(x_i) # [E, out_channels]
        key   = self.to_k(x_j) # [E, out_channels]
        value = self.to_v(x_j) # [E, out_channels]

        # 2. انکودینگ موقعیت نسبی
        pos_enc = self.pos_embedding(pos_i - pos_j) # [E, out_channels]

        # 3. اضافه کردن انکودینگ موقعیت به Key (روشی استاندارد)
        key = key + pos_enc

        # 4. تغییر ابعاد برای MHA
        # [E, out_channels] -> [E, num_heads, head_dim]
        query = query.view(-1, self.num_heads, self.head_dim)
        key   = key.view(-1, self.num_heads, self.head_dim)
        value = value.view(-1, self.num_heads, self.head_dim)

        # 5. محاسبه امتیاز توجه (Attention Score)
        # (Q * K^T) / sqrt(d_k) -> در اینجا به صورت (Q * K)
        attention_score = (query * key).sum(dim=-1) / math.sqrt(self.head_dim) # [E, num_heads]

        # 6. اعمال Softmax (در راستای همسایگان هر گره مقصد)
        attention_weights = pyg_softmax(attention_score, index) # [E, num_heads]

        # 7. وزن‌دهی به Value ها
        # (weights * V) -> [E, num_heads, head_dim]
        message_output = attention_weights.unsqueeze(-1) * value

        # 8. بازگرداندن به ابعاد اصلی برای aggregate
        # [E, num_heads, head_dim] -> [E, out_channels]
        return message_output.view(-1, self.out_channels)
    
    def update(self, aggr_out, x):
        # aggr_out: [N, out_channels] (خروجی aggregate شده از message ها)
        # x: [N, in_channels] (ویژگی‌های گره مرکزی از forward)
        
        # اعمال لایه خطی نهایی
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
        self.neighbor_finder = neighbor_finder # ذخیره نام متد
        self.search_radius = search_radius # ذخیره شعاع

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
                          num_heads=self.num_heads, # ✅ ارسال num_heads
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
            
            # --- ساخت گراف (فقط یک بار در هر مرحله) ---
            edge_index = None
            if not isinstance(stage, nn.Sequential) and current_x.size(0) > 1:
                 # ✅✅✅ استفاده از تابع همسایه‌یاب ماژولار ✅✅✅
                 edge_index = find_neighbors(
                     current_pos, 
                     batch=current_batch,
                     method=self.neighbor_finder,
                     k=self.knn_param,
                     r=self.search_radius
                 )
            # --- پردازش توسط مرحله (Stage) ---
            if isinstance(stage, nn.Sequential):
                current_x = stage(current_x)
            elif edge_index is not None and edge_index.size(1) > 0: # 💡 بررسی خالی نبودن گراف
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
            # اگر ورودی لایه درشت خالی باشد، نمی‌توان درون‌یابی کرد.
            # در این حالت، ویژگی‌های لایه ریز را با یک MLP ساده پردازش می‌کنیم.
            return self.mlp(torch.cat([torch.zeros_like(fine_features), fine_features], dim=1))

        # استفاده از knn_interpolate برای درون‌یابی امن و بهینه
        # این تابع برای هر نقطه در fine_pos، سه همسایه نزدیک در coarse_pos پیدا می‌کند،
        # ویژگی‌های آنها را بر اساس فاصله وزن‌دهی کرده و به نقطه fine_pos منتقل می‌کند.
        k_safe = min(self.interpolation_k, coarse_pos.size(0))
        interpolated_features = knn_interpolate(
            coarse_features, coarse_pos, fine_pos, k=k_safe
        )

        # ترکیب ویژگی‌های درون‌یابی شده با ویژگی‌های اصلی از طریق skip-connection
        combined = torch.cat([interpolated_features, fine_features], dim=1)
        return self.mlp(combined)

class Decoder(nn.Module):
    def __init__(self, main_output_dim, stages_config, interpolation_k, dropout_param=0.1):
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
    def __init__(self, feature_dim, main_input_dim, main_output_dim, stages_config, 
                 knn_param, num_heads, neighbor_finder, search_radius, interpolation_k, 
                 dropout_param=0.1, kpconv_radius=0.1, kpconv_kernel_size=15):
        """
        Args:
            kpconv_radius (float): شعاع برای لایه KPConv اولیه.
            kpconv_kernel_size (int): تعداد نقاط کرنل برای KPConv اولیه.
        """
        super(ASGFormer, self).__init__()
        
        self.knn_param = knn_param
        self.neighbor_finder = neighbor_finder
        self.search_radius = search_radius

        # --- ۱. انکودر اولیه EdgeConv (جایگزین KPConv) ---
        edgeconv_output_dim = 64 # ابعاد خروجی انکودر اولیه
        
        # EdgeConv یک MLP را به عنوان ورودی می‌گیرد تا روی یال‌ها اعمال شود
        # ورودی MLP: (2 * feature_dim) -> (ویژگی نقطه مرکزی + ویژگی همسایه)
        # 💡 نکته: ما از (2 * (feature_dim + 3)) استفاده می‌کنیم تا pos را هم صریحا در نظر بگیریم
        # این کار به EdgeConv قدرت بیشتری در درک هندسه می‌دهد.
        
        # ما از یک MLP ساده برای EdgeConv استفاده می‌کنیم:
        initial_encoder_nn = nn.Sequential(
            nn.Linear(2 * (feature_dim + 3), edgeconv_output_dim), # (2 * (9+3)) = 24
            nn.ReLU(),
            nn.LayerNorm(edgeconv_output_dim)
        )

        print(f"Initializing EdgeConv layer with input MLP: 2*({feature_dim}+3) -> {edgeconv_output_dim}")
        # ✅ استفاده از لایه EdgeConv که می‌دانیم در محیط شما وجود دارد
        self.initial_encoder_conv = EdgeConv(nn=initial_encoder_nn, aggr='max')
        self.initial_encoder_norm = nn.LayerNorm(edgeconv_output_dim)
        
        # --- ۱. انکودر اولیه KPConv ---
        # ابعاد خروجی این لایه (kpconv_output_dim) یک هایپرپارامتر جدید است
        kpconv_output_dim = 64
        #print(f"Initializing KPConv layer with in_channels={feature_dim}, out_channels={kpconv_output_dim}, radius={kpconv_radius}")
        #self.initial_kpconv = KPConv(
        #    in_channels=feature_dim,        # ورودی: 9 ویژگی خام
        #    out_channels=kpconv_output_dim, # خروجی: 64 ویژگی غنی‌شده محلی
        #    dim=3,
        #    kernel_size=kpconv_kernel_size,
        #    radius=kpconv_radius,
        #    aggr='mean' # یا aggr='add'
        #)
        # 💡 نکته: ممکن است بخواهید یک LayerNorm یا ReLU بعد از KPConv اضافه کنید
        self.kpconv_norm = nn.LayerNorm(kpconv_output_dim)

        # --- ۲. MLPهای اصلی برای эмبدینگ ---
        print(f"Initializing Embedding MLPs: x_mlp input={edgeconv_output_dim}, pos_mlp input=3, output={main_input_dim}")
        # ✅ ورودی x_mlp اکنون خروجی KPConv است (kpconv_output_dim)
        self.x_mlp = nn.Sequential(
            nn.Linear(edgeconv_output_dim, main_input_dim),
            nn.ReLU(),
            nn.LayerNorm(main_input_dim)
        )        
        
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, main_input_dim),
            nn.ReLU(),
            nn.LayerNorm(main_input_dim)
        )

        # --- ۳. انکودر اصلی ASGFormer ---
        print(f"Initializing Main Encoder with input_dim={main_input_dim}")
        self.encoder = Encoder(
            input_dim=main_input_dim, # ورودی انکودر اصلی main_input_dim است
            stages_config=stages_config,
            knn_param=knn_param,
            num_heads=num_heads, 
            neighbor_finder=neighbor_finder, # ✅ ارسال
            search_radius=search_radius,
            dropout_param=dropout_param
        )

        # --- ۴. دیکودر اصلی ASGFormer ---
        print("Initializing Main Decoder...")
        self.decoder = Decoder(
            main_output_dim=main_output_dim,
            stages_config=stages_config,
            interpolation_k=interpolation_k,
            dropout_param=dropout_param
        )

        # --- ۵. مقداردهی اولیه وزن‌ها ---
        self._initialize_weights()
        print("Model Initialization Complete.")
        

    def forward(self, data):
        # ✅ اصلاح: ورودی مدل اکنون آبجکت data از PyG است
        x_initial, pos, labels, batch = data.x, data.pos, data.y, data.batch
        # x_initial: [N, 9], pos: [N, 3], labels: [N], batch: [N]

        # --- ۱. اجرای انکودر اولیه EdgeConv ---        
        # ✅✅✅ استفاده از تابع همسایه‌یاب ماژولار ✅✅✅
        edge_index = find_neighbors(
            pos, 
            batch=batch,
            method=self.neighbor_finder,
            k=self.knn_param,
            r=self.search_radius
        )

        # 💡 ترکیب X و Pos برای ورودی غنی‌تر به EdgeConv
        # این کار به MLP داخل EdgeConv اجازه می‌دهد هم ویژگی‌ها و هم موقعیت را ببیند
        combined_x_pos = torch.cat([x_initial, pos], dim=-1) # [N, 12]

        # اجرای EdgeConv
        # ورودی: (x, edge_index) -> (ویژگی‌ها، گراف)
        x_encoded = self.initial_encoder_conv(x=combined_x_pos, edge_index=edge_index)
        x_encoded = self.initial_encoder_norm(x_encoded) # [N, 64]

        # --- ۱. اجرای انکودر اولیه KPConv ---
        # KPConv ویژگی‌های محلی غنی‌شده را استخراج می‌کند
        # ورودی: x_initial (9 بعدی), pos, batch
        # خروجی: x_encoded (64 بعدی)
        # print(f"KPConv Input shapes: x={x_initial.shape}, pos={pos.shape}, batch={batch.shape if batch is not None else 'None'}")
        #x_encoded2 = self.initial_kpconv(x=x_initial, pos=pos, batch=batch)
        #x_encoded2 = self.kpconv_norm(x_encoded2) # اعمال نرمال‌سازی
        # print(f"KPConv Output shape: {x_encoded.shape}") # Should be [N, 64]

        # --- ۲. اجرای MLPهای اصلی برای эмبدینگ ---
        # ✅ x_mlp اکنون روی خروجی KPConv کار می‌کند
        x_emb = self.x_mlp(x_encoded) # ورودی: [N, 64], خروجی: [N, main_input_dim=32]
        # pos_mlp همچنان روی pos اصلی کار می‌کند
        pos_emb = self.pos_mlp(pos)   # ورودی: [N, 3], خروجی: [N, main_input_dim=32]
        # print(f"Embedding shapes: x_emb={x_emb.shape}, pos_emb={pos_emb.shape}")
        
        # ترکیب دو эмبدینگ
        combined_features = x_emb + pos_emb # [N, main_input_dim=32]
        # print(f"Combined features shape: {combined_features.shape}")
        
        # --- ۳. اجرای انکودر و دیکودر اصلی ASGFormer ---
        # انکودر اصلی ویژگی‌های ترکیب‌شده و pos اصلی را دریافت می‌کند
        # print("Entering Main Encoder...")
        encoder_features, positions, sampled_labels, batches = self.encoder(combined_features, pos, labels, batch)
        # print("Exited Main Encoder. Entering Main Decoder...")

        # 💡 نکته: اگر KPConv داون‌سمپلینگ انجام می‌داد، باید skip connections دیکودر را تنظیم می‌کردیم.
        # اما KPConv ساده، تعداد نقاط را تغییر نمی‌دهد.

        logits, final_labels_from_decoder = self.decoder(encoder_features, positions, sampled_labels, batches)
        # print("Exited Main Decoder.")
        # print(f"Logits shape: {logits.shape}, Final Labels shape: {final_labels_from_decoder.shape}")

        # ✅ خروجی باید labels اصلی باشد، نه برچسب‌های داون‌سمپل شده
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
            # 💡 بهبود: می‌توان مقداردهی اولیه خاصی برای KPConv اضافه کرد (اختیاری)
            #elif isinstance(m, KPConv):
            #     nn.init.xavier_uniform_(m.weight)
            #     if m.bias is not None:
            #         nn.init.zeros_(m.bias)