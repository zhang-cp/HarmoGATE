import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GCNConv, global_mean_pool
except ImportError:
    GCNConv = None
    global_mean_pool = None


class SequenceCNNTransformerEncoder(nn.Module):
    def __init__(self, vocab_size,max_len, embed_dim=128, hidden_dim=256,num_heads=4, dropout=0.2,transformer_layers=1, use_transformer=True):
        super().__init__()
        self.use_transformer = use_transformer
        # 修改：token embedding，对应 MolTrans 的序列 token 表示
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # 修改：CNN 局部特征提取，对应 DeepDTA 的卷积思想
        self.conv1 = nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3)

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # 修改：位置编码，对应 MolTrans/Transformer 的 positional embedding
        self.pos_embedding = nn.Embedding(max_len, hidden_dim)

        # 修改：TransformerEncoder，学习 SMILES/蛋白序列的长距离依赖
        if self.use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=transformer_layers
            )
        else:
            self.transformer = None

    def masked_mean(self, x, mask):
        if mask is None:
            return x.mean(dim=1)

        mask_f = mask.unsqueeze(-1).float()
        return (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)

    def forward(self, x, mask=None):
        emb = self.embedding(x)                  # [B, L, E]
        h = emb.transpose(1, 2)                  # [B, E, L]

        h1 = F.relu(self.conv1(h))
        h2 = F.relu(self.conv2(h1))
        h3 = F.relu(self.conv3(h2) + h1)         # [B, H, L]

        seq_feat = h3.transpose(1, 2)            # [B, L, H]

        seq_len = seq_feat.size(1)
        # 新增：防止输入序列长度超过位置编码最大长度，避免索引越界
        if seq_len > self.pos_embedding.num_embeddings:
             raise ValueError(
                  f"Sequence length {seq_len} exceeds max_len {self.pos_embedding.num_embeddings}."
             )
        pos_ids = torch.arange(seq_len, device=seq_feat.device).unsqueeze(0)
        seq_feat = seq_feat + self.pos_embedding(pos_ids)

        seq_feat = self.norm(seq_feat)
        seq_feat = self.dropout(seq_feat)

        key_padding_mask = (mask == 0) if mask is not None else None

        # 修改：CNN 后接 TransformerEncoder，形成 DeepDTA + MolTrans 的序列编码器
        if self.transformer is not None:
            seq_feat = self.transformer(
                seq_feat,
                src_key_padding_mask=key_padding_mask
            )

        pooled = self.masked_mean(seq_feat, mask)
        return seq_feat, pooled


class DrugGraphEncoder(nn.Module):
    def __init__(self, in_dim=26, hidden_dim=256, dropout=0.2):
        super().__init__()
        if GCNConv is None:
            raise ImportError("torch_geometric is required for DrugGraphEncoder.")

        self.gcn1 = GCNConv(in_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.gcn3 = GCNConv(hidden_dim, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, graph):
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch

        x = F.relu(self.norm1(self.gcn1(x, edge_index)))
        x = self.dropout(x)

        residual = x
        x = F.relu(self.norm2(self.gcn2(x, edge_index)))
        x = self.dropout(x)

        x = F.relu(self.gcn3(x, edge_index) + residual)

        graph_feat = global_mean_pool(x, batch)  # [B, H]
        return graph_feat


class BiCrossAttention(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=4, dropout=0.2, return_attention=False):
        super().__init__()
        # 新增：是否返回注意力权重。默认 False，可减少显存占用。
        self.return_attention = return_attention
        self.drug_to_protein = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.protein_to_drug = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.norm_d = nn.LayerNorm(hidden_dim)
        self.norm_p = nn.LayerNorm(hidden_dim)

    def masked_mean(self, x, mask):
        if mask is None:
            return x.mean(dim=1)
        mask_f = mask.unsqueeze(-1).float()
        return (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)

    def forward(self, drug_seq, protein_seq, drug_mask=None, protein_mask=None):
        drug_key_padding = (drug_mask == 0) if drug_mask is not None else None
        protein_key_padding = (protein_mask == 0) if protein_mask is not None else None

        d2p, attn_d2p = self.drug_to_protein(
            query=drug_seq,
            key=protein_seq,
            value=protein_seq,
            key_padding_mask=protein_key_padding,
            need_weights=self.return_attention
        )

        p2d, attn_p2d = self.protein_to_drug(
            query=protein_seq,
            key=drug_seq,
            value=drug_seq,
            key_padding_mask=drug_key_padding,
            need_weights=self.return_attention
        )

        d2p = self.norm_d(d2p + drug_seq)
        p2d = self.norm_p(p2d + protein_seq)

        d_feat = self.masked_mean(d2p, drug_mask)
        p_feat = self.masked_mean(p2d, protein_mask)

        interaction_feat = self.proj(torch.cat([d_feat, p_feat], dim=-1))
        return interaction_feat, attn_d2p, attn_p2d


class GatedMultimodalFusion(nn.Module):
    def __init__(
        self,
        hidden_dim=256,
        num_modalities=4,
        dropout=0.2,
        use_residual_mlp=True,
    ):
        super().__init__()
        self.use_residual_mlp = use_residual_mlp
        self.gate = nn.Linear(hidden_dim * num_modalities, num_modalities)
        self.proj = None
        if self.use_residual_mlp:
            self.proj = nn.Sequential(
                nn.Linear(hidden_dim * num_modalities, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, features):
        concat = torch.cat(features, dim=-1)
        gates = torch.softmax(self.gate(concat), dim=-1)          # [B, M]

        stacked = torch.stack(features, dim=1)                    # [B, M, H]
        gated = (stacked * gates.unsqueeze(-1)).sum(dim=1)        # [B, H]

        projected = self.proj(concat) if self.proj is not None else torch.zeros_like(gated)
        fused = self.norm(gated + projected)                     # residual fusion
        return fused, gates


class GlobalAttentionResidualFusion(nn.Module):
    """Shared global modality weights with the full model's residual MLP retained."""
    def __init__(self, hidden_dim=256, num_modalities=4, dropout=0.2):
        super().__init__()
        self.global_logits = nn.Parameter(torch.zeros(num_modalities))
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * num_modalities, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, features):
        concat = torch.cat(features, dim=-1)
        stacked = torch.stack(features, dim=1)
        weights = torch.softmax(self.global_logits, dim=0)
        weighted = (stacked * weights.view(1, -1, 1)).sum(dim=1)
        fused = self.norm(weighted + self.proj(concat))
        gates = weights.view(1, -1).expand(stacked.size(0), -1)
        return fused, gates


class JCZSFiLM(nn.Module):
    def __init__(self, hidden_dim=256, num_roles=4):
        super().__init__()
        self.embedding = nn.Embedding(num_roles, hidden_dim)
        self.gamma = nn.Linear(hidden_dim, hidden_dim)
        self.beta = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, jc_code):
        jc_code = jc_code.clamp(min=0, max=3)
        role_feat = self.embedding(jc_code)

        gamma = torch.tanh(self.gamma(role_feat))
        beta = self.beta(role_feat)

        return x * (1.0 + gamma) + beta


class HarmoGATE(nn.Module):
    def __init__(
        self,
        drug_vocab_size=90,
        protein_vocab_size=27,
        graph_in_dim=26,
        embed_dim=128,
        hidden_dim=256,
        num_heads=4,
        dropout=0.2,
        use_jczs=True,
        use_pretrained_drug=False,
        use_pretrained_protein=False,
        pretrained_drug_dim=None,
        pretrained_protein_dim=None,
        use_graph=True,
        use_cross_attention=True,
        use_gated_fusion=True,
        use_transformer=True,
        fusion_mode=None,
        use_interaction_branch=True,
        use_fusion_residual=True,
    ):
        super().__init__()

        self.use_jczs = use_jczs
        self.use_pretrained_drug = use_pretrained_drug
        self.use_pretrained_protein = use_pretrained_protein
        self.use_graph = use_graph
        self.use_cross_attention = use_cross_attention
        self.use_interaction_branch = use_interaction_branch
        self.fusion_mode = fusion_mode or ("gated" if use_gated_fusion else "equal")
        if self.fusion_mode not in {
            "gated", "equal", "global_attention", "global_attention_residual"
        }:
            raise ValueError(f"Unknown fusion mode: {self.fusion_mode}")
        self.use_gated_fusion = self.fusion_mode == "gated"

        self.drug_seq_encoder = SequenceCNNTransformerEncoder(
           vocab_size=drug_vocab_size,
           max_len=128,
           embed_dim=embed_dim,
           hidden_dim=hidden_dim,
           num_heads=num_heads,
           dropout=dropout,
           transformer_layers=1,
           use_transformer=use_transformer,
        )

        self.protein_encoder = SequenceCNNTransformerEncoder(
           vocab_size=protein_vocab_size,
           max_len=1024,
           embed_dim=embed_dim,
           hidden_dim=hidden_dim,
           num_heads=num_heads,
           dropout=dropout,
           transformer_layers=1,
           use_transformer=use_transformer,
        )


        if self.use_graph:
            self.graph_encoder = DrugGraphEncoder(
                in_dim=graph_in_dim,
                hidden_dim=hidden_dim,
                dropout=dropout
            )
        else:
            self.graph_encoder = None

        self.cross_attention = BiCrossAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            return_attention=False
        )

        num_modalities = 2 + int(self.use_graph) + int(self.use_interaction_branch)
        if self.fusion_mode == "gated":
            self.fusion = GatedMultimodalFusion(
                hidden_dim=hidden_dim,
                num_modalities=num_modalities,
                dropout=dropout,
                use_residual_mlp=use_fusion_residual,
            )
        elif self.fusion_mode == "global_attention_residual":
            self.fusion = GlobalAttentionResidualFusion(
                hidden_dim=hidden_dim,
                num_modalities=num_modalities,
                dropout=dropout,
            )
        else:
            self.fusion = None
        if self.fusion_mode == "global_attention":
            self.global_fusion_logits = nn.Parameter(torch.zeros(num_modalities))
        else:
            self.register_parameter("global_fusion_logits", None)
        self.simple_fusion_norm = nn.LayerNorm(hidden_dim)

        if self.use_jczs:
            self.jczs_film = JCZSFiLM(hidden_dim=hidden_dim, num_roles=4)

        if use_pretrained_drug and pretrained_drug_dim is not None:
            self.pretrained_drug_proj = nn.Linear(pretrained_drug_dim, hidden_dim)
        else:
            self.pretrained_drug_proj = None

        if use_pretrained_protein and pretrained_protein_dim is not None:
            self.pretrained_protein_proj = nn.Linear(pretrained_protein_dim, hidden_dim)
        else:
            self.pretrained_protein_proj = None

        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim)
        )

        self.head_dti = nn.Linear(hidden_dim, 1)

        # 修改：DTA 是连续亲和力回归任务，单层 Linear 表达能力偏弱。
        # 使用更深的残差式 MLP 回归头，重点提升 RMSE/MAE/R2。
        self.head_dta = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        drug_seq,
        protein_seq,
        drug_mask,
        protein_mask,
        graph,
        jc_code=None,
        pretrained_drug=None,
        pretrained_protein=None,
        return_features=False
    ):
        drug_token_feat, drug_seq_feat = self.drug_seq_encoder(drug_seq, drug_mask)
        protein_token_feat, protein_feat = self.protein_encoder(protein_seq, protein_mask)
        if self.graph_encoder is not None:
            graph_feat = self.graph_encoder(graph)
        else:
            graph_feat = torch.zeros_like(drug_seq_feat)

        if self.pretrained_drug_proj is not None and pretrained_drug is not None:
            drug_seq_feat = drug_seq_feat + self.pretrained_drug_proj(pretrained_drug)

        if self.pretrained_protein_proj is not None and pretrained_protein is not None:
            protein_feat = protein_feat + self.pretrained_protein_proj(pretrained_protein)

        if self.use_interaction_branch and self.use_cross_attention:
            interaction_feat, attn_d2p, attn_p2d = self.cross_attention(
                drug_token_feat,
                protein_token_feat,
                drug_mask,
                protein_mask,
            )
        elif self.use_interaction_branch:
            interaction_feat = 0.5 * (drug_seq_feat + protein_feat)
            attn_d2p, attn_p2d = None, None
        else:
            interaction_feat = None
            attn_d2p, attn_p2d = None, None

        fusion_features = [drug_seq_feat]
        if self.use_graph:
            fusion_features.append(graph_feat)
        fusion_features.append(protein_feat)
        if self.use_interaction_branch:
            fusion_features.append(interaction_feat)

        if self.fusion_mode in {"gated", "global_attention_residual"}:
            fused, modality_gates = self.fusion(fusion_features)
        elif self.fusion_mode == "equal":
            stacked = torch.stack(fusion_features, dim=1)
            fused = self.simple_fusion_norm(stacked.mean(dim=1))
            modality_gates = torch.full(
                (stacked.size(0), stacked.size(1)),
                1.0 / stacked.size(1),
                device=stacked.device,
                dtype=stacked.dtype,
            )
        else:
            stacked = torch.stack(fusion_features, dim=1)
            weights = torch.softmax(self.global_fusion_logits, dim=0)
            fused = self.simple_fusion_norm(
                (stacked * weights.view(1, -1, 1)).sum(dim=1)
            )
            modality_gates = weights.view(1, -1).expand(stacked.size(0), -1)

        if self.use_jczs and jc_code is not None:
            fused = self.jczs_film(fused, jc_code)

        feat = self.shared(fused)

        dti_logit = self.head_dti(feat).squeeze(-1)
        dta_pred = self.head_dta(feat).squeeze(-1)

        outputs = {
            "dti_logit": dti_logit,
            "dta_pred": dta_pred,
            "attn_d2p": attn_d2p,
            "attn_p2d": attn_p2d,
            "modality_gates": modality_gates,
        }

        if return_features:
            outputs.update({
                "drug_seq_feat": drug_seq_feat,
                "drug_graph_feat": graph_feat,
                "protein_feat": protein_feat,
                "interaction_feat": interaction_feat,
                "fused_feat": feat,
            })

        return outputs
