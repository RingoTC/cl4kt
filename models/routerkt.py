import torch
import torch.nn as nn
from torch.nn import Module, Embedding, Linear, LayerNorm, Dropout, BCELoss, ModuleList, Parameter, ReLU
from torch.nn.functional import binary_cross_entropy, cross_entropy
import numpy as np
import os
from datetime import datetime
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
elif torch.backends.mps.is_available():
    torch.set_default_tensor_type(torch.FloatTensor)

class RouterKT(Module):
    def __init__(
        self,
        num_skills,
        num_questions,
        seq_len,
        embedding_size,
        num_blocks,
        kq_same,
        model_type="routerkt",
        num_attn_heads=8,
        final_fc_dim=512,
        d_ff=2048,
        l2=1e-5,
        dropout=0.2,
        separate_qr=False,
        num_shared_heads=1,
        num_selected_heads=2,
        balance_loss_weight=0.001,
        routing_mode="dynamic",
        **kwargs
    ):
        super(RouterKT, self).__init__()

        """
        params:
            num_skills: # of skills
            num_questions: # of questions
            embedding_size: embedding dim
            num_blocks: # of attn blocks
            seq_len: max length of sequenc
            kq_same: key랑 query랑 같은지
            num_attn_heads: number of heads if multi-headed attention
            final_fc_dim: dimension of final fully connected net before prediction
            d_ff: dimension for fully connected net inside the basic block
            
        """
        self.num_skills = num_skills
        self.num_questions = num_questions
        self.embedding_size = embedding_size
        self.num_blocks = num_blocks
        self.seq_len = seq_len
        self.kq_same = kq_same
        print("kq_same", kq_same)
        self.model_type = model_type
        self.num_attn_heads = num_attn_heads
        self.final_fc_dim = final_fc_dim
        self.d_ff = d_ff
        self.l2 = l2
        self.dropout = dropout
        self.separate_qr = separate_qr
        self.balance_loss_weight = balance_loss_weight
        self.num_attn_heads = num_attn_heads
        self.num_shared_heads = num_shared_heads
        self.num_selected_heads = num_selected_heads

        if self.num_questions > 0:
            self.difficult_param = Embedding(
                self.num_questions + 1, 1, padding_idx=0
            )  # /mu_{q_t} parameter
            self.q_embed_diff = Embedding(
                self.num_skills + 1, self.embedding_size, padding_idx=0
            )  # d_{c_t}
            self.qr_embed_diff = Embedding(
                2 * self.num_skills + 1, self.embedding_size, padding_idx=0
            )  # f_{(c_t, r_t)} or h_{r_t}
        self.q_embed = Embedding(
            self.num_skills, self.embedding_size, padding_idx=0
        )  # c_{c_t}
        if self.separate_qr:
            self.qr_embed = Embedding(
                2 * self.num_skills, self.embedding_size, padding_idx=0
            )  # e_{(c_t, r_t)}
        else:
            self.qr_embed = Embedding(2 * self.num_skills + 1, self.embedding_size, padding_idx=0)

        self.model = RouterKTArchitecture(
            n_question=self.num_skills,
            n_blocks=self.num_blocks,
            n_heads=self.num_attn_heads,
            dropout=self.dropout,
            d_model=self.embedding_size,
            d_feature=self.embedding_size // self.num_attn_heads,
            d_ff=self.d_ff,
            kq_same=self.kq_same,
            n_shared_heads=self.num_shared_heads,  # Pass number of shared heads
            n_selected_heads=self.num_selected_heads,
        )

        self.out = nn.Sequential(
            nn.Linear(2 * self.embedding_size, self.final_fc_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.final_fc_dim, self.final_fc_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            Linear(self.final_fc_dim // 2, 1),
        )
        self.reset()
        self.loss_fn = nn.BCELoss(reduction="mean")

    def reset(self):
        """Reset parameters initialization."""
        for p in self.parameters():
            if p.size(0) == self.num_questions + 1 and self.num_questions > 0:
                torch.nn.init.constant_(p, 0.)
            elif p.dim() > 1:  # 添加对其他参数的初始化
                torch.nn.init.xavier_uniform_(p)

    def forward(self, feed_dict):
        q = feed_dict["skills"]
        r = feed_dict["responses"]
        attention_mask = feed_dict["attention_mask"]
        masked_r = r * (r > -1).long()
        pid_data = feed_dict["questions"]

        qr = q + self.num_skills * masked_r

        q_embed_data = self.q_embed(q)  # c_{c_t}: [batch_size, seq_len, embedding_size]
        qr_embed_data = self.qr_embed(
            qr
        )  # f_{(c_t, r_t)}: [batch_size, seq_len, d_model]

        if self.num_questions > 0:
            q_embed_diff_data = self.q_embed_diff(q)  # d_{c_t}: variation vector
            pid_embed_data = self.difficult_param(pid_data)  # \mu_{q_t}
            q_embed_data = (
                q_embed_data + pid_embed_data * q_embed_diff_data
            )  # x_t = c_{c_t} + \mu_{q_t} + d_{c_t}
            qr_embed_diff_data = self.qr_embed_diff(qr)  # f_{(c_t, r_t)} or h_{r_t}

            if self.separate_qr:
                qr_embed_data = qr_embed_data + pid_embed_data * qr_embed_diff_data
            else:
                # y_t = e_{(c_t, r_t)} + \mu_{q_t} * f_{(c_t, r_t)}
                # , where e_{(c_t, r_t)} = c_{c_t} + g_{r_t}
                # f_{(c_t, r_t)} = f_{(c_t, r_t)} + d_{c_t}
                # e_{(c_t, r_t)} + \mu_{q_t} * (h_{r_t} + d_{c_t})
                qr_embed_data = qr_embed_data + pid_embed_data * (
                    qr_embed_diff_data + q_embed_diff_data
                )

            c_reg_loss = torch.mean(pid_embed_data ** 2.0) * self.l2
        else:
            c_reg_loss = 0

        pooled_ques_score = (self.q_embed(q) * attention_mask.unsqueeze(-1)).sum(
            1
        ) / attention_mask.sum(-1).unsqueeze(-1)
        pooled_inter_score = (qr_embed_data * attention_mask.unsqueeze(-1)).sum(
            1
        ) / attention_mask.sum(-1).unsqueeze(-1)

        # [batch_size, seq_len, d_model]
        # pass to the decoder
        # output shape [batch_size, seq_len, d_model or d_model//2]
        # d_output is h_t
        d_output, attn = self.model(q_embed_data, qr_embed_data)  # 211x512

        concat_q = torch.cat([d_output, q_embed_data], dim=-1)  # concat([h_t, x_t])
        output = torch.sigmoid(self.out(concat_q)).squeeze()

        if self.training:
            out_dict = {
                "pred": output[:, 1:],
                "true": r[:, 1:].float(),
                "c_reg_loss": c_reg_loss,
            }
        else:
            out_dict = {
                "pred": output[:, 1:],
                "true": r[:, 1:].float(),
                "c_reg_loss": c_reg_loss,
                "q_embed": pooled_ques_score,
                "qr_embed": pooled_inter_score,
            }
        return out_dict

    def loss(self, feed_dict, out_dict):
        pred = out_dict["pred"].flatten()
        true = out_dict["true"].flatten()
        c_reg_loss = out_dict["c_reg_loss"]
        mask = true > -1
        balance_loss = self.balance_loss_weight * self.model.get_balance_loss()
        loss = self.loss_fn(pred[mask], true[mask])
        return loss + c_reg_loss + balance_loss, len(pred[mask]), true[mask].sum().item()

class RouterKTArchitecture(Module):
    def __init__(self, n_question, n_blocks, d_model, d_feature,
                 d_ff, n_heads, n_shared_heads, n_selected_heads, dropout,
                 kq_same):
        super().__init__()
        self.d_model = d_model
        
        # Transformer blocks with MoH attention for knowledge encoder
        self.blocks_1 = ModuleList(
            [
                RouterTransformerLayer(
                    d_model=d_model,
                    d_feature=d_feature,
                    d_ff=d_ff,
                    dropout=dropout,
                    n_heads=n_heads,
                    n_shared_heads=n_shared_heads,
                    n_selected_heads=n_selected_heads,
                    kq_same=kq_same
                )
                for _ in range(n_blocks)
            ]
        )
        
        # Transformer blocks with MoH attention for question encoder
        self.blocks_2 = ModuleList(
            [
                RouterTransformerLayer(
                    d_model=d_model,
                    d_feature=d_feature,
                    d_ff=d_ff,
                    dropout=dropout,
                    n_heads=n_heads,
                    n_shared_heads=n_shared_heads,
                    n_selected_heads=n_selected_heads,
                    kq_same=kq_same
                )
                for _ in range(n_blocks * 2)
            ]
        )
        
    def forward(self, q_embed_data, qa_embed_data, diff=None, r=None):
        # Initialize variables
        seqlen, batch_size = q_embed_data.size(1), q_embed_data.size(0)
        
        y = qa_embed_data
        x = q_embed_data
        
        # Knowledge encoder
        for block in self.blocks_1:
            y, _ = block(mask=1, query=y, key=y, values=y, diff=diff, response=r, q4router=x)
            
        # Question encoder
        flag_first = True
        for block in self.blocks_2:
            if flag_first:
                # x can see both current and past information
                x, _ = block(mask=1, query=x, key=x, values=x, diff=diff, response=r, apply_pos=False, q4router=x)
                flag_first = False
            else:# dont peek current response
                # knoweldge retriever
                # h can see past only
                x, attn = block(mask=0, query=x, key=x, values=y, diff=diff, response=r, apply_pos=True, q4router=x)
                flag_first = True
                
        return x, attn
    
    def get_balance_loss(self):
        balance_loss = 0
        for block in self.blocks_1:
            balance_loss += block.attn.get_balance_loss()
        for block in self.blocks_2:
            balance_loss += block.attn.get_balance_loss()
        return balance_loss

class RouterTransformerLayer(Module):
    def __init__(self, d_model, d_feature, d_ff, dropout, n_heads, 
                 n_shared_heads, n_selected_heads, kq_same):
        super(RouterTransformerLayer, self).__init__()
        
        # Pass parameters to MoHAttention
        self.attn = MoHAttention(
            d_model=d_model,
            d_feature=d_feature,
            n_heads=n_heads,
            n_shared_heads=n_shared_heads,
            n_selected_heads=n_selected_heads,
            dropout=dropout,
            kq_same=kq_same
        )
        
        # Layer norm and dropout layers
        self.layer_norm1 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)

        # Feed forward layers
        self.linear1 = Linear(d_model, d_ff)
        self.activation = ReLU()
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(d_ff, d_model)

        self.layer_norm2 = LayerNorm(d_model)
        self.dropout2 = Dropout(dropout)
        
    def forward(self, mask, query, key, values, diff=None, response=None, apply_pos=True, q4router=None):
        batch_size, seqlen = query.size(0), query.size(1)
        
        # Create attention mask based on the mask parameter
        device = query.get_device()
        nopeek_mask = np.triu(np.ones((1, 1, seqlen, seqlen)), k=mask).astype("uint8")
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(device)
        
        # Apply MoH attention
        # query2 = self.attn(query, key, values, src_mask, q4router)

        if mask == 0:
            query2 = self.attn(query, key, values, mask=src_mask, q4router=q4router)
        elif mask == 1:
            query2 = self.attn(query, key, values, mask=src_mask, q4router=q4router)
        else:  # mask == 2
            raise NotImplementedError


        # First residual connection and layer norm
        query = query + self.dropout1(query2)
        query = self.layer_norm1(query)

        if apply_pos:
            # Feed forward network
            query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
            query = query + self.dropout2(query2)
            query = self.layer_norm2(query)

        return query, self.attn.get_balance_loss()

class MoHAttention(Module):
    def __init__(self, d_model, d_feature, n_heads, n_shared_heads, 
                 n_selected_heads, dropout, kq_same):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_feature
        self.h = n_heads
        self.h_shared = n_shared_heads
        self.h_selected = n_selected_heads
        self.kq_same = kq_same
        
        # Linear layers for Q, K, V
        self.q_linear = Linear(d_model, d_model)
        if not kq_same:
            self.k_linear = Linear(d_model, d_model)
        self.v_linear = Linear(d_model, d_model)
        
        # Router for dynamic heads
        self.wg = nn.Linear(d_model, n_heads - n_shared_heads, bias=False)
            
        self.dropout = Dropout(dropout)
        self.out = Linear(d_model, d_model)
        
        # Track routing statistics for load balancing
        self.register_buffer('head_selections', torch.zeros(n_heads - n_shared_heads))
        self.register_buffer('head_routing_probs', torch.zeros(n_heads - n_shared_heads))
        
    def get_balance_loss(self):
        # Calculate load balance loss for dynamic heads
        f = self.head_selections / (self.head_selections.sum() + 1e-5)
        P = self.head_routing_probs / (self.head_routing_probs.sum() + 1e-5)
        balance_loss = (f * P).sum()
        return balance_loss
        
    def forward(self, q, k, v, mask, q4router):
        bs = q.size(0)
        seq_len = q.size(1)
        
        # Linear projections
        q = self.q_linear(q)  # [bs, seq_len, d_model]
        if self.kq_same:
            k = q
        else:
            k = self.k_linear(k)
        v = self.v_linear(v)
        
        # Reshape for attention computation
        q = q.view(bs, -1, self.h, self.d_k).transpose(1, 2)  # [bs, h, seq_len, d_k]
        k = k.view(bs, -1, self.h, self.d_k).transpose(1, 2)
        v = v.view(bs, -1, self.h, self.d_k).transpose(1, 2)
        
        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)  # [bs, h, seq_len, seq_len]
            
        # scores = scores.masked_fill(mask == 0, -1e9)
            
        # Reshape q4router to match expected dimensions
        # Always use the question information for routing
        q_for_routing = q4router.view(bs, seq_len, self.h, self.d_k)  # [bs, seq_len, h, d_k]
        q_for_routing = q_for_routing.permute(0, 2, 1, 3)  # [bs, h, seq_len, d_k]
        q_for_routing = q_for_routing.reshape(bs * seq_len, self.h * self.d_k)

        # Use learned routing weights
        logits = self.wg(q_for_routing)  # [bs*seq_len, n_dynamic_heads]
        gates = F.softmax(logits, dim=1)  # [bs*seq_len, n_dynamic_heads]
        
        # Select top-k heads
        _, indices = torch.topk(gates, k=self.h_selected, dim=1)
        dynamic_mask = torch.zeros_like(gates).scatter_(1, indices, 1.0)
        
        self.dynamic_scores = gates * dynamic_mask
        
        # Update routing statistics
        self.head_routing_probs = gates.mean(dim=0)
        self.head_selections = dynamic_mask.sum(dim=0)
        
        # Handle shared heads routing
        # All shared heads have equal weight of 1.0
        self.shared_scores = torch.ones(bs, seq_len, self.h_shared).to(q.device)
        
        dynamic_scores_reshaped = self.dynamic_scores.view(bs, seq_len, -1)
        routing_mask = torch.zeros(bs, seq_len, self.h).to(q.device)
        routing_mask[:, :, :self.h_shared] = 1.0  # Shared heads always active
        routing_mask[:, :, self.h_shared:] = dynamic_scores_reshaped  # Add dynamic head weights
        
        # Reshape routing mask to match attention dimensions [bs, h, seq_len, 1]
        routing_mask = routing_mask.mean(dim=1).unsqueeze(-1).unsqueeze(-1)
        
        # Apply attention
        
        scores = scores.masked_fill(mask == 0, -1e9)
        scores = self.dropout(torch.softmax(scores, dim=-1))
        
        output = torch.matmul(scores, v)  # [bs, h, seq_len, d_k]
        
        # Apply routing mask
        output = output * routing_mask
        
        # Combine heads
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        
        return self.out(output)