import copy
import math
import os
import random
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn
from typing import Type

class MaskHead(nn.Module):
    def __init__(self, inplanes, conv_dims=256, num_upconv=2):
        super(MaskHead, self).__init__()

        self.upconvs = []
        for k in range(num_upconv):
            upconv = nn.Sequential(
                nn.Conv2d(
                    inplanes if k == 0 else conv_dims,
                    conv_dims,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                nn.BatchNorm2d(conv_dims),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    conv_dims if num_upconv > 0 else inplanes,
                    conv_dims,
                    kernel_size=2,
                    stride=2,
                    padding=0,
                )
            )
            self.add_module("upconv{}".format(k + 1), upconv)
            self.upconvs.append(upconv)
                    
        self.predictor = nn.Conv2d(
                conv_dims,
                2,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        

    def forward(self, x):
        for layer in self.upconvs:
            x = layer(x)

        return self.predictor(x)
    

class OneNIP(nn.Module):
    def __init__(
        self,
        inplanes=384,
        feature_size=[16, 16],
        feature_jitter={'prob': 1.0, 'scale': 20.0},
        neighbor_mask={'mask': [True, True, True], 'neighbor_size': [8, 8]}, # default [7, 7]
        hidden_dim=256,
        pos_embed_type='learned',
        image_size=224,
        **kwargs,
    ):
        super().__init__()

        self.feature_size = feature_size
        self.num_queries = feature_size[0] * feature_size[1]
        self.feature_jitter = feature_jitter
        self.pos_embed = build_position_embedding(
            pos_embed_type, feature_size, hidden_dim * 2
        )

        self.transformer = Transformer(
            hidden_dim, feature_size, neighbor_mask, **kwargs
        )
        self.input_proj_prompt = nn.Linear(inplanes, hidden_dim)
        self.input_proj_query = nn.Linear(inplanes, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, inplanes)
        
        # supervised refiner
        self.seghead = MaskHead(inplanes, 128, 2) 

        self.segupsample = nn.UpsamplingBilinear2d(size=image_size)


    def add_jitter(self, feature_tokens, scale, prob):
        if random.uniform(0, 1) <= prob:
            num_tokens, batch_size, dim_channel = feature_tokens.shape
            feature_norms = (
                feature_tokens.norm(dim=2).unsqueeze(2) / dim_channel
            )  # (H x W) x B x 1
            jitter = torch.randn((num_tokens, batch_size, dim_channel)).cuda()
            jitter = jitter * feature_norms * scale
            feature_tokens = feature_tokens + jitter
        return feature_tokens

    def forward(self, input):
        
        # query feat: 正常图像的特征
        # synthetic feat: 合成异常图像的特征
        # prompt feat: 提示正常图像的特征
        query_feat = input["pseudo_feature"]
        prompt_feat = input["prompt_feature"]

        # query_batch_size = query_feat.shape[0]

        # query_prompt_feat = torch.cat([query_feat, input["prompt_feature"]], 0) # torch.Size([16, 384, 16, 16])

        # query_prompt_tokens = rearrange(
        #     query_prompt_feat, "b c h w -> (h w) b c"
        # )  # (H x W) x B x C # torch.Size([256, 16, 384])
        
        query_tokens = rearrange(query_feat, "b c h w -> (h w) b c")
        prompt_tokens = rearrange(prompt_feat, "b c h w -> (h w) b c")
     
        if self.training and self.feature_jitter:
            query_tokens = self.add_jitter(
                query_tokens, self.feature_jitter['scale'], self.feature_jitter['prob']
            )
            prompt_tokens = self.add_jitter(
                prompt_tokens, self.feature_jitter['scale'], self.feature_jitter['prob']
            )

        query_tokens = self.input_proj_query(query_tokens)  # (H x W) x B x C
        prompt_tokens = self.input_proj_prompt(prompt_tokens) # (H x W) x B x C

        pos_embed = self.pos_embed(query_tokens)  # (H x W) x C
        # pos_embed = pos_embed.repeat(1, 2)

        encoder_tokens, decoder_tokens = self.transformer(
            query_tokens, prompt_tokens, pos_embed
        )  # (H x W) x B x C
        

        rec_query_tokens = self.output_proj(decoder_tokens)  # (H x W) x B x C
        rec_query_feat = rearrange(
            rec_query_tokens, "(h w) b c -> b c h w", h=self.feature_size[0]
        )  # B x C X H x W

        latent = rearrange(
            encoder_tokens, "(h w) b c -> b c h w", h=self.feature_size[0]
        )
        
        ref_logit = self.seghead((rec_query_feat - query_feat) ** 2) #2B x c x h/2 x w/2
        ref_logit = self.segupsample(ref_logit)
        ref_pred = ref_logit.sigmoid() 
        
        return {
            "rec_feat": rec_query_feat,
            "ref_pred": ref_pred,
            "latent": latent,
        }


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        dropout: 0.1,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class Transformer(nn.Module):
    def __init__(
        self,
        hidden_dim=256,
        feature_size=[16, 16],
        neighbor_mask={'mask': [True, True, True],
                       'neighbor_size': [7, 7]},
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
    ):
        super().__init__()
        self.feature_size = feature_size
        self.neighbor_mask = neighbor_mask

        encoder_layer = TransformerEncoderLayer(
            hidden_dim * 2, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        encoder_norm = nn.LayerNorm(hidden_dim) if normalize_before else None
        self.encoder = TransformerEncoder(
            encoder_layer, int(num_encoder_layers), encoder_norm
        )

        decoder_layer = TransformerDecoderLayer(
            hidden_dim,
            feature_size,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            normalize_before,
        )
        decoder_norm = nn.LayerNorm(hidden_dim)
        self.decoder = TransformerDecoder(
            decoder_layer,
            int(num_decoder_layers),
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        
        self.query_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.prompt_proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def generate_mask(self, feature_size, neighbor_size):
        """
        Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        h, w = feature_size
        hm, wm = neighbor_size
        mask = torch.ones(h, w, h, w)
        for idx_h1 in range(h):
            for idx_w1 in range(w):
                idx_h2_start = max(idx_h1 - hm // 2, 0)
                idx_h2_end = min(idx_h1 + hm // 2 + 1, h)
                idx_w2_start = max(idx_w1 - wm // 2, 0)
                idx_w2_end = min(idx_w1 + wm // 2 + 1, w)
                mask[
                    idx_h1, idx_w1, idx_h2_start:idx_h2_end, idx_w2_start:idx_w2_end
                ] = 0
        mask = mask.view(h * w, h * w)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
            .cuda()
        )
        return mask

    def forward(self, query, prompt, pos_embed):
        
        # pos_embed torch.Size([256, 256])

        query_batch_size = query.shape[1] # 8
        prompt_batch_size = prompt.shape[1] # 8
        # query_prompt_batch_size = query_batch_size + prompt_batch_size
        query_prompt = torch.cat((query, prompt), dim=2)

        pos_embed = torch.cat(
            [pos_embed.unsqueeze(1)] * query_batch_size, dim=1
        )  # (H X W) x B x C

        if self.neighbor_mask:
            mask = self.generate_mask(
                self.feature_size, self.neighbor_mask['neighbor_size']
            )
            mask_enc = mask if self.neighbor_mask['mask'][0] else None
            mask_dec = mask if self.neighbor_mask['mask'][1] else None
        else:
            mask_enc = mask_dec = None
            
        # torch.Size([256, 16, 256]) torch.Size([256, 16, 256])
        encoder_tokens = self.encoder(
            query_prompt, mask=mask_enc, pos=pos_embed
        )  # (H X W) x B x C
        # torch.Size([256, 16, 256])
        
        # query_tokens = encoder_tokens[:, :, :self.hidden_dim]
        # prompt_tokens = encoder_tokens[:, :, self.hidden_dim:]
        query_tokens = self.query_proj(encoder_tokens)
        prompt_tokens = self.prompt_proj(encoder_tokens)
        pos_decoder = pos_embed[:, :, :self.hidden_dim]
    
        # print(prompt_tokens.shape, query_tokens.shape, pos_decoder.shape)
        # torch.Size([256, 8, 256]) torch.Size([256, 8, 256]) torch.Size([256, 8, 256])

        decoder_tokens = self.decoder(
            prompt_tokens,
            query_tokens,
            att_mask=mask_dec,
            pos=pos_decoder,
        )  # (H X W) x B x C

        return encoder_tokens, decoder_tokens


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        output = src

        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.final_attn_token_to_image = nn.MultiheadAttention(256, 8, dropout=0.1)
        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        queries,
        keys,
        att_mask: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
    
        intermediate = []

        for layer in self.layers:
            queries, keys  = layer(
                queries,
                keys,
                att_mask=att_mask,
                padding_mask=padding_mask,
                pos=pos,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(queries))
        
        # Apply the final attention layer from the points to the image
        q = queries + pos
        k = keys + pos
        attn_out = self.final_attn_token_to_image(q, k, value=keys)[0]
        queries = queries + self.dropout(attn_out)
        output = queries
       
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_dim,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        
        q = k = self.with_pos_embed(src, pos)
        
        src2 = self.self_attn(
            q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        
        src = src + self.dropout1(src2)
        
        src = self.norm1(src)
        
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        
        src = src + self.dropout2(src2)
        
        src = self.norm2(src)
        
        return src

    def forward_pre(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(
            q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_dim,
        feature_size,
        nhead,
        dim_feedforward,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()

        self.cross_attn_prompt_to_image = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        self.cross_attn_image_to_prompt = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)

        self.mlp = MLPBlock(hidden_dim, dim_feedforward, dropout, _get_activation_fn(activation))

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        queries,
        keys,
        att_mask: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        
        # Cross attention block, tokens attending to image embedding
        q = queries + pos
        k = keys + pos

        attn_out = self.cross_attn_prompt_to_image(
            q, 
            k, 
            value=keys, 
            attn_mask=att_mask, 
            key_padding_mask=padding_mask
            )[0]

        queries = queries + self.dropout1(attn_out)
        queries = self.norm1(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + self.dropout2(mlp_out)
        queries = self.norm2(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + pos
        k = keys + pos

        attn_out = self.cross_attn_image_to_prompt(
            k, 
            q, 
            value=queries, 
            attn_mask=att_mask, 
            key_padding_mask=padding_mask
            )[0]

        keys = keys + self.dropout3(attn_out)
        keys = self.norm3(keys)

        return queries, keys

    def forward_pre(
        self,
        queries,
        keys,
        att_mask: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        _, batch_size, _ = queries.shape

        # Cross attention block, tokens attending to image embedding
        queries = self.norm1(queries)
        q = queries + pos
        k = keys + pos

        attn_out = self.cross_attn_prompt_to_image(
            q=q, 
            k=k, 
            v=keys, 
            attn_mask=att_mask, 
            key_padding_mask=padding_mask
            )

        queries = queries + self.dropout1(attn_out)
        
        # MLP block
        queries = self.norm1(queries)
        mlp_out = self.mlp(queries)
        queries = queries + self.dropout2(mlp_out)
        queries = self.norm2(queries)

        # Cross attention block, image embedding attending to tokens
        queries = self.norm1(queries)
        q = queries + pos
        k = keys + pos

        attn_out = self.cross_attn_image_to_prompt(
            q=k, 
            k=q, 
            v=queries, 
            attn_mask=att_mask, 
            key_padding_mask=padding_mask
            )

        keys = keys + self.dropout3(attn_out)

        return queries, keys

    def forward(
        self,
        queries,
        keys,
        att_mask: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                queries,
                keys,
                att_mask,
                padding_mask,
                pos,
            )
        return self.forward_post(
            queries,
            keys,
            att_mask,
            padding_mask,
            pos,
        )


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(
        self,
        feature_size,
        num_pos_feats=128,
        temperature=10000,
        normalize=False,
        scale=None,
    ):
        super().__init__()
        self.feature_size = feature_size
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor):
        not_mask = torch.ones((self.feature_size[0], self.feature_size[1]))  # H x W
        y_embed = not_mask.cumsum(0, dtype=torch.float32)
        x_embed = not_mask.cumsum(1, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[-1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3
        ).flatten(2)
        pos_y = torch.stack(
            (pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3
        ).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=2).flatten(0, 1)  # (H X W) X C
        return pos.to(tensor.device)


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, feature_size, num_pos_feats=128):
        super().__init__()
        self.feature_size = feature_size  # H, W
        self.row_embed = nn.Embedding(feature_size[0], num_pos_feats)
        self.col_embed = nn.Embedding(feature_size[1], num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor):
        i = torch.arange(self.feature_size[1], device=tensor.device)  # W
        j = torch.arange(self.feature_size[0], device=tensor.device)  # H
        x_emb = self.col_embed(i)  # W x C // 2
        y_emb = self.row_embed(j)  # H x C // 2
        pos = torch.cat(
            [
                torch.cat(
                    [x_emb.unsqueeze(0)] * self.feature_size[0], dim=0
                ),  # H x W x C // 2
                torch.cat(
                    [y_emb.unsqueeze(1)] * self.feature_size[1], dim=1
                ),  # H x W x C // 2
            ],
            dim=-1,
        ).flatten(
            0, 1
        )  # (H X W) X C
        return pos


def build_position_embedding(pos_embed_type, feature_size, hidden_dim):
    if pos_embed_type in ("v2", "sine"):
        # TODO find a better way of exposing other arguments
        pos_embed = PositionEmbeddingSine(feature_size, hidden_dim // 2, normalize=True)
    elif pos_embed_type in ("v3", "learned"):
        pos_embed = PositionEmbeddingLearned(feature_size, hidden_dim // 2)
    else:
        raise ValueError(f"not supported {pos_embed_type}")
    return pos_embed


if __name__ == '__main__':
    inputs = {}
    inputs["feature"] = torch.randn(4, 272, 14, 14).cuda()
    inputs["pseudo_feature"] = torch.randn(4, 272, 14, 14).cuda()
    inputs["prompt_feature"] = torch.randn(4, 272, 14, 14).cuda()
    
    model = OneNIP().cuda()
    
    # model.train()
    model.eval()
    
    outputs = model(inputs)
    
    for key in outputs.keys():
        print(key, outputs[key].shape)
        
    ''' train
    torch.Size([8, 272, 14, 14])
    torch.Size([8, 272, 14, 14])
    torch.Size([12, 256, 14, 14])
    torch.Size([8, 1, 224, 224])
    '''
    

    ''' eval
    torch.Size([4, 272, 14, 14])
    torch.Size([4, 272, 14, 14])
    torch.Size([8, 256, 14, 14])
    torch.Size([4, 1, 224, 224])
    '''