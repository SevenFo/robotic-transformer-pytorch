from __future__ import annotations

import torch
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from torch import nn, einsum, Tensor

from typing import Callable
from beartype import beartype

from einops import pack, unpack, repeat, reduce, rearrange
from einops.layers.torch import Rearrange, Reduce

from functools import partial

from classifier_free_guidance_pytorch import (
    TextConditioner,
    AttentionTextConditioner,
    classifier_free_guidance,
)

# helpers


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cast_tuple(val, length=1):
    return val if isinstance(val, tuple) else ((val,) * length)


def pack_one(x, pattern):
    return pack([x], pattern)


def unpack_one(x, ps, pattern):
    return unpack(x, ps, pattern)[0]


# sinusoidal positions


def posemb_sincos_1d(seq, dim, temperature=10000, device=None, dtype=torch.float32):
    n = torch.arange(seq, device=device)
    omega = torch.arange(dim // 2, device=device) / (dim // 2 - 1)
    omega = 1.0 / (temperature**omega)

    n = n[:, None] * omega[None, :]
    pos_emb = torch.cat((n.sin(), n.cos()), dim=1)
    return pos_emb.type(dtype)


# helper classes


class Residual(Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class LayerNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class FeedForward(Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        self.norm = LayerNorm(dim)

        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, cond_fn=None):
        x = self.norm(x)

        if exists(cond_fn):
            # adaptive layernorm
            x = cond_fn(x)

        return self.net(x)


# MBConv


class SqueezeExcitation(Module):
    def __init__(self, dim, shrinkage_rate=0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = nn.Sequential(
            Reduce("b c h w -> b c", "mean"),
            nn.Linear(dim, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim, bias=False),
            nn.Sigmoid(),
            Rearrange("b c -> b c 1 1"),
        )

    def forward(self, x):
        return x * self.gate(x)


class MBConvResidual(Module):
    def __init__(self, fn, dropout=0.0):
        super().__init__()
        self.fn = fn
        self.dropsample = Dropsample(dropout)

    def forward(self, x):
        out = self.fn(x)
        out = self.dropsample(out)
        return out + x


class Dropsample(Module):
    def __init__(self, prob=0):
        super().__init__()
        self.prob = prob

    def forward(self, x):
        device = x.device

        if self.prob == 0.0 or (not self.training):
            return x

        keep_mask = (
            torch.FloatTensor((x.shape[0], 1, 1, 1), device=device).uniform_()
            > self.prob
        )
        return x * keep_mask / (1 - self.prob)


def MBConv(
    dim_in, dim_out, *, downsample, expansion_rate=4, shrinkage_rate=0.25, dropout=0.0
):
    hidden_dim = int(expansion_rate * dim_out)
    stride = 2 if downsample else 1

    net = nn.Sequential(
        nn.Conv2d(dim_in, hidden_dim, 1),
        nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        nn.Conv2d(
            hidden_dim, hidden_dim, 3, stride=stride, padding=1, groups=hidden_dim
        ),
        nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        SqueezeExcitation(hidden_dim, shrinkage_rate=shrinkage_rate),
        nn.Conv2d(hidden_dim, dim_out, 1),
        nn.BatchNorm2d(dim_out),
    )

    if dim_in == dim_out and not downsample:
        net = MBConvResidual(net, dropout=dropout)

    return net


# attention related classes


class WindowAttention(Module):
    """窗口注意力，类似于卷积操作，卷积核在输入特征图上滑动，这里相对应是“注意力核”在输入特征图上滑动
    每个窗口的一个像素的所有通道作为一个token，flatten后作为一个token序列进行自注意力计算
    注意位置编码是相对位置编码，和TransformerAttention不同
    """

    def __init__(
        self,
        dim,  # 输入特征的维度
        dim_head=32,  # 每个注意力头的维度
        dropout=0.0,  # Dropout概率
        window_size=7,  # 局部窗口的尺寸（用于相对位置编码）
        num_mem_kv=4,  # 记忆键值对的数量
    ):
        super().__init__()
        assert (dim % dim_head) == 0, (
            "dimension should be divisible by dimension per head"
        )

        self.norm = LayerNorm(dim)  # 对输入特征进行归一化

        # attention = softmax((Q)(K)^T / sqrt(dim_head)) * V
        self.heads = dim // dim_head  # 注意力头的数量
        self.scale = dim_head**-0.5  # 缩放因子，稳定softmax

        self.to_qkv = nn.Linear(
            dim, dim * 3, bias=False
        )  # project x to query, key at once, 先投影再分割为Q,K,V

        # 可学习的记忆键值对（全局上下文记忆）
        # 第一个维度表示k,v，第二个维度表示头数，第三个维度表示记忆键值对的数量，第四个维度表示每个头的维度
        # 相当于是额外的全局token，不由外部输入，而是随机生成的，并且由于参与了sim的计算，因此是可学习的
        self.mem_kv = nn.Parameter(torch.randn(2, self.heads, num_mem_kv, dim_head))

        self.attend = nn.Sequential(
            nn.Softmax(
                dim=-1
            ),  # 最后一个维度进行softmax，因为最后一个维度是权重，需要归一化
            nn.Dropout(dropout),
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias=False),  # 用于将多头注意力的输出映射回原始维度
            nn.Dropout(dropout),
        )

        # relative positional bias
        # 位置编码嵌入表：每个相对位置对应一个可学习的向量
        # 每个head的位置编码是不一样的
        self.rel_pos_bias = nn.Embedding((2 * window_size - 1) ** 2, self.heads)

        # 生成相对位置编码（提前计算）
        pos = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(pos, pos, indexing="ij"))  # shape: 2,pos,pos
        # shape: pos*pos,2, e.g. [0,0],[0,1],[0,2]...]
        grid = rearrange(grid, "c i j -> (i j) c")
        # shape: pos**2, pos**2, 2, e,g
        # [0,0], [0,-1], [0,-2], ..., [-5,-5]
        # [0,1], [0,0], [0,-1], ..., [-5,-4]
        # ...
        # [5,5], [5,4], [5,3], ..., [0,0]
        # rel_pos 的 min max 为 -window_size+1, window_size-1
        rel_pos = rearrange(grid, "i ... -> i 1 ...") - rearrange(
            grid, "j ... -> 1 j ..."
        )
        # rel_pos 的 min max 为 0, 2*window_size-2
        rel_pos += window_size - 1  # offset to make it non-negative
        # 因此 rel_pos[i,j][1] 相当于 2*window_size-1 个 rel_pos[i,j][0]
        rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(
            dim=-1
        )  # 将vector转换为index

        # 注册为不参与梯度更新的缓冲区
        self.register_buffer("rel_pos_indices", rel_pos_indices, persistent=False)

    def forward(self, x):
        batch, height, width, window_height, window_width, _, device, h = (
            *x.shape,
            x.device,
            self.heads,
        )

        x = self.norm(x)

        # flatten
        # batch, height, width 转化为 batch, window_height, window_width 转化为 num_tokens
        # n_token = window_height * window_width
        # n_batch = batch * height * width
        # 每个通道作为一个token，但是不同窗口的token是共享权重的
        x = rearrange(x, "b x y w1 w2 d -> (b x y) (w1 w2) d")

        # project for queries, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        # split heads

        q, k, v = map(
            lambda t: rearrange(t, "b n_token (h d) -> b h n_token d", h=h), (q, k, v)
        )

        # scale

        q = q * self.scale

        # null / memory / register kv

        mk, mv = map(
            lambda t: repeat(t, "h n_mem d -> b h n_mem d", b=q.shape[0]), self.mem_kv
        )
        num_mem = mk.shape[-2]

        # concat memory token from previous time steps
        k = torch.cat((mk, k), dim=-2)  # 将记忆键值对和当前的键值对拼接
        v = torch.cat((mv, v), dim=-2)  # 将记忆键值对和当前的键值对拼接

        # sim
        # sim = Q/sqrt(d)*K^T -> b h i i+num_mem
        # sim的第一行：
        # abs pos 0,0->0,0, 0,0->0,1, 0,0->0,2, ..., 0,0->ws-1,ws-1, ..., 0,0->(ws-1,ws-1)+num_mem
        # rel pos 0,0,      0,-1,     0,-2,     ..., -ws+1,-ws+1,    ..., -ws+1,-ws+1 (mem get 0,0)
        # sim的第二行：
        # abs pos 0,1->0,0, 0,1->0,1, 0,1->0,2, ..., 0,1->ws-1,ws-1, ..., 0,1->(ws-1,ws-1)+num_mem
        # rel pos 0,1,      0,0,      0,-1,     ..., -ws+1,-ws,      ..., -ws+1,-ws, (mem get 0,0)
        # ...
        # sim的第ws*ws行：
        # asb pos ws-1,ws-1->0,0, ws-1,ws-1->0,1, ..., ws-1,ws-1->ws-1,ws-1, ..., ws-1,ws-1->(ws-1,ws-1)+num_mem
        # rel pos ws-1,ws-1, ws-1,ws-2, ..., ws-1,ws-1-ws+1, ..., ws-1,ws-1-ws+1 (mem get 0,0)
        # 以上的位置关系就和self.rel_pos_indices对应上了
        sim = einsum("b h i d, b h j d -> b h i j", q, k)

        # add positional bias
        # 取出每个sim对应的自注意力Q_K对的相对位置的偏置参数
        # rel_pos_bias: (2*window_size-1)**2, h
        # rel_pos_indices: window_size**2, window_size**2
        # bias: window_size**2, window_size**2, h
        bias = self.rel_pos_bias(self.rel_pos_indices)

        # pad the 2nd to the last dimension by (num_mem, 0)
        bias = F.pad(bias, (0, 0, num_mem, 0), value=0.0)

        sim = sim + rearrange(bias, "i j h -> h i j")

        # attention

        attn = self.attend(sim)  # softmax last dim and dropout

        # aggregate
        # out = attn * V
        out = einsum("b h i j, b h j d -> b h i d", attn, v)

        # merge heads

        out = rearrange(
            out, "b h (w1 w2) d -> b w1 w2 (h d)", w1=window_height, w2=window_width
        )

        # combine heads out

        out = self.to_out(out)
        return rearrange(out, "(b x y) ... -> b x y ...", x=height, y=width)


class MaxViT(Module):
    def __init__(
        self,
        *,
        num_classes,
        dim,
        depth,
        dim_head=32,
        dim_conv_stem=None,
        window_size=7,
        mbconv_expansion_rate=4,
        mbconv_shrinkage_rate=0.25,
        dropout=0.1,
        channels=3,
    ):
        super().__init__()
        assert isinstance(depth, tuple), (
            "depth needs to be tuple if integers indicating number of transformer blocks at that stage"
        )

        # convolutional stem

        dim_conv_stem = default(dim_conv_stem, dim)

        self.conv_stem = nn.Sequential(
            nn.Conv2d(channels, dim_conv_stem, 3, stride=2, padding=1),
            nn.Conv2d(dim_conv_stem, dim_conv_stem, 3, padding=1),
        )

        # variables

        num_stages = len(depth)

        dims = tuple(map(lambda i: (2**i) * dim, range(num_stages)))
        dims = (dim_conv_stem, *dims)
        dim_pairs = tuple(zip(dims[:-1], dims[1:]))

        self.layers = ModuleList([])

        # shorthand for window size for efficient block - grid like attention

        w = window_size

        # iterate through stages

        cond_hidden_dims = []

        for ind, ((layer_dim_in, layer_dim), layer_depth) in enumerate(
            zip(dim_pairs, depth)
        ):
            for stage_ind in range(layer_depth):
                is_first = stage_ind == 0
                stage_dim_in = layer_dim_in if is_first else layer_dim

                cond_hidden_dims.append(stage_dim_in)

                block = nn.Sequential(
                    MBConv(
                        stage_dim_in,
                        layer_dim,
                        downsample=is_first,
                        expansion_rate=mbconv_expansion_rate,
                        shrinkage_rate=mbconv_shrinkage_rate,
                    ),
                    Rearrange(
                        "b d (x w1) (y w2) -> b x y w1 w2 d", w1=w, w2=w
                    ),  # block-like attention
                    Residual(
                        WindowAttention(
                            dim=layer_dim,
                            dim_head=dim_head,
                            dropout=dropout,
                            window_size=w,
                        )
                    ),
                    Residual(FeedForward(dim=layer_dim, dropout=dropout)),
                    Rearrange("b x y w1 w2 d -> b d (x w1) (y w2)"),
                    Rearrange(
                        "b d (w1 x) (w2 y) -> b x y w1 w2 d", w1=w, w2=w
                    ),  # grid-like attention
                    Residual(
                        WindowAttention(
                            dim=layer_dim,
                            dim_head=dim_head,
                            dropout=dropout,
                            window_size=w,
                        )
                    ),
                    Residual(FeedForward(dim=layer_dim, dropout=dropout)),
                    Rearrange("b x y w1 w2 d -> b d (w1 x) (w2 y)"),
                )

                self.layers.append(block)

        embed_dim = dims[-1]
        self.embed_dim = dims[-1]

        self.cond_hidden_dims = cond_hidden_dims

        # mlp head out

        self.mlp_head = nn.Sequential(
            Reduce("b d h w -> b d", "mean"),
            LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )

    @beartype
    def forward(
        self,
        x,
        texts: list[str] | None = None,
        cond_fns: tuple[Callable, ...] | None = None,
        cond_drop_prob=0.0,
        return_embeddings=False,
    ):
        x = self.conv_stem(x)

        cond_fns = iter(default(cond_fns, []))

        for stage in self.layers:
            cond_fn = next(cond_fns, None)

            if exists(cond_fn):
                x = cond_fn(x)

            x = stage(x)

        if return_embeddings:
            return x

        return self.mlp_head(x)


# attention


class TransformerAttention(Module):
    def __init__(
        self,
        dim,
        causal=False,
        dim_head=64,
        dim_context=None,
        heads=8,
        norm_context=False,
        dropout=0.1,
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5
        self.causal = causal
        inner_dim = dim_head * heads

        dim_context = default(dim_context, dim)

        self.norm = LayerNorm(dim)
        self.context_norm = LayerNorm(dim_context) if norm_context else nn.Identity()

        self.attn_dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_context, dim_head * 2, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False), nn.Dropout(dropout)
        )

    def forward(
        self,
        x,
        context=None,
        mask=None,
        attn_bias=None,
        attn_mask=None,
        cond_fn: Callable | None = None,
    ):
        b = x.shape[0]

        if exists(context):
            context = self.context_norm(context)

        kv_input = default(context, x)

        x = self.norm(x)

        if exists(cond_fn):
            # adaptive layer-norm
            x = cond_fn(x)

        q, k, v = self.to_q(x), *self.to_kv(kv_input).chunk(2, dim=-1)

        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)

        q = q * self.scale

        sim = einsum("b h i d, b j d -> b h i j", q, k)

        if exists(attn_bias):
            sim = sim + attn_bias

        if exists(attn_mask):
            sim = sim.masked_fill(~attn_mask, -torch.finfo(sim.dtype).max)

        if exists(mask):
            mask = rearrange(mask, "b j -> b 1 1 j")
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype=torch.bool, device=x.device).triu(
                j - i + 1
            )
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = einsum("b h i j, b j d -> b h i d", attn, v)

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(Module):
    @beartype
    def __init__(
        self, dim, dim_head=64, heads=8, depth=6, attn_dropout=0.0, ff_dropout=0.0
    ):
        super().__init__()
        self.layers = ModuleList([])
        for _ in range(depth):
            self.layers.append(
                ModuleList(
                    [
                        TransformerAttention(
                            dim=dim, heads=heads, dropout=attn_dropout
                        ),
                        FeedForward(dim=dim, dropout=ff_dropout),
                    ]
                )
            )

    @beartype
    def forward(self, x, cond_fns: tuple[Callable, ...] | None = None, attn_mask=None):
        cond_fns = iter(default(cond_fns, []))

        for attn, ff in self.layers:
            x = attn(x, attn_mask=attn_mask, cond_fn=next(cond_fns, None)) + x
            x = ff(x, cond_fn=next(cond_fns, None)) + x
        return x


# token learner module


class TokenLearner(Module):
    """
    https://arxiv.org/abs/2106.11297
    using the 1.1 version with the MLP (2 dense layers with gelu) for generating attention map
    """

    def __init__(self, *, dim, ff_mult=2, num_output_tokens=8, num_layers=2):
        super().__init__()
        inner_dim = dim * ff_mult * num_output_tokens

        self.num_output_tokens = num_output_tokens
        self.net = nn.Sequential(
            nn.Conv2d(dim * num_output_tokens, inner_dim, 1, groups=num_output_tokens),
            nn.GELU(),
            nn.Conv2d(inner_dim, num_output_tokens, 1, groups=num_output_tokens),
        )

    def forward(self, x):
        x, ps = pack_one(x, "* c h w")
        x = repeat(x, "b c h w -> b (g c) h w", g=self.num_output_tokens)
        attn = self.net(x)

        attn = rearrange(attn, "b g h w -> b 1 g h w")
        x = rearrange(x, "b (g c) h w -> b c g h w", g=self.num_output_tokens)

        x = reduce(x * attn, "b c g h w -> b c g", "mean")
        x = unpack_one(x, ps, "* c n")
        return x


# Robotic Transformer


class RT1(Module):
    @beartype
    def __init__(
        self,
        *,
        vit: MaxViT,
        num_actions=11,
        action_bins=256,
        depth=6,
        heads=8,
        dim_head=64,
        token_learner_ff_mult=2,
        token_learner_num_layers=2,
        token_learner_num_output_tokens=8,
        cond_drop_prob=0.2,
        use_attn_conditioner=False,
        conditioner_kwargs: dict = dict(),
    ):
        super().__init__()
        self.vit = vit

        self.num_vit_stages = len(vit.cond_hidden_dims)

        conditioner_klass = (
            AttentionTextConditioner if use_attn_conditioner else TextConditioner
        )

        self.conditioner = conditioner_klass(
            hidden_dims=(*tuple(vit.cond_hidden_dims), *((vit.embed_dim,) * depth * 2)),
            hiddens_channel_first=(
                *((True,) * self.num_vit_stages),
                *((False,) * depth * 2),
            ),
            cond_drop_prob=cond_drop_prob,
            **conditioner_kwargs,
        )

        self.token_learner = TokenLearner(
            dim=vit.embed_dim,
            ff_mult=token_learner_ff_mult,
            num_output_tokens=token_learner_num_output_tokens,
            num_layers=token_learner_num_layers,
        )

        self.num_learned_tokens = token_learner_num_output_tokens

        self.transformer_depth = depth

        self.transformer = Transformer(
            dim=vit.embed_dim, dim_head=dim_head, heads=heads, depth=depth
        )

        self.cond_drop_prob = cond_drop_prob

        self.to_logits = nn.Sequential(
            LayerNorm(vit.embed_dim),
            nn.Linear(vit.embed_dim, num_actions * action_bins),
            Rearrange("... (a b) -> ... a b", b=action_bins),
        )

    @beartype
    def embed_texts(self, texts: list[str]):
        return self.conditioner.embed_texts(texts)

    @classifier_free_guidance
    @beartype
    def forward(
        self,
        video,
        texts: list[str] | None = None,
        text_embeds: Tensor | None = None,
        cond_drop_prob=0.0,
    ):
        assert exists(texts) ^ exists(text_embeds)

        if exists(texts):
            num_texts = len(texts)
        elif exists(text_embeds):
            num_texts = text_embeds.shape[0]

        assert num_texts == video.shape[0], (
            f"you only passed in {num_texts} strings for guiding the robot actions, but received batch size of {video.shape[0]} videos"
        )

        cond_kwargs = dict(texts=texts, text_embeds=text_embeds)

        depth = self.transformer_depth
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        frames, device = video.shape[2], video.device

        cond_fns, _ = self.conditioner(
            **cond_kwargs,
            cond_drop_prob=cond_drop_prob,
            repeat_batch=(
                *((frames,) * self.num_vit_stages),
                *((1,) * self.transformer_depth * 2),
            ),
        )

        vit_cond_fns, transformer_cond_fns = (
            cond_fns[: -(depth * 2)],
            cond_fns[-(depth * 2) :],
        )

        video = rearrange(video, "b c f h w -> b f c h w")
        images, packed_shape = pack_one(video, "* c h w")

        tokens = self.vit(
            images,
            texts=texts,
            cond_fns=vit_cond_fns,
            cond_drop_prob=cond_drop_prob,
            return_embeddings=True,
        )

        tokens = unpack_one(tokens, packed_shape, "* c h w")
        learned_tokens = self.token_learner(tokens)

        learned_tokens = rearrange(learned_tokens, "b f c n -> b (f n) c")

        # causal attention mask

        attn_mask = torch.ones((frames, frames), dtype=torch.bool, device=device).triu(
            1
        )
        attn_mask = repeat(
            attn_mask,
            "i j -> (i r1) (j r2)",
            r1=self.num_learned_tokens,
            r2=self.num_learned_tokens,
        )

        # sinusoidal positional embedding

        pos_emb = posemb_sincos_1d(
            frames,
            learned_tokens.shape[-1],
            dtype=learned_tokens.dtype,
            device=learned_tokens.device,
        )

        learned_tokens = learned_tokens + repeat(
            pos_emb, "n d -> (n r) d", r=self.num_learned_tokens
        )

        # attention

        attended_tokens = self.transformer(
            learned_tokens, cond_fns=transformer_cond_fns, attn_mask=~attn_mask
        )

        pooled = reduce(attended_tokens, "b (f n) d -> b f d", "mean", f=frames)

        logits = self.to_logits(pooled)
        return logits
