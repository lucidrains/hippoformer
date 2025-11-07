from __future__ import annotations

import torch
from torch import nn, Tensor, cat, stack, arange, zeros_like, einsum, tensor
import torch.nn.functional as F
from torch.nn import Module
from torch.jit import ScriptModule, script_method
from torch.func import vmap, grad, functional_call

from beartype import beartype

from einx import multiply
from einops import repeat, rearrange, pack, unpack
from einops.layers.torch import Rearrange

from x_mlps_pytorch import create_mlp

from assoc_scan import AssocScan

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def pack_with_inverse(t, pattern):
    packed, packed_shape = pack([t], pattern)

    def inverse(out, inv_pattern = None):
        inv_pattern = default(inv_pattern, pattern)
        unpacked, = unpack(out, packed_shape, inv_pattern)
        return unpacked

    return packed, inverse

def l2norm(t):
    return F.normalize(t, dim = -1)

# Muon - Jordan et al from oss community - applied to the latest version of titans

def newtonschulz5(
    t,
    steps = 5,
    eps = 1e-7,
    coefs = (3.4445, -4.7750, 2.0315)
):
    not_weights = t.ndim <= 3

    if not_weights:
        return t

    shape = t.shape
    should_transpose = shape[-2] > shape[-1]

    if should_transpose:
        t = t.transpose(-1, -2)

    t, inv_pack = pack_with_inverse(t, '* i j')
    t = t / t.norm(dim = (-1, -2), keepdim = True).clamp(min = eps)

    a, b, c = coefs

    for _ in range(steps):
        A = t @ t.transpose(-1, -2)
        B = b * A + c * A @ A
        t = a * t + B @ t

    if should_transpose:
        t = t.transpose(-1, -2)

    return inv_pack(t)

# sensory encoder decoder for 2d

grid_sensory_enc_dec = (
    create_mlp(
        dim = 32 * 2,
        dim_in = 9,
        dim_out = 32,
        depth = 3,
    ),
    create_mlp(
        dim = 32 * 2,
        dim_in = 32,
        dim_out = 9,
        depth = 3,
    ),
)

# sensory encoder decoder for 3d maze

class EncoderPackTime(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        x = rearrange(x, 'b c t h w -> b t c h w')
        x, packed_shape = pack([x], '* c h w')

        x = self.fn(x)

        x, = unpack(x, packed_shape, '* d')
        print(x.shape)
        return x

class DecoderPackTime(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        x, packed_shape = pack(x, '* d')

        x = self.fn(x)

        x = unpack(x, packed_shape, '* c h w')
        x = rearrange(x, 'b t c h w -> b c t h w')
        return x

maze_sensory_enc_dec = (
    EncoderPackTime(nn.Sequential(
        nn.Conv2d(3, 16, 7, 2, padding = 3),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, 2, 1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, 2, 1),
        nn.ReLU(),
        nn.Conv2d(64, 128, 3, 2, 1),
        nn.ReLU(),
        Rearrange('b ... -> b (...)'),
        nn.Linear(2048, 32)
    )),
    DecoderPackTime(nn.Sequential(
        nn.Linear(32, 2048),
        Rearrange('b (c h w) -> b c h w', c = 128, h = 4),
        nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding = (1, 1)),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 32, 3, 2, 1, output_padding = (1, 1)),
        nn.ReLU(),
        nn.ConvTranspose2d(32, 16, 3, 2, 1, output_padding = (1, 1)),
        nn.ReLU(),
        nn.ConvTranspose2d(16, 3, 3, 2, 1, output_padding = (1, 1))
    ))
)

# path integration

class RNN(ScriptModule):
    def __init__(
        self,
        dim,
    ):
        super().__init__()
        self.init_hidden = nn.Parameter(torch.randn(1, dim) * 1e-2)

    @script_method
    def forward(
        self,
        transitions: Tensor,
        hidden: Tensor | None = None
    ) -> Tensor:

        batch, seq_len = transitions.shape[:2]

        if hidden is None:
            hidden = l2norm(self.init_hidden)
            hidden = hidden.expand(batch, -1)

        hiddens: list[Tensor] = []

        for i in range(seq_len):
            transition = transitions[:, i]

            hidden = einsum('b i, b i j -> b j', hidden, transition)
            hidden = F.relu(hidden)
            hidden = l2norm(hidden)

            hiddens.append(hidden)

        return stack(hiddens, dim = 1)

class PathIntegration(Module):
    def __init__(
        self,
        dim_action,
        dim_structure,
        mlp_hidden_dim = None,
        mlp_depth = 2
    ):
        # they use the same approach from Ruiqi Gao's paper from 2021
        super().__init__()

        self.init_structure = nn.Parameter(torch.randn(dim_structure))

        self.to_transitions = create_mlp(
            default(mlp_hidden_dim,  dim_action * 4),
            dim_in = dim_action,
            dim_out = dim_structure * dim_structure,
            depth = mlp_depth
        )

        self.mlp_out_to_weights = Rearrange('... (i j) -> ... i j', j = dim_structure)

        self.rnn = RNN(dim_structure)

    def forward(
        self,
        actions,                 # (b n d)
        prev_structural = None   # (b n d) | (b d)
    ):
        batch = actions.shape[0]

        transitions = self.to_transitions(actions)
        transitions = self.mlp_out_to_weights(transitions)

        if exists(prev_structural) and prev_structural.ndim == 3:
            prev_structural = prev_structural[:, -1]

        return self.rnn(transitions, prev_structural)

# custom transformer proposed by James Whittington that bridges to hippocampal models with a few twists

# the mmTEM can be seen as a linear attention / TTT variant of what he proposed
# needed for the baseline as well as the parallel block to bolster local time prediction

# https://arxiv.org/abs/2112.04035

def FeedForward(dim, mult = 4.):
    dim_inner = int(dim * mult)
    return nn.Sequential(
        nn.Linear(dim, dim_inner),
        nn.GELU(),
        nn.Linear(dim_inner, dim)
    )

class Attention(Module):
    def __init__(
        self,
        dim_q,
        dim_kv,
        window_size,
        dim_head = 64,
        heads = 8,
    ):
        super().__init__()
        dim_inner = dim_head * heads
        self.scale = dim_head ** -0.5

        self.to_queries = nn.Linear(dim_q, dim_inner, bias = False)
        self.to_key_values = nn.Linear(dim_kv, dim_inner * 2, bias = False)

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.window_size = window_size

        self.to_out = nn.Linear(dim_inner, dim_q, bias = False)
        self.attn_head_sink = nn.Parameter(torch.randn(heads) * 1e-2) # needed as the diagonal is masked out, and for attention sink

    def forward(
        self,
        queries_input,
        key_values_input,
        kv_cache = None
    ):
        batch, seq_len, device = *queries_input.shape[:2], queries_input.device

        q = self.to_queries(queries_input)

        k, v = self.to_key_values(key_values_input).chunk(2, dim = -1)

        q, k, v = tuple(self.split_heads(t) for t in (q, k, v))

        if exists(kv_cache):
            ck, cv = kv_cache
            k = cat((ck, k), dim = -2)
            v = cat((cv, v), dim = -2)

        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # the diagonal is masked out

        i, j = sim.shape[-2:]

        j_seq = arange(j, device = device)[:, None]
        i_seq = arange(i, device = device)[None, :] + (j - i)

        windowed_causal_mask_without_diagonal = (i_seq > j_seq) & ((i_seq - j_seq) <= self.window_size)

        sim = sim.masked_fill(windowed_causal_mask_without_diagonal, -torch.finfo(sim.dtype).max)

        # attention sink, for token as well as for attention sinking - from gpt-oss

        attn_sink = repeat(self.attn_head_sink, 'h -> b h i 1', b = batch, i = seq_len)

        sim = cat((attn_sink, sim), dim = -1)

        attn = sim.softmax(dim = -1)

        attn = attn[..., 1:] # remove sink

        # aggregate

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = self.merge_heads(out)

        return self.to_out(out), stack((k, v))

class TEMTransformerBlock(Module):
    def __init__(
        self,
        dim_structure,
        dim_encoded_sensory,
        dim_head = 64,
        heads = 8,
        ff_expansion_factor = 4.,
        window_size = 64
    ):
        super().__init__()

        self.attn = Attention(dim_structure, dim_structure + dim_encoded_sensory, window_size, dim_head = dim_head, heads = heads)
        self.ff = FeedForward(dim_structure, ff_expansion_factor)

        self.window_size = window_size

    def forward(
        self,
        structural_codes,
        encoded_sensory,
        kv_cache = None
    ):
        structure_and_sensory = cat((structural_codes, encoded_sensory), dim = -1)

        retrieved, next_kv_cache = self.attn(structural_codes, structure_and_sensory, kv_cache = kv_cache)

        x = retrieved + structural_codes

        x = self.ff(x) + x

        next_kv_cache = next_kv_cache[:, -self.window_size:]

        return x, next_kv_cache

# proposed mmTEM

class mmTEM(Module):
    @beartype
    def __init__(
        self,
        dim,
        *,
        sensory_encoder_decoder: tuple[Module, Module],
        dim_sensory,
        dim_action,
        dim_encoded_sensory,
        dim_structure,
        meta_mlp_depth = 2,
        decoder_mlp_depth = 2,
        structure_variance_pred_mlp_depth = 2,
        path_integrate_kwargs: dict = dict(),
        loss_weight_generative = 1.,
        loss_weight_inference = 1.,
        loss_weight_consistency = 1.,
        loss_weight_relational = 1.,
        integration_ratio_learned = True,
        muon_update = False,
        assoc_scan_kwargs: dict = dict()
    ):
        super().__init__()

        # sensory

        sensory_encoder, sensory_decoder = sensory_encoder_decoder

        self.sensory_encoder = sensory_encoder
        self.sensory_decoder = sensory_decoder

        dim_joint_rep = dim_encoded_sensory + dim_structure

        self.dim_encoded_sensory = dim_encoded_sensory
        self.dim_structure = dim_structure
        self.joint_dims = (dim_structure, dim_encoded_sensory)

        # path integrator

        self.path_integrator = PathIntegration(
            dim_action = dim_action,
            dim_structure = dim_structure,
            **path_integrate_kwargs
        )

        # meta mlp related

        self.to_queries = nn.Linear(dim_joint_rep, dim, bias = False)
        self.to_keys = nn.Linear(dim_joint_rep, dim, bias = False)
        self.to_values = nn.Linear(dim_joint_rep, dim, bias = False)

        self.to_learned_optim_hparams = nn.Linear(dim_joint_rep, 3, bias = False) # for learning rate, forget gate, and momentum
        self.assoc_scan = AssocScan(*assoc_scan_kwargs)

        self.meta_memory_mlp = create_mlp(
            dim = dim * 2,
            depth = meta_mlp_depth,
            dim_in = dim,
            dim_out = dim,
            activation = nn.ReLU()
        )

        def forward_with_mse_loss(params, keys, values):
            pred = functional_call(self.meta_memory_mlp, params, keys)
            return F.mse_loss(pred, values)

        grad_fn = grad(forward_with_mse_loss)

        self.per_sample_grad_fn = vmap(vmap(grad_fn, in_dims = (None, 0, 0)), in_dims = (0, 0, 0))

        # mlp decoder (from meta mlp output to joint)

        self.memory_output_decoder = create_mlp(
            dim = dim * 2,
            dim_in = dim,
            dim_out = dim_joint_rep,
            depth = decoder_mlp_depth,
            activation = nn.ReLU()
        )

        # the mlp that predicts the variance for the structural code
        # for correcting the generated structural code modeling the feedback from HC to MEC

        self.structure_variance_pred_mlp = create_mlp(
            dim = dim_structure * 2,
            dim_in = dim_structure * 2 + 1,
            dim_out = dim_structure,
            depth = structure_variance_pred_mlp_depth
        )

        # loss related

        self.loss_weight_generative = loss_weight_generative
        self.loss_weight_inference = loss_weight_inference
        self.loss_weight_relational = loss_weight_relational
        self.loss_weight_consistency = loss_weight_consistency
        self.register_buffer('zero', tensor(0.), persistent = False)

        # update with muon

        self.muon_update = muon_update

        # there is an integration ratio for error correction, but unclear what value this is fixed to or whether it is learned

        self.integration_ratio = nn.Parameter(tensor(0.), requires_grad = integration_ratio_learned)

    def init_params_and_momentum(
        self,
        batch_size
    ):

        params_dict = dict(self.meta_memory_mlp.named_parameters())

        params = {name: repeat(param, '... -> b ...', b = batch_size) for name, param in params_dict.items()}

        momentums = {name: zeros_like(param) for name, param in params.items()}

        return params, momentums

    def retrieve(
        self,
        structural_codes,
        encoded_sensory
    ):
        joint = cat((structural_codes, encoded_sensory), dim = -1)

        queries = self.to_queries(joint)

        retrieved = self.meta_memory_mlp(queries)

        return self.memory_output_decoder(retrieved).split(self.joint_dims, dim = -1)

    def forward(
        self,
        sensory,
        actions,
        memory_mlp_params = None,
        return_losses = False,
        return_memory_mlp_params = False
    ):
        batch = actions.shape[0]

        structural_codes = self.path_integrator(actions)

        encoded_sensory = self.sensory_encoder(sensory)

        # 1. first have the structure code be able to fetch from the meta memory mlp

        decoded_gen_structure, decoded_encoded_sensory = self.retrieve(structural_codes, zeros_like(encoded_sensory))

        decoded_sensory = self.sensory_decoder(decoded_encoded_sensory)

        generative_pred_loss = F.mse_loss(sensory, decoded_sensory)

        # 2. relational

        # 2a. structure from content

        decoded_structure, decoded_encoded_sensory = self.retrieve(zeros_like(structural_codes), encoded_sensory)

        structure_from_content_loss = F.mse_loss(decoded_structure, structural_codes)

        # 2b. structure from structure

        decoded_structure, decoded_encoded_sensory = self.retrieve(zeros_like(structural_codes), encoded_sensory)

        structure_from_structure_loss = F.mse_loss(decoded_structure, structural_codes)

        relational_loss = structure_from_content_loss + structure_from_structure_loss

        # 3. consistency - modeling a feedback system from hippocampus to path integration

        corrected_structural_code, corrected_encoded_sensory = self.retrieve(decoded_gen_structure, encoded_sensory)

        sensory_sse = (corrected_encoded_sensory - encoded_sensory).norm(dim = -1, keepdim = True).pow(2)

        pred_variance = self.structure_variance_pred_mlp(cat((corrected_structural_code, decoded_gen_structure, sensory_sse), dim = -1))

        inf_structural_code = decoded_gen_structure + (corrected_structural_code - decoded_gen_structure) * self.integration_ratio.sigmoid() * pred_variance

        consistency_loss = F.mse_loss(decoded_gen_structure, inf_structural_code)

        # 4. final inference loss

        final_structural_code, inf_encoded_sensory = self.retrieve(inf_structural_code, zeros_like(encoded_sensory))

        decoded_inf_sensory = self.sensory_decoder(inf_encoded_sensory)

        inference_pred_loss = F.mse_loss(sensory, decoded_inf_sensory)

        # 5. store the final structural code from step 4 + encoded sensory

        joint_code_to_store = cat((final_structural_code, encoded_sensory), dim = -1)

        keys = self.to_keys(joint_code_to_store)
        values = self.to_values(joint_code_to_store)

        lr, forget, beta = self.to_learned_optim_hparams(joint_code_to_store).unbind(dim = -1)

        if exists(memory_mlp_params):
            params, momentums = memory_mlp_params
        else:
            params, momentums = self.init_params_and_momentum(batch)

        # store by getting gradients of mse loss of keys and values

        grads = self.per_sample_grad_fn(params, keys, values)

        # update the meta mlp parameters and momentums

        next_params = dict()
        next_momentum = dict()

        for (
            (key, param),
            (_, grad),
            (_, momentum)
        ) in zip(
            params.items(),
            grads.items(),
            momentums.items()
        ):

            grad, inverse_pack = pack_with_inverse(grad, 'b t *')

            grad = multiply('b t ..., b t', grad, lr)

            expanded_beta = repeat(beta, 'b t -> b t w', w = grad.shape[-1])

            update = self.assoc_scan(grad, expanded_beta.sigmoid(), momentum)

            # store next momentum

            next_momentum[key] = update[:, -1]

            # maybe muon

            if self.muon_update:
                update = newtonschulz5(update)

            # with forget gating

            expanded_forget = repeat(forget, 'b t -> b t w', w = grad.shape[-1])

            acc_update = self.assoc_scan(-update, expanded_forget.sigmoid(), param)

            acc_update = inverse_pack(acc_update)

            # set the next params and momentum, which can be passed back in

            next_params[key] =  acc_update[:, -1]

        # losses

        total_loss = (
            generative_pred_loss * self.loss_weight_generative +
            relational_loss * self.loss_weight_relational +
            consistency_loss * self.loss_weight_consistency +
            inference_pred_loss * self.loss_weight_inference
        )

        losses = (
            generative_pred_loss,
            relational_loss,
            consistency_loss,
            inference_pred_loss
        )

        if return_memory_mlp_params:
            return next_params, next_momentum

        if not return_losses:
            return total_loss

        return total_loss, losses
