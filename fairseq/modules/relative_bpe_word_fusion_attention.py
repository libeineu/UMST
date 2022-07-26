# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.incremental_decoding_utils import with_incremental_state
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from torch import Tensor, nn
from torch.nn import Parameter


@with_incremental_state
class RelativeBPEWORDFusionMultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        max_relative_length,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False,
        q_noise=0.0,
        qn_block_size=8,
        k_only=True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5
        self.max_relative_length = max_relative_length
        self.k_only = k_only
        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )

        self.k_proj = quant_noise(
            nn.Linear(self.kdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.k_proj_word = quant_noise(
            nn.Linear(self.kdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.v_proj = quant_noise(
            nn.Linear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.q_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.q_proj_word = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )

        self.out_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        #self.attn_gate = nn.Linear(self.head_dim, 1, bias=bias)
        
        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        # relative positional representations (key) and (value)
        self.relative_position_keys_bpe = Parameter(torch.Tensor(2 * self.max_relative_length + 1, self.head_dim))
        if not self.k_only:
            self.relative_position_values_bpe = Parameter(torch.Tensor(2 * self.max_relative_length + 1, self.head_dim))

        # relative positional representations (key) and (value)
        self.relative_position_keys_word = Parameter(torch.Tensor(2 * self.max_relative_length + 1, self.head_dim))
        if not self.k_only:
            self.relative_position_values_word = Parameter(torch.Tensor(2 * self.max_relative_length + 1, self.head_dim))

        self.reset_parameters()

        self.onnx_trace = False
        self.tpu = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def prepare_for_tpu_(self, **kwargs):
        self.tpu = True

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.k_proj_word.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj_word.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.k_proj_word.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)
            nn.init.xavier_uniform_(self.q_proj_word.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        #nn.init.xavier_uniform_(self.attn_gate.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)
        
        nn.init.xavier_uniform_(self.relative_position_keys_bpe)
        if not self.k_only:
            nn.init.xavier_uniform_(self.relative_position_values_bpe)
        nn.init.xavier_uniform_(self.relative_position_keys_word)
        if not self.k_only:
            nn.init.xavier_uniform_(self.relative_position_values_word)

    def forward(
        self,
        query_bpe,
        query_word,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        key_padding_mask_word: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
        mapping: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        tgt_len_bpe, bsz, embed_dim = query_bpe.size()
        tgt_len_word, bsz, embed_dim = query_word.size()
        assert embed_dim == self.embed_dim
        assert list(query_bpe.size()) == [tgt_len_bpe, bsz, embed_dim]
        assert list(query_word.size()) == [tgt_len_word, bsz, embed_dim]

        if (
            not self.onnx_trace
            and not self.tpu  # don't use PyTorch version on TPUs
            and incremental_state is None
            and not static_kv
            # A workaround for quantization to work. Otherwise JIT compilation
            # treats bias in linear module as method.
            and not torch.jit.is_scripting()
        ):
            assert key is not None and value is not None
            

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None

        if self.self_attention:
            q_bpe = self.q_proj(query_bpe)   
            k_bpe = self.k_proj(query_bpe)
            q_word = self.q_proj_word(query_word)
            k_word = self.k_proj_word(query_word)
            v = self.v_proj(query_bpe)
        
        q_bpe *= self.scaling
        q_word *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k_word = torch.cat([k_word, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        key_padding_mask.new_zeros(key_padding_mask.size(0), 1),
                    ],
                    dim=1,
                )

        q_bpe = (
            q_bpe.contiguous()
            .view(tgt_len_bpe, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        q_word = (
            q_word.contiguous()
            .view(tgt_len_word, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        if k_bpe is not None:
            k_bpe = (
                k_bpe.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
        if k_word is not None:
            k_word = (
                k_word.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
        if v is not None:
            v = (
                v.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )

        assert k_bpe is not None
        src_len_bpe = k_bpe.size(1)
        assert k_word is not None
        src_len_word = k_word.size(1)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len_bpe

        if self.add_zero_attn:
            assert v is not None
            src_len_bpe += 1
            k_bpe = torch.cat([k_bpe, k_bpe.new_zeros((k_bpe.size(0), 1) + k_bpe.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        torch.zeros(key_padding_mask.size(0), 1).type_as(
                            key_padding_mask
                        ),
                    ],
                    dim=1,
                )
        
        relative_positions_matrix_bpe = self._generate_relative_positions_matrix(
            src_len_bpe, self.max_relative_length, incremental_state
        )

        relative_positions_matrix_word = self._generate_relative_positions_matrix(
            src_len_word, self.max_relative_length, incremental_state
        )

        if self.k_only:
            relation_keys_bpe = F.embedding(relative_positions_matrix_bpe.long().cuda(), self.relative_position_keys_bpe)
            relation_keys_word = F.embedding(relative_positions_matrix_word.long().cuda(), self.relative_position_keys_bpe)
        else:
            relation_keys_bpe = F.embedding(relative_positions_matrix_bpe.long().cuda(), self.relative_position_keys_bpe)
            relation_values_bpe = F.embedding(relative_positions_matrix_bpe.long().cuda(), self.relative_position_values_bpe)
            relation_keys_word = F.embedding(relative_positions_matrix_word.long().cuda(), self.relative_position_keys_word)
            relation_values_word = F.embedding(relative_positions_matrix_word.long().cuda(), self.relative_position_values_word)


        attn_weights_bpe = self._relative_attention_inner(q_bpe, k_bpe, relation_keys_bpe, transpose=True)
        attn_weights_bpe = self.apply_sparse_mask(attn_weights_bpe, tgt_len_bpe, src_len_bpe, bsz)

        attn_weights_word = self._relative_attention_inner(q_word, k_word, relation_keys_word, transpose=True)
        attn_weights_word = self.apply_sparse_mask(attn_weights_word, tgt_len_word, src_len_word, bsz)
        
        


        assert list(attn_weights_bpe.size()) == [bsz * self.num_heads, tgt_len_bpe, src_len_bpe]
        assert list(attn_weights_word.size()) == [bsz * self.num_heads, tgt_len_word, src_len_word]
        
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights_bpe.size(0), 1, 1)
            attn_weights_bpe += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights_bpe = attn_weights_bpe.view(bsz, self.num_heads, tgt_len_bpe, src_len_bpe)
            #attn_weights_word = attn_weights_word.view(bsz, self.num_heads, tgt_len_word, src_len_word)
            if not self.tpu:
                attn_weights_bpe = attn_weights_bpe.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    float("-inf"),
                )
                
            else:
                attn_weights_bpe = attn_weights_bpe.transpose(0, 2)
                attn_weights_bpe = attn_weights_bpe.masked_fill(key_padding_mask, float("-inf"))
                attn_weights_bpe = attn_weights_bpe.transpose(0, 2)
                
                
            attn_weights_bpe = attn_weights_bpe.view(bsz * self.num_heads, tgt_len_bpe, src_len_bpe)
            #attn_weights_word = attn_weights_word.view(bsz * self.num_heads, tgt_len_word, src_len_word)
        if key_padding_mask_word is not None:
            # don't attend to padding symbols
            #attn_weights_bpe = attn_weights_bpe.view(bsz, self.num_heads, tgt_len_bpe, src_len_bpe)
            attn_weights_word = attn_weights_word.view(bsz, self.num_heads, tgt_len_word, src_len_word)
            if not self.tpu:
                
                attn_weights_word = attn_weights_word.masked_fill(
                    key_padding_mask_word.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    float("-inf"),
                )
            else:
                
                attn_weights_word = attn_weights_word.transpose(0, 2)
                attn_weights_word = attn_weights_word.masked_fill(key_padding_mask_word, float("-inf"))
                attn_weights_word = attn_weights_word.transpose(0, 2)
            #attn_weights_bpe = attn_weights_bpe.view(bsz * self.num_heads, tgt_len_bpe, src_len_bpe)
            attn_weights_word = attn_weights_word.view(bsz * self.num_heads, tgt_len_word, src_len_word)

        if before_softmax:
            return attn_weights_bpe, v

        attn_weights_bpe_float = utils.softmax(
            attn_weights_bpe, dim=-1, onnx_trace=self.onnx_trace
        )
        attn_weights_bpe = attn_weights_bpe_float.type_as(attn_weights_bpe)
        attn_probs = self.dropout_module(attn_weights_bpe)
        
        
        
        
        attn_weights_word_float = utils.softmax(
            attn_weights_word, dim=-1, onnx_trace=self.onnx_trace
        )
        attn_weights_word = attn_weights_word_float.type_as(attn_weights_word)
        mapping = mapping.unsqueeze(1).repeat([1, self.num_heads, 1, 1]).contiguous().view(-1, mapping.shape[-2], mapping.shape[-1])
        attn_weights_word = mapping@attn_weights_word@mapping.transpose(1,2)
        attn_probs_word =  self.dropout_module(attn_weights_word)
        attn_probs = 0.5*attn_probs_word + 0.5*attn_probs
        assert v is not None
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len_bpe, self.head_dim]
        if self.onnx_trace and attn.size(1) == 1:
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len_bpe, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len_bpe, bsz, embed_dim)
        attn = self.out_proj(attn)
        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = attn_weights_bpe_float.view(
                bsz, self.num_heads, tgt_len_bpe, src_len_bpe
            ).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights

    @staticmethod
    def _append_prev_key_padding_mask(
        key_padding_mask: Optional[Tensor],
        prev_key_padding_mask: Optional[Tensor],
        batch_size: int,
        src_len: int,
        static_kv: bool,
    ) -> Optional[Tensor]:
        # saved key padding masks have shape (bsz, seq_len)
        if prev_key_padding_mask is not None and static_kv:
            new_key_padding_mask = prev_key_padding_mask
        elif prev_key_padding_mask is not None and key_padding_mask is not None:
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), key_padding_mask.float()], dim=1
            )
        # During incremental decoding, as the padding token enters and
        # leaves the frame, there will be a time when prev or current
        # is None
        elif prev_key_padding_mask is not None:
            filler = torch.zeros(
                (batch_size, src_len - prev_key_padding_mask.size(1)),
                device=prev_key_padding_mask.device,
            )
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), filler.float()], dim=1
            )
        elif key_padding_mask is not None:
            filler = torch.zeros(
                (batch_size, src_len - key_padding_mask.size(1)),
                device=key_padding_mask.device,
            )
            new_key_padding_mask = torch.cat(
                [filler.float(), key_padding_mask.float()], dim=1
            )
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

    @torch.jit.export
    def reorder_incremental_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        new_order: Tensor,
    ):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    if self.encoder_decoder_attention and input_buffer_k.size(
                        0
                    ) == new_order.size(0):
                        break
                    input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def _get_input_buffer(
        self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        buffer: Dict[str, Optional[Tensor]],
    ):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)

    def apply_sparse_mask(self, attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""
        items_to_add = {}
        keys_to_remove = []
        for k in state_dict.keys():
            if k.endswith(prefix + "in_proj_weight"):
                # in_proj_weight used to be q + k + v with same dimensions
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
                items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim : 2 * dim]
                items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim :]

                keys_to_remove.append(k)

                k_bias = prefix + "in_proj_bias"
                if k_bias in state_dict.keys():
                    dim = int(state_dict[k].shape[0] / 3)
                    items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
                    items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][
                        dim : 2 * dim
                    ]
                    items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim :]

                    keys_to_remove.append(prefix + "in_proj_bias")

        for k in keys_to_remove:
            del state_dict[k]

        for key, value in items_to_add.items():
            state_dict[key] = value
    
    def _generate_relative_positions_matrix(self, length, max_relative_length, incremental_state):
        if not incremental_state:
            # training process
            range_vec = torch.arange(length)
            range_mat = range_vec.repeat(length, 1)
            distance_mat = range_mat - range_mat.transpose(0, 1)
        else:
            distance_mat = torch.range(-length + 1, 0).view(1, -1)

        distance_mat_clipped = torch.clamp(distance_mat, -max_relative_length, max_relative_length)

        # position difference.
        final_mat = distance_mat_clipped + max_relative_length

        return final_mat

    def _relative_attention_inner(self, x, y, z, transpose=True):
        """Relative position-aware dot-product attention inner calculation.

        This batches matrix multiply calculations to avoid unnecessary broadcasting.

        Args:
          x: Tensor with shape [batch_size*heads, length, length or depth].
          y: Tensor with shap e [batch_size*heads, length, depth].
          z: Tensor with shape [length, length, depth].
          transpose: Whether to tranpose inner matrices of y and z. Should be true if
              last dimension of x is depth, not length.

        Returns:
          A Tensor with shape [batch_size*heads, length, length or depth].

          wq: this function actually does 'X(Y+Z)', where Z is vector,
          but factor above formular as: 'XY + XZ'
        """
        batch_size_mul_head = x.size()[0]
        length = z.size()[0]
        # print(batch_size_mul_head, length)
        # xy_matmul is [batch_size*heads, length, length or depth]
        if transpose:
            y = y.transpose(1, 2)
        xy_matmul = torch.bmm(x, y)
        # x_t is [length, batch_size * heads, length or depth]
        x_t = x.transpose(0, 1)
        # x_tz_matmul is [length, batch_size * heads, length or depth]
        if transpose:
            z = z.transpose(1, 2)
        x_tz_matmul = torch.bmm(x_t, z).transpose(0, 1).view(batch_size_mul_head, length, -1)

        attn = xy_matmul + x_tz_matmul

        return attn
