import json
from typing import Optional, Union, List, Tuple

import torch
from torch import nn
from torch.nn import Parameter, CrossEntropyLoss
from transformers import BartForConditionalGeneration, BartModel, BartForQuestionAnswering
from transformers.modeling_outputs import Seq2SeqLMOutput, Seq2SeqQuestionAnsweringModelOutput
from transformers.pytorch_utils import Conv1D

from logging import getLogger

logger = getLogger(__name__)


class Plug_LM_Header(torch.nn.Module):
    r"""
    This module implements lm_header that enables hidden_states transformed to incremental vocabulary in a plug-and-play way.
    """

    def __init__(self, config, num_incremental_embeddings: Optional[int] = 0, _weight: Optional[torch.Tensor] = None,
                 device=None, dtype=None, bias=None) -> None:
        '''
        num_embeddings: output size of lm head, which should be the size of target vocab

        num_incremental_embeddings: if output size is larger than input size of embedding layer, that incremetnal size should
        be the reduction of target vocab size and source vocab size. It can be 0 if there is no index overlap between source and target vocabulary.

        mapping_index_file: if not none, a cross-attention mechanism of the incremental weight over source weight will be used.


        '''
        factory_kwargs = {'device': device, 'dtype': dtype, 'bias': bias}
        self.config = config
        self.out_features = config.vocab_size
        self.in_features = config.n_embd if hasattr(config, "n_embd") else config.d_model
        super(Plug_LM_Header, self).__init__()
        self.num_incremental_embeddings = num_incremental_embeddings
        self.lazy_mapping_weight = config.lazy_mapping_weight
        if num_incremental_embeddings > 0:
            self.c_attn = Conv1D(2 * self.in_features, self.in_features)
            self.q_attn = Conv1D(self.in_features, self.in_features)
            self.c_proj = Conv1D(self.in_features, self.in_features)
            self.resid_dropout = nn.Dropout(config.resid_pdrop if hasattr(config, "resid_pdrop") else config.dropout)
            self.attn_dropout = nn.Dropout(
                config.attn_pdrop if hasattr(config, "attn_pdrop") else config.attention_dropout)
        self.device = device
        self.dtype = dtype
        self.mapping_index = None
        if _weight is None:
            self.lm_head = torch.nn.Linear(self.in_features, self.out_features, **factory_kwargs) # 10 -> 10 weigth.size 100
            print("self.lm_head size:", self.lm_head.weight.size(), "device:", self.lm_head.weight.device)
            # self.weight = Parameter(torch.empty((self.out_features - num_incremental_embeddings, self.in_features), **factory_kwargs))
            # self.lm_head = torch.nn.Linear(self.in_features, self.out_features - self.num_incremental_embeddings, **factory_kwargs)
            # if self.num_incremental_embeddings > 0:
            #     # self.weight_incremental = Parameter(torch.empty((num_incremental_embeddings, self.in_features), **factory_kwargs))
            #     self.incremental_lm_head = torch.nn.Linear(self.in_features, self.num_incremental_embeddings, **factory_kwargs)
            #     print("self.incremental_lm_head size:", self.incremental_lm_head.weight.size())
        else:
            assert list(_weight.shape) == [self.out_features, self.in_features], \
                'Shape of weight does not match num_embeddings and embedding_dim'
            self.lm_head = torch.nn.Linear(self.in_features, self.out_features, **factory_kwargs)
            # self.lm_head = torch.nn.Linear(self.in_features, self.out_features - self.num_incremental_embeddings, **factory_kwargs)
            self.lm_head.weight = Parameter(_weight)

        if config.mapping_index_file:
            mapping_index_matrix = json.load(open(config.mapping_index_file, "r"))
            # convert None to 0
            mapping_index_matrix = [[1 if x is None else x for x in row] for row in mapping_index_matrix]
            self.mapping_index = torch.Tensor(mapping_index_matrix).to(self.device).to(torch.long)
            self.mapping_lm_head = torch.nn.Linear(self.in_features, self.out_features, **factory_kwargs)
            # self.tie_mapping_lm_head(self.dtype, self.device)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        attn_weights = attn_weights / torch.full(
            [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
        )

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def cross_attention(self, query_embeds, kv_embeds, num_heads=4, attention_mask=None, output_attentions=False):
        self.num_heads = num_heads
        self.head_dim = self.in_features // self.num_heads
        query = self.q_attn(query_embeds)
        key, value = self.c_attn(kv_embeds).split(self.in_features, dim=2)
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)
        attn_output, attn_weights = self._attn(query, key, value, attention_mask=attention_mask)
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        outputs = (attn_output)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs

    def get_mapping_weight(self, dtype, device):
        if not hasattr(self, "mapping_index") or self.mapping_index is None:
            return

        device = self.lm_head.weight.device
        q = self.lm_head.weight[-self.num_incremental_embeddings:, None, :]
        kv_table = self.lm_head.weight
        kv = torch.clone(kv_table[self.mapping_index].detach())
        kv.requires_grad = False

        attention_mask = torch.not_equal(self.mapping_index, self.config.pad_token_id).to(device).to(torch.long)
        attention_mask = attention_mask[:, None, None, :]
        attention_mask = attention_mask.to(device)
        attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        attention_mask = (1.0 - attention_mask) * torch.finfo(dtype if dtype is not None else torch.float).min
        outputs = self.cross_attention(q, kv, attention_mask=attention_mask)

        weight_incremental_after_attention = outputs.view(-1, self.in_features)
        # weight = torch.concat([self.lm_head.weight,weight_incremental_after_attention], dim = 0)
        weight = torch.concat(
            [self.lm_head.weight[:-self.num_incremental_embeddings, :], weight_incremental_after_attention], dim=0)

        # self.mapping_lm_head.weight is used for parameter storing. However parameter is not in computation graph, for gradient prop, we use variable weight to calculate loss.
        return weight

    def tie_mapping_lm_head(self, dtype, device):
        if not hasattr(self, "mapping_index") or self.mapping_index is None:
            return

        weight = self.get_mapping_weight(dtype, device)

        self.mapping_lm_head.weight = nn.Parameter(weight)

    def init_lm_head_by_mapping(self, dtype, device):
        if not hasattr(self, "mapping_index") or self.mapping_index is None:
            return
        # logger.info("init_lm_head_by_mapping")
        kv_table = self.lm_head.state_dict()
        # logger.info(f"{(kv_table['weight'])[-1:,:]}")
        kv = kv_table["weight"][self.mapping_index]
        # logger.info(f"{kv[-1:, :, :].size()}")
        # logger.info(self.mapping_index.size())
        # logger.info(f"{self.mapping_index[-1:, :]}")
        # logger.info(self.config.pad_token_id)
        attention_mask = torch.concat([torch.ones_like(self.mapping_index[:, :1], dtype=dtype),
                                       torch.not_equal(self.mapping_index[:, 1:], self.config.pad_token_id).to(
                                           dtype=dtype)], dim=1)
        # logger.info(f"{attention_mask[-1:,:]}")
        # logger.info(f"{torch.mm(attention_mask[-1:,:],kv[-1,:,:])[0,:10] }")
        # logger.info(kv.permute(0, 2, 1).size())
        # logger.info(attention_mask.size())
        # logger.info(f"{torch.bmm(kv.permute(0, 2, 1), attention_mask[:,:,None]).size()}")
        # logger.info(f"{(torch.matmul(attention_mask, kv))[:1,:]}")
        # logger.info(f"{torch.sum(attention_mask, 1).size()}")

        # logger.info(f"{(torch.bmm(kv.permute(0, 2, 1), attention_mask[:,:,None]))[-1, :10, 0]}")

        kv_table["weight"][-self.num_incremental_embeddings:, :] = torch.div(
            torch.bmm(kv.permute(0, 2, 1), attention_mask[:, :, None]),
            torch.sum(attention_mask.long(), 1)[:, None, None]).squeeze(2)
        # logger.info(f"{kv_table['weight'][-1:, :]}")
        self.lm_head.load_state_dict(kv_table)

    def forward(self, hidden_states: torch.Tensor, mapping_index: Optional[torch.Tensor] = None) -> torch.Tensor:

        # Apply a cross-attention between the weights of incremental vocabulary and the weights of their corresponding subwords (containing themselves).
        # mapping_index stores the indices of the corresponding subwords of each word in the incremental vocabulary.
        # mapping_index size: (num_incremental_embeddings, num_subwords)
        if mapping_index is not None:
            self.mapping_index = mapping_index

        logits = self.lm_head(hidden_states)

        # if self.mapping_index is not None:
        #     if not self.lazy_mapping_weight:
        #         weight = self.get_mapping_weight(hidden_states.dtype, hidden_states.device)
        #         logits = torch.nn.functional.linear(hidden_states, weight)
        #     else:
        #         # if use lazy, parameter of lm_head is staticly stored in self.mapping_lm_head
        #         logits = self.mapping_lm_head(hidden_states)
        # else:
        #     logits = self.lm_head(hidden_states)
        #     # logger.info("Out mapping index")

        return logits


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


class CusVocab_BartLMHeadModel(BartForConditionalGeneration):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [
        r"final_logits_bias",
        r"lm_head.weight",
        "encoder.embed_tokens.weight",
        "decoder.embed_tokens.weight",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.model = BartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        # self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        incremental_vocab_size = config.vocab_size - config.base_vocab_size
        print("incremental_vocab_size:", incremental_vocab_size)
        self.customed_lm_head = Plug_LM_Header(config, incremental_vocab_size)

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)



    def set_output_embeddings(self, embeddings):
        self.customed_lm_head.lm_head.weight = embeddings

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            decoder_head_mask: Optional[torch.Tensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[List[torch.FloatTensor]] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.customed_lm_head(outputs[0])
        lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
            self,
            decoder_input_ids,
            past_key_values=None,
            attention_mask=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            use_cache=None,
            encoder_outputs=None,
            **kwargs
    ):
        # cut decoder_input_ids if past_key_values is used
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past


class Plug_QA_Header(torch.nn.Module):
    def __init__(self, config, num_incremental_embeddings: Optional[int] = 0, _weight: Optional[torch.Tensor] = None,
                 device=None, dtype=None, bias=None) -> None:

        factory_kwargs = {'device': device, 'dtype': dtype, 'bias': bias}
        self.config = config
        self.out_features = config.num_labels
        self.in_features = config.hidden_size
        super(Plug_QA_Header, self).__init__()
        self.num_incremental_embeddings = num_incremental_embeddings
        self.lazy_mapping_weight = config.lazy_mapping_weight
        if num_incremental_embeddings > 0:
            self.c_attn = Conv1D(2 * self.in_features, self.in_features)
            self.q_attn = Conv1D(self.in_features, self.in_features)
            self.c_proj = Conv1D(self.in_features, self.in_features)
            self.resid_dropout = nn.Dropout(config.resid_pdrop if hasattr(config, "resid_pdrop") else config.dropout)
            self.attn_dropout = nn.Dropout(
                config.attn_pdrop if hasattr(config, "attn_pdrop") else config.attention_dropout)
        self.device = device
        self.dtype = dtype
        self.mapping_index = None
        if _weight is None:
            self.qa_head = torch.nn.Linear(self.in_features, self.out_features, **factory_kwargs)
            print("self.qa_head size:", self.qa_head.weight.size(), "device:", self.qa_head.weight.device)
            # self.weight = Parameter(torch.empty((self.out_features - num_incremental_embeddings, self.in_features), **factory_kwargs))
            # self.lm_head = torch.nn.Linear(self.in_features, self.out_features - self.num_incremental_embeddings, **factory_kwargs)
            # if self.num_incremental_embeddings > 0:
            #     # self.weight_incremental = Parameter(torch.empty((num_incremental_embeddings, self.in_features), **factory_kwargs))
            #     self.incremental_lm_head = torch.nn.Linear(self.in_features, self.num_incremental_embeddings, **factory_kwargs)
            #     print("self.incremental_lm_head size:", self.incremental_lm_head.weight.size())
        else:
            assert list(_weight.shape) == [self.out_features, self.in_features], \
                'Shape of weight does not match num_embeddings and embedding_dim'
            self.qa_head = torch.nn.Linear(self.in_features, self.out_features, **factory_kwargs)
            # self.lm_head = torch.nn.Linear(self.in_features, self.out_features - self.num_incremental_embeddings, **factory_kwargs)
            self.qa_head.weight = Parameter(_weight)

        if config.mapping_index_file:
            mapping_index_matrix = json.load(open(config.mapping_index_file, "r"))
            # convert None to 0
            mapping_index_matrix = [[1 if x is None else x for x in row] for row in mapping_index_matrix]
            self.mapping_index = torch.Tensor(mapping_index_matrix).to(self.device).to(torch.long)
            self.mapping_lm_head = torch.nn.Linear(self.in_features, self.out_features, **factory_kwargs)
            # self.tie_mapping_lm_head(self.dtype, self.device)

        # print the weight size of lm_head
        print("qa_head size:", self.qa_head.weight.size())

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        attn_weights = attn_weights / torch.full(
            [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
        )

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def cross_attention(self, query_embeds, kv_embeds, num_heads=4, attention_mask=None, output_attentions=False):
        self.num_heads = num_heads
        self.head_dim = self.in_features // self.num_heads
        query = self.q_attn(query_embeds)
        key, value = self.c_attn(kv_embeds).split(self.in_features, dim=2)
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)
        attn_output, attn_weights = self._attn(query, key, value, attention_mask=attention_mask)
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        outputs = (attn_output)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs

    def get_mapping_weight(self, dtype, device):
        if not hasattr(self, "mapping_index") or self.mapping_index is None:
            return

        device = self.qa_head.weight.device
        q = self.qa_head.weight[-self.num_incremental_embeddings:, None, :]
        kv_table = self.qa_head.weight
        kv = torch.clone(kv_table[self.mapping_index].detach())
        kv.requires_grad = False

        attention_mask = torch.not_equal(self.mapping_index, self.config.pad_token_id).to(device).to(torch.long)
        attention_mask = attention_mask[:, None, None, :]
        attention_mask = attention_mask.to(device)
        attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        attention_mask = (1.0 - attention_mask) * torch.finfo(dtype if dtype is not None else torch.float).min
        outputs = self.cross_attention(q, kv, attention_mask=attention_mask)

        weight_incremental_after_attention = outputs.view(-1, self.in_features)
        # weight = torch.concat([self.lm_head.weight,weight_incremental_after_attention], dim = 0)
        weight = torch.concat(
            [self.qa_head.weight[:-self.num_incremental_embeddings, :], weight_incremental_after_attention], dim=0)

        # self.mapping_lm_head.weight is used for parameter storing. However parameter is not in computation graph, for gradient prop, we use variable weight to calculate loss.
        return weight

    def tie_mapping_lm_head(self, dtype, device):
        if not hasattr(self, "mapping_index") or self.mapping_index is None:
            return

        weight = self.get_mapping_weight(dtype, device)

        self.mapping_lm_head.weight = nn.Parameter(weight)

    def init_lm_head_by_mapping(self, dtype, device):
        if not hasattr(self, "mapping_index") or self.mapping_index is None:
            return
        # logger.info("init_lm_head_by_mapping")
        kv_table = self.qa_head.state_dict()
        # logger.info(f"{(kv_table['weight'])[-1:,:]}")
        kv = kv_table["weight"][self.mapping_index]
        # logger.info(f"{kv[-1:, :, :].size()}")
        # logger.info(self.mapping_index.size())
        # logger.info(f"{self.mapping_index[-1:, :]}")
        # logger.info(self.config.pad_token_id)
        attention_mask = torch.concat([torch.ones_like(self.mapping_index[:, :1], dtype=dtype),
                                       torch.not_equal(self.mapping_index[:, 1:], self.config.pad_token_id).to(
                                           dtype=dtype)], dim=1)
        # logger.info(f"{attention_mask[-1:,:]}")
        # logger.info(f"{torch.mm(attention_mask[-1:,:],kv[-1,:,:])[0,:10] }")
        # logger.info(kv.permute(0, 2, 1).size())
        # logger.info(attention_mask.size())
        # logger.info(f"{torch.bmm(kv.permute(0, 2, 1), attention_mask[:,:,None]).size()}")
        # logger.info(f"{(torch.matmul(attention_mask, kv))[:1,:]}")
        # logger.info(f"{torch.sum(attention_mask, 1).size()}")

        # logger.info(f"{(torch.bmm(kv.permute(0, 2, 1), attention_mask[:,:,None]))[-1, :10, 0]}")

        kv_table["weight"][-self.num_incremental_embeddings:, :] = torch.div(
            torch.bmm(kv.permute(0, 2, 1), attention_mask[:, :, None]),
            torch.sum(attention_mask.long(), 1)[:, None, None]).squeeze(2)
        # logger.info(f"{kv_table['weight'][-1:, :]}")
        self.qa_head.load_state_dict(kv_table)
        # print lm_head size
        print("lm_head size after innit_with_mapping:", self.qa_head.weight.size())

    def forward(self, hidden_states: torch.Tensor, mapping_index: Optional[torch.Tensor] = None) -> torch.Tensor:

        # Apply a cross-attention between the weights of incremental vocabulary and the weights of their corresponding subwords (containing themselves).
        # mapping_index stores the indices of the corresponding subwords of each word in the incremental vocabulary.
        # mapping_index size: (num_incremental_embeddings, num_subwords)
        if mapping_index is not None:
            self.mapping_index = mapping_index

        logits = self.qa_head(hidden_states)
        # print output size
        print("logits size:", logits.size())

        # if self.mapping_index is not None:
        #     if not self.lazy_mapping_weight:
        #         weight = self.get_mapping_weight(hidden_states.dtype, hidden_states.device)
        #         logits = torch.nn.functional.linear(hidden_states, weight)
        #     else:
        #         # if use lazy, parameter of lm_head is staticly stored in self.mapping_lm_head
        #         logits = self.mapping_lm_head(hidden_states)
        # else:
        #     logits = self.lm_head(hidden_states)
        #     # logger.info("Out mapping index")

        return logits


class CusVocab_BartQAModel(BartForQuestionAnswering):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [
        r"qa_outputs_bias",
        r"qa_outputs.weight",
        "encoder.embed_tokens.weight",
        "decoder.embed_tokens.weight",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.model = BartModel(config)
        self.register_buffer("qa_outputs_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        # self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        incremental_vocab_size = config.vocab_size - config.base_vocab_size
        print("incremental_vocab_size:", incremental_vocab_size)
        self.cus_qa_output = Plug_QA_Header(config, incremental_vocab_size)
        print(self.cus_qa_output.qa_head.weight.size())
        # print the input size
        print("hidden_size:", config.hidden_size)
        # print the num labels
        print("num_labels:", config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()
        print(f"qa_head size after post_init: {self.cus_qa_output.qa_head.weight.size()}")

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("qa_outputs_bias", new_bias)

    def get_output_embeddings(self):
        return self.cus_qa_output.qa_head

    def set_output_embeddings(self, embeddings):
        print("embeddings size:", embeddings.size())
        self.cus_qa_output.qa_head.weight = embeddings

    def forward(
            self,
            input_ids: torch.Tensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            decoder_head_mask: Optional[torch.Tensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[List[torch.FloatTensor]] = None,
            start_positions: Optional[torch.LongTensor] = None,
            end_positions: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqQuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (*sequence_length*). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (*sequence_length*). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if start_positions is not None and end_positions is not None:
            use_cache = False

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.cus_qa_output(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (
                         start_logits,
                         end_logits,
                     ) + outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return Seq2SeqQuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)
