"""
LLMBlock — the frozen-LLM pipeline for target features (paper Components
①②③, reworked): a learned text-prototype bank replaces the original
vocabulary projection, patches attend to that bank via cross-attention,
then flow through a frozen, pretrained GPT-2.

1. Text-Prototype Bank : a small, directly-learned dictionary of
   domain-relevant temporal primitives (nn.Parameter, not a projection of
   GPT-2's vocabulary — see the comment below for why)
2. Patch Embedding     : splits the input time series into overlapping patches
3. Cross-Attention     : aligns patches with the prototype bank, through a
   GLU-gated bottleneck (see modules.cross_attention.CrossAttentionLayer)
4. Frozen GPT-2        : processes the reprogrammed embeddings. Weights are
   downloaded from Hugging Face's official 'openai-community/gpt2'
   repository on first run, cached locally under configs.gpt2_weights_dir,
   and never updated — no separately fine-tuned checkpoint is used, and no
   tokenizer is loaded, since patches are fed directly to GPT-2 via
   `inputs_embeds`, bypassing its text embedding layer entirely.
5. Output Projection   : reshape and project back to the forecast horizon
"""
import torch
import torch.nn as nn
import transformers
from transformers import GPT2Config, GPT2Model

from modules.cross_attention import CrossAttentionLayer
from modules.embed import PatchEmbedding
from modules.flatten_head import FlattenHead

transformers.logging.set_verbosity_error()  # suppress verbose HF loading messages

GPT2_CHECKPOINT = 'openai-community/gpt2'


class LLMBlock(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.device = configs.device
        self.prediction_length = configs.prediction_length
        self.feedforward_dimension = configs.feedforward_dimension
        self.patch_length = configs.patch_length
        self.patch_stride = configs.patch_stride

        # ── Frozen GPT-2 backbone ──────────────────────────────────────────
        # cache_dir makes this a one-time download: subsequent runs load
        # straight from configs.gpt2_weights_dir instead of re-fetching.
        gpt2_config = GPT2Config.from_pretrained(GPT2_CHECKPOINT, cache_dir=configs.gpt2_weights_dir)
        gpt2_config.num_hidden_layers = configs.num_llm_layers
        gpt2_config.output_attentions = True
        gpt2_config.output_hidden_states = True
        self.llm_model = GPT2Model.from_pretrained(
            GPT2_CHECKPOINT, config=gpt2_config, cache_dir=configs.gpt2_weights_dir)

        # Read off the loaded model's own config rather than a hand-set
        # value, so it can never drift from what GPT-2 actually reports.
        self.d_llm = gpt2_config.n_embd

        for param in self.llm_model.parameters():
            param.requires_grad = False
        n_params = sum(p.numel() for p in self.llm_model.parameters())
        print(f"GPT-2 frozen: 0/{n_params:,} parameters trainable")

        # ── Patch embedding ──────────────────────────────────────────────
        self.patch_embedding = PatchEmbedding(
            configs.model_dimension, self.patch_length, self.patch_stride, configs.dropout_rate)
        # number of patches produced per series, e.g. (72 - 16) / 8 + 2 = 9
        self.num_patches = int(
            (configs.lookback_window_length - self.patch_length) / self.patch_stride + 2)

        # ── Text-Prototype Bank ───────────────────────────────────────────
        # Replaces the original vocabulary-projection approach: rather than
        # compressing GPT-2's ~50k-token vocabulary down to a smaller space,
        # this learns a small (num_prototypes, d_llm) dictionary of
        # "temporal primitive" embeddings from scratch, specialized purely
        # for what cross-attention with the time-series patches needs.
        self.text_prototypes = nn.Parameter(
            torch.randn(configs.num_text_prototypes, self.d_llm) * 0.02)

        # ── Cross-attention + output projection ───────────────────────────
        self.cross_attention_layer = CrossAttentionLayer(
            configs.model_dimension, configs.num_attention_heads, self.feedforward_dimension, self.d_llm)
        flattened_input_dim = self.feedforward_dimension * self.num_patches
        self.output_projection = FlattenHead(
            flattened_input_dim, self.prediction_length + configs.label_sequence_length,
            head_dropout=configs.dropout_rate)

        self.to(self.device)

    def forward(self, x_enc):
        # Step 1: read the learned text-prototype bank
        text_prototypes = self.text_prototypes

        # Step 2: convert time series to patches — permute to
        # (batch, num_channels, seq_len), required by PatchEmbedding.
        #
        # NOTE: the .to(torch.bfloat16) below is currently a no-op as far as
        # memory savings go — TokenEmbedding.forward() (used inside
        # PatchEmbedding) immediately calls x.float() on its input before
        # the Conv1d, converting straight back to float32. Left as-is
        # pending a decision: either drop this cast as dead code, or
        # actually thread bf16 through (would need the conv's own weights
        # in bf16 too, plus an explicit cast back to float32 before this
        # reaches the cross-attention/GPT-2 stack).
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, num_channels = self.patch_embedding(x_enc.to(torch.bfloat16))
        # enc_out: (batch * num_channels, num_patches, model_dimension)

        # Step 3: cross-attention reprogramming (paper Eq. 2-4), GLU-gated
        enc_out = self.cross_attention_layer(enc_out, text_prototypes, text_prototypes)
        # enc_out: (batch * num_channels, num_patches, d_llm)

        # Step 4: frozen GPT-2 (paper Eq. 5). `inputs_embeds` bypasses
        # GPT-2's own embedding layer since enc_out is already an embedding.
        dec_out = self.llm_model(inputs_embeds=enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.feedforward_dimension]  # keep only the first feedforward_dimension dims

        # Step 5: reshape and project to the forecast horizon
        dec_out = torch.reshape(dec_out, (-1, num_channels, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()
        dec_out = self.output_projection(dec_out[:, :, :, -self.num_patches:])
        return dec_out.permute(0, 2, 1).contiguous()