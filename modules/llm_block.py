"""
LLMBlock — the frozen-LLM pipeline for target features (paper Components
①②③). It runs in one of two modes, chosen by `use_word_projection`:

Improved (default, use_word_projection=False) — reworked ①②:
  a learned text-prototype bank replaces the original vocabulary
  projection, and patches attend to that bank through a GLU-gated
  cross-attention (M2).

Base (use_word_projection=True) — the paper's original ①②:
  GPT-2's whole vocabulary embedding matrix is linearly projected down to
  `configs.word_projection_size` word vectors (paper Eq. 1), and patches
  attend to those through plain cross-attention with no gate.

Everything else is shared by both modes:

1. Source embeddings   : what the patches attend to — the text-prototype
   bank (improved: a directly-learned nn.Parameter, not a projection of
   GPT-2's vocabulary) or the projected vocabulary (base)
2. Patch Embedding     : splits the input time series into overlapping patches
3. Cross-Attention     : aligns patches with the source embeddings (GLU-gated
   in improved mode, see modules.cross_attention.CrossAttentionLayer)
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

    def __init__(self, configs, use_word_projection=False):
        super().__init__()
        self.device = configs.device
        self.use_word_projection = use_word_projection
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

        # ── Source embeddings the patches attend to ───────────────────────
        if use_word_projection:
            # Base (paper ①, Eq. 1): compress GPT-2's ~50k-token vocabulary
            # into a small set of word vectors. The linear layer runs over the
            # *vocabulary* axis: (d_llm, vocab_size) -> (d_llm, word_projection_size).
            vocab_size = self.llm_model.get_input_embeddings().weight.shape[0]
            self.word_projection = nn.Linear(vocab_size, configs.word_projection_size)
        else:
            # Improved (M2): replaces the vocabulary-projection approach. Rather
            # than compressing GPT-2's ~50k-token vocabulary down to a smaller
            # space, this learns a small (num_prototypes, d_llm) dictionary of
            # "temporal primitive" embeddings from scratch, specialized purely
            # for what cross-attention with the time-series patches needs.
            self.text_prototypes = nn.Parameter(
                torch.randn(configs.num_text_prototypes, self.d_llm) * 0.02)

        # ── Cross-attention + output projection ───────────────────────────
        # The GLU gate is part of the improved model only; base mode uses the
        # paper's plain cross-attention.
        self.cross_attention_layer = CrossAttentionLayer(
            configs.model_dimension, configs.num_attention_heads, self.feedforward_dimension, self.d_llm,
            use_gate=not use_word_projection)
        flattened_input_dim = self.feedforward_dimension * self.num_patches
        self.output_projection = FlattenHead(
            flattened_input_dim, self.prediction_length + configs.label_sequence_length,
            head_dropout=configs.dropout_rate)

        self.to(self.device)

    def forward(self, x_enc):
        # Step 1: the source embeddings the patches will attend to
        if self.use_word_projection:
            # frozen (vocab_size, d_llm) matrix from GPT-2, projected on every
            # pass so the projection weights receive gradients
            word_embeddings = self.llm_model.get_input_embeddings().weight
            source_embeddings = self.word_projection(word_embeddings.permute(1, 0)).permute(1, 0)
            # (word_projection_size, d_llm)
        else:
            source_embeddings = self.text_prototypes
            # (num_text_prototypes, d_llm)

        # Step 2: convert time series to patches — permute to
        # (batch, num_channels, seq_len), required by PatchEmbedding.
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, num_channels = self.patch_embedding(x_enc)
        # enc_out: (batch * num_channels, num_patches, model_dimension)

        # Step 3: cross-attention reprogramming (paper Eq. 2-4); GLU-gated
        # in improved mode, plain in base mode
        enc_out = self.cross_attention_layer(enc_out, source_embeddings, source_embeddings)
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