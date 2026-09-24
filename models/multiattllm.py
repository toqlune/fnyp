"""
MultiAttLLM — improved, channel-independent architecture.

Two encoder paths run in parallel:
  Path A (LLM encoder)       : target features    -> LLMBlock       -> enc_out_target
  Path B (covariate encoder) : non-target features -> linear layer  -> enc_out_other
A fusion decoder then combines both via cross-attention and projects to the
final forecast.

RevIN wraps the target series only (the c_out channels the model is scored
on), normalizing them right before LLMBlock and reversing that exact
transform on the model's final output.

Channel independence (CI): LLMBlock is already channel-independent
internally — every target series is patch-embedded and passed through
GPT-2 on its own, batch and n_vars merged into one dimension. The fusion
decoder is made channel-independent the same way: the c_out target
channels are folded into the batch dimension before decoding, so no target
channel's embedding or projection is ever influenced by another target
channel's values. Weights are shared across channels; each channel is just
processed as its own sequence. Channels are only re-joined at the very
end, right before RevIN denormalization.
"""
import torch.nn as nn

from modules.attention import AttentionLayer, FullAttention
from modules.decoder import Decoder, DecoderLayer
from modules.embed import DataEmbedding
from modules.llm_block import LLMBlock
from modules.revin import RevIN


class Model(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len
        self.c_out = configs.c_out

        n_covariates = configs.enc_in - self.c_out

        # ── Covariate feature extractor ───────────────────────────────────
        self.feature_extractor = nn.Linear(n_covariates, configs.d_model)

        # ── RevIN: reversible instance normalization for target series ────
        self.revin_layer = RevIN(self.c_out, affine=True)

        # ── LLM encoder (target features only) ──────────────────────────────
        self.LLM_encoder = LLMBlock(configs)

        # ── Fusion decoder (channel-independent) ─────────────────────────────
        # dec_embedding takes a single channel (c_in=1) at a time, so no
        # target variable's embedding is influenced by another target
        # variable's values — weights are shared, channels are folded into
        # the batch dimension in forecast() below.
        self.dec_embedding = DataEmbedding(1, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.selfattention_layer = Decoder(
            [
                DecoderLayer(
                    # self-attention on the decoder sequence (causal)
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    # cross-attention between decoder and covariate encoding
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    4 * configs.d_model,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.d_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
        )

        # maps d_model -> 1 (single channel): shared weights, but each
        # channel's projection only ever sees its own decoder output
        self.out_projection = nn.Linear(configs.d_model, 1)

    def forecast(self, x_enc, x_mark_dec):
        # split input into target and covariate feature groups — the data
        # loader always places target columns last
        x_enc_other = x_enc[:, :, :-self.c_out]
        x_enc_target = x_enc[:, :, -self.c_out:]

        # normalize target series per-instance; caches stats for the
        # denorm step at the end of this method
        x_enc_target = self.revin_layer(x_enc_target, 'norm')

        # Path A: target features through the frozen-LLM encoder (already
        # channel-independent internally)
        enc_out_target = self.LLM_encoder(x_enc_target)
        # (batch, pred_len + label_len, c_out)

        # Path B: covariate features through the linear extractor
        enc_out_other = self.feature_extractor(x_enc_other)
        # (batch, seq_len, d_model)

        # fold the c_out target channels into the batch dimension, the same
        # way LLMBlock already folds batch*n_vars — so the decoder processes
        # each target channel as its own independent sequence
        B, T_out, C = enc_out_target.shape
        enc_out_target_ci = enc_out_target.permute(0, 2, 1).contiguous().reshape(B * C, T_out, 1)

        # time-mark features and the covariate encoding are shared context
        # for every channel — repeat once per channel so batch dims line up
        x_mark_dec_ci = x_mark_dec.repeat_interleave(C, dim=0)
        enc_out_other_ci = enc_out_other.repeat_interleave(C, dim=0)

        dec_in = self.dec_embedding(enc_out_target_ci, x_mark_dec_ci)
        dec_out = self.selfattention_layer(dec_in, enc_out_other_ci, x_mask=None, cross_mask=None)
        dec_out = self.out_projection(dec_out)
        # (batch*c_out, T_out, 1)

        # un-fold back to (batch, T_out, c_out) — plain concatenation, no
        # learned mixing — right before RevIN reverses the normalization
        dec_out = dec_out.reshape(B, C, T_out).permute(0, 2, 1).contiguous()
        return self.revin_layer(dec_out, 'denorm')

    def forward(self, x_enc, x_mark_dec):
        dec_out = self.forecast(x_enc, x_mark_dec)
        # keep only the last pred_len steps — discard the label_len warm-up prefix
        return dec_out[:, -self.pred_len:, :]