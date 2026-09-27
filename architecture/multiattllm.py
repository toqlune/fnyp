"""
MultiAttLLM — improved, channel-independent architecture.

Two encoder paths run in parallel:
  Path A (LLM encoder)       : target features    -> LLMBlock       -> enc_out_target
  Path B (covariate encoder) : non-target features -> linear layer  -> enc_out_other
A fusion decoder then combines both via cross-attention and projects to the
final forecast.

RevIN wraps the target series only (the num_target_channels channels the
model is scored on), normalizing them right before LLMBlock and reversing
that exact transform on the model's final output.

Channel independence (CI): LLMBlock is already channel-independent
internally — every target series is patch-embedded and passed through
GPT-2 on its own, batch and n_vars merged into one dimension. The fusion
decoder is made channel-independent the same way: the num_target_channels
target channels are folded into the batch dimension before decoding, so no
target channel's embedding or projection is ever influenced by another
target channel's values. Weights are shared across channels; each channel
is just processed as its own sequence. Channels are only re-joined at the
very end, right before RevIN denormalization.
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
        self.prediction_length = configs.prediction_length
        self.num_target_channels = configs.num_target_channels

        num_covariates = configs.num_input_channels - self.num_target_channels

        # ── Covariate feature extractor ───────────────────────────────────
        self.feature_extractor = nn.Linear(num_covariates, configs.model_dimension)

        # ── RevIN: reversible instance normalization for target series ────
        self.revin_layer = RevIN(self.num_target_channels, affine=True)

        # ── LLM encoder (target features only) ──────────────────────────────
        self.llm_encoder = LLMBlock(configs)

        # ── Fusion decoder (channel-independent) ─────────────────────────────
        # dec_embedding takes a single channel (c_in=1) at a time, so no
        # target variable's embedding is influenced by another target
        # variable's values — weights are shared, channels are folded into
        # the batch dimension in forecast() below.
        self.dec_embedding = DataEmbedding(
            1, configs.model_dimension, configs.time_embedding_type,
            configs.time_frequency, configs.dropout_rate)
        self.fusion_decoder = Decoder(
            [
                DecoderLayer(
                    # self-attention on the decoder sequence (causal)
                    AttentionLayer(
                        FullAttention(True, configs.attention_scaling_factor,
                                      attention_dropout=configs.dropout_rate, output_attention=False),
                        configs.model_dimension, configs.num_attention_heads),
                    # cross-attention between decoder and covariate encoding
                    AttentionLayer(
                        FullAttention(False, configs.attention_scaling_factor,
                                      attention_dropout=configs.dropout_rate, output_attention=False),
                        configs.model_dimension, configs.num_attention_heads),
                    configs.model_dimension,
                    4 * configs.model_dimension,
                    dropout=configs.dropout_rate,
                    activation=configs.activation_function,
                )
                for _ in range(configs.num_decoder_layers)
            ],
            norm_layer=nn.LayerNorm(configs.model_dimension),
        )

        # maps model_dimension -> 1 (single channel): shared weights, but
        # each channel's projection only ever sees its own decoder output
        self.output_projection = nn.Linear(configs.model_dimension, 1)

    def forecast(self, x_enc, x_mark_dec):
        # split input into target and covariate feature groups — the data
        # loader always places target columns last
        x_enc_other = x_enc[:, :, :-self.num_target_channels]
        x_enc_target = x_enc[:, :, -self.num_target_channels:]

        # normalize target series per-instance; caches stats for the
        # denorm step at the end of this method
        x_enc_target = self.revin_layer(x_enc_target, 'norm')

        # Path A: target features through the frozen-LLM encoder (already
        # channel-independent internally)
        enc_out_target = self.llm_encoder(x_enc_target)
        # (batch, prediction_length + label_sequence_length, num_target_channels)

        # Path B: covariate features through the linear extractor
        enc_out_other = self.feature_extractor(x_enc_other)
        # (batch, lookback_window_length, model_dimension)

        # fold the target channels into the batch dimension, the same way
        # LLMBlock already folds batch*n_vars — so the decoder processes
        # each target channel as its own independent sequence
        batch_size, decoder_seq_len, _ = enc_out_target.shape
        enc_out_target_ci = enc_out_target.permute(0, 2, 1).contiguous().reshape(
            batch_size * self.num_target_channels, decoder_seq_len, 1)

        # time-mark features and the covariate encoding are shared context
        # for every channel — repeat once per channel so batch dims line up
        x_mark_dec_ci = x_mark_dec.repeat_interleave(self.num_target_channels, dim=0)
        enc_out_other_ci = enc_out_other.repeat_interleave(self.num_target_channels, dim=0)

        dec_in = self.dec_embedding(enc_out_target_ci, x_mark_dec_ci)
        dec_out = self.fusion_decoder(dec_in, enc_out_other_ci, x_mask=None, cross_mask=None)
        dec_out = self.output_projection(dec_out)
        # (batch*num_target_channels, decoder_seq_len, 1)

        # un-fold back to (batch, decoder_seq_len, num_target_channels) —
        # plain concatenation, no learned mixing — right before RevIN
        # reverses the normalization
        dec_out = dec_out.reshape(batch_size, self.num_target_channels, decoder_seq_len).permute(0, 2, 1).contiguous()
        return self.revin_layer(dec_out, 'denorm')

    def forward(self, x_enc, x_mark_dec):
        dec_out = self.forecast(x_enc, x_mark_dec)
        # keep only the last prediction_length steps — discard the
        # label_sequence_length warm-up prefix
        return dec_out[:, -self.prediction_length:, :]