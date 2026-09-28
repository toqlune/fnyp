"""
MultiAttLLM — base architecture (the paper's original design), i.e. the
same pipeline as architecture/multiattllm.py with the four improvements
switched off. Selected with `python main.py --base-model` (`-bm`).

Same pipeline as the improved model:
  Path A (LLM encoder)       : target features    -> LLMBlock       -> enc_out_target
  Path B (covariate encoder) : non-target features -> linear layer  -> enc_out_other
  A fusion decoder combines both via cross-attention and projects to the forecast.

What is turned off, relative to the improved model:

  M1  RevIN                  no instance normalization anywhere; the targets
                             are only globally Z-scored by the data loader
                             (paper Sec. 3.2), and the prediction is
                             inverse-transformed at evaluation time.
  M2  Text-prototype bank    LLMBlock(use_word_projection=True): GPT-2's
      + GLU gate             vocabulary embedding matrix is projected down to
                             `word_projection_size` word vectors (paper ①,
                             Eq. 1) and attended to with plain cross-attention.
  M3  Channel independence   the target channels stay together: the decoder
                             embedding sees all of them at once
                             (c_in = num_target_channels) and the output
                             projection predicts all of them jointly, so
                             channels can influence each other, as in the
                             original design.
  M4  Adaptive gated fusion  DecoderLayer(use_adaptive_gated_fusion=False):
                             plain additive residual around cross-attention.

Hyperparameters (model dimension, layers, heads, dropout, ...) come from the
same config as the improved model, so the two differ only in the four
switches above.

Forward signature and output shape match the improved model exactly
(x_enc, x_mark_dec) -> (batch, prediction_length, num_target_channels), so
engine/trainer.py treats both identically.
"""
import torch.nn as nn

from modules.attention import AttentionLayer, FullAttention
from modules.decoder import Decoder, DecoderLayer
from modules.embed import DataEmbedding
from modules.llm_block import LLMBlock


class BaseModel(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.prediction_length = configs.prediction_length
        self.num_target_channels = configs.num_target_channels

        num_covariates = configs.num_input_channels - self.num_target_channels

        # ── Covariate feature extractor (paper ④) ─────────────────────────
        # Same single linear layer as the improved model.
        self.feature_extractor = nn.Linear(num_covariates, configs.model_dimension)

        # ── LLM encoder (paper ①②③), base mode ─────────────────────────────
        # Word projection + plain cross-attention + frozen GPT-2. No RevIN
        # around it (M1 off).
        self.llm_encoder = LLMBlock(configs, use_word_projection=True)

        # ── Fusion decoder (channel-dependent: M3 off) ─────────────────────
        # All target channels enter the embedding together, so the embedding
        # convolution mixes them.
        self.dec_embedding = DataEmbedding(
            self.num_target_channels, configs.model_dimension, configs.time_embedding_type,
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
                    use_adaptive_gated_fusion=False,  # M4 off
                )
                for _ in range(configs.num_decoder_layers)
            ],
            norm_layer=nn.LayerNorm(configs.model_dimension),
        )

        # maps model_dimension -> all target channels jointly (M3 off)
        self.output_projection = nn.Linear(configs.model_dimension, self.num_target_channels)

    def forecast(self, x_enc, x_mark_dec):
        # split input into target and covariate feature groups — the data
        # loader always places target columns last
        x_enc_other = x_enc[:, :, :-self.num_target_channels]
        x_enc_target = x_enc[:, :, -self.num_target_channels:]

        # Path A: target features through the frozen-LLM encoder
        enc_out_target = self.llm_encoder(x_enc_target)
        # (batch, prediction_length + label_sequence_length, num_target_channels)

        # Path B: covariate features through the linear extractor
        enc_out_other = self.feature_extractor(x_enc_other)
        # (batch, lookback_window_length, model_dimension)

        # decoder input is the LLM output with every channel embedded together
        dec_in = self.dec_embedding(enc_out_target, x_mark_dec)
        dec_out = self.fusion_decoder(dec_in, enc_out_other, x_mask=None, cross_mask=None)
        return self.output_projection(dec_out)
        # (batch, prediction_length + label_sequence_length, num_target_channels)

    def forward(self, x_enc, x_mark_dec):
        dec_out = self.forecast(x_enc, x_mark_dec)
        # keep only the last prediction_length steps — discard the
        # label_sequence_length warm-up prefix
        return dec_out[:, -self.prediction_length:, :]