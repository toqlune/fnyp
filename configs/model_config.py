"""
Model configuration — MultiAttLLM architecture hyperparameters.

A line-by-line comparison against the paper's reported hyperparameters
is planned separately.

NOTE: model_dimension/feedforward_dimension below (32/64) match this
codebase's actual training config, but the paper's Table 3 reports
d_model=64, d_ff=128 for MultiAttLLM. num_llm_layers and
num_decoder_layers do match the paper. Flagging this rather than
silently resolving it either way.
"""

# ── Core dimensions ──────────────────────────────────────────────────────
model_dimension = 32
feedforward_dimension = model_dimension * 2  # 64 — see discrepancy note above
num_attention_heads = 8
num_decoder_layers = 4    # CI decoder layer stack depth (renamed from `d_layers`)
num_llm_layers = 6        # transformer blocks used from GPT-2 (of 12 available)

# ── Improved-architecture addition (M2: text prototypes) ──────────────────
num_text_prototypes = 64

# ── Sequence patching (LLM encoder input) ─────────────────────────────────
patch_length = 16
patch_stride = 8

# ── Shared building blocks ───────────────────────────────────────────────
dropout_rate = 0.1
activation_function = 'gelu'
attention_scaling_factor = 1   # used by the fusion decoder
time_embedding_type = 'timeF'  # time-feature encoding style — see pending-cascade
                               # note in utils/data_loader.py and models/multiattllm.py