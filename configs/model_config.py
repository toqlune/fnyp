"""
Model configuration — MultiAttLLM architecture hyperparameters.

Variable names are carried over as-is from the original codebase for now;
a readability pass (renaming) and a line-by-line comparison against the
paper's reported hyperparameters are both planned separately.

NOTE: d_model/d_ff below (32/64) match this codebase's actual training
config, but the paper's Table 3 reports d_model=64, d_ff=128 for
MultiAttLLM. llm_layers and d_layers do match the paper. Flagging this
rather than silently resolving it either way.
"""

# ── Core dimensions ──────────────────────────────────────────────────────
d_model = 32
d_ff = d_model * 2  # 64 — see discrepancy note above
n_heads = 8
d_layers = 4         # CI decoder layer stack depth
llm_layers = 6       # transformer blocks used from GPT-2 (of 12 available)

# ── Improved-architecture addition (M2: text prototypes) ──────────────────
num_text_prototypes = 64

# ── Sequence patching (LLM encoder input) ─────────────────────────────────
patch_len = 16
stride = 8

# ── Shared building blocks ───────────────────────────────────────────────
dropout = 0.1
activation = 'gelu'
factor = 1         # attention scaling factor, used by the fusion decoder
embed = 'timeF'    # time-feature encoding style — see pending-cascade note
                   # in utils/data_loader.py and models/multiattllm.py