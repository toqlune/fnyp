"""
Environment / training-run configuration: everything that governs how a
run executes (device, epochs, optimizer schedule, checkpointing) rather
than what data it uses or what the model looks like.

Variable names are carried over as-is for now; a readability pass is
planned separately.
"""

# ── Run identity ─────────────────────────────────────────────────────────
model_id = 'test'   # placeholder run label — naming scheme to be revisited
is_training = True
num_runs = 1         # repeat the full train+test cycle this many times
                      # (renamed from the original `itr`)

# ── Reproducibility / device ─────────────────────────────────────────────
fix_seed = 4213
use_gpu = True
gpu = 0

# ── Checkpointing ────────────────────────────────────────────────────────
checkpoints = './results/'

# ── Training loop ─────────────────────────────────────────────────────────
train_epochs = 20
batch_size = 16
num_workers = 1
patience = 6         # early-stopping patience, in epochs

# ── Optimizer / learning-rate schedule ────────────────────────────────────
learning_rate = 0.01
lradj = 'PEMS'        # decay schedule — see utils/tools.py:adjust_learning_rate
pct_start = 0.2       # OneCycleLR warmup fraction (used when lradj != 'COS')
use_amp = False       # mixed-precision training

# ── Validation ───────────────────────────────────────────────────────────
val = True            # use the held-out val split; falls back to test if False