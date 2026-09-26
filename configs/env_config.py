"""
Environment / training-run configuration: everything that governs how a
run executes (device, epochs, optimizer schedule, checkpointing) rather
than what data it uses or what the model looks like.
"""

# ── Run identity ─────────────────────────────────────────────────────────
run_label = 'test'   # placeholder run label — naming scheme to be revisited
is_training_mode = True
num_runs = 1          # repeat the full train+test cycle this many times
                       # (renamed from the original `itr`)

# ── Reproducibility / device ─────────────────────────────────────────────
random_seed = 4213
use_gpu = True
gpu_device_id = 0

# ── Checkpointing ────────────────────────────────────────────────────────
checkpoint_dir = './results/'

# ── Training loop ─────────────────────────────────────────────────────────
num_train_epochs = 20
batch_size = 16
num_data_loader_workers = 1
early_stopping_patience = 6   # early-stopping patience, in epochs

# ── Optimizer / learning-rate schedule ────────────────────────────────────
learning_rate = 0.01
lr_schedule_type = 'PEMS'      # decay schedule — see utils/tools.py:adjust_learning_rate
onecycle_warmup_fraction = 0.2 # OneCycleLR warmup fraction (used when lr_schedule_type != 'COS')
use_mixed_precision = False    # mixed-precision training

# ── Validation ───────────────────────────────────────────────────────────
use_validation_split = True    # use the held-out val split; falls back to test if False