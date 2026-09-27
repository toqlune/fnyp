"""
Console output formatting for training/testing runs — a clean, readable
config summary and section headers, aimed at being easy to scan even for
someone without a deep learning background.
"""
from shutil import get_terminal_size


def _rule(title, width=None):
    width = width or min(get_terminal_size((100, 20)).columns, 100)
    label = f" {title} "
    dashes = max((width - len(label)) // 2, 5)
    return ('─' * dashes) + label + ('─' * dashes)


def print_section(title):
    """Prints a horizontal rule with a centered title, e.g.
    ───────── CONFIGS ─────────"""
    print(f"\n{_rule(title)}\n")


def print_run_header(stage, configs, highlight_keys):
    """Prints a section header tagged with whichever config value(s) vary
    between experiment runs, e.g.
    ───────── TRAINING MODEL [num_text_prototypes=512] ─────────
    `highlight_keys` is the same list used to name saved-model/results
    files, so the header always matches what the filenames say."""
    tag = ", ".join(f"{key}={getattr(configs, key)}" for key in highlight_keys)
    print_section(f"{stage} [{tag}]")


def _format_value(value):
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (list, tuple)):
        if len(value) <= 4:
            return ', '.join(str(v) for v in value)
        preview = ', '.join(str(v) for v in value[:3])
        return f"{len(value)} items ({preview}, ...)"
    return str(value)


def print_config_table(configs, exclude=()):
    """Prints every attribute on `configs` as a clean two-column table,
    sorted alphabetically. `exclude` skips attributes not worth showing."""
    items = sorted(
        (key, _format_value(value)) for key, value in vars(configs).items()
        if not key.startswith('_') and key not in exclude
    )
    entries = [f"{key}: {value}" for key, value in items]
    if not entries:
        return

    col_width = max(len(entry) for entry in entries) + 4
    half = (len(entries) + 1) // 2
    left_col, right_col = entries[:half], entries[half:]

    print_section("CONFIGS")
    for i in range(half):
        left = left_col[i]
        right = right_col[i] if i < len(right_col) else ''
        print(f"{left.ljust(col_width)}{right}")
    print()