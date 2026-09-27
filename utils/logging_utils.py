"""
Console output formatting for training/testing runs — a clean, readable
config summary and section headers, aimed at being easy to scan even for
someone without a deep learning background.
"""
from shutil import get_terminal_size

# ANSI bold — renders as real bold text in Jupyter/Colab/most terminals.
# A plain terminal or a log file that doesn't interpret ANSI will instead
# show the raw escape codes around the group name; swap _BOLD/_RESET for
# empty strings if that happens in your environment.
_BOLD = '\033[1m'
_RESET = '\033[0m'

# Config keys added dynamically outside the three config modules (by
# main.py / engine/trainer.py) don't carry group information the way
# module-sourced keys do, so they're assigned a group here explicitly.
# Anything not covered by either falls into an "Other" catch-all.
_DYNAMIC_KEY_GROUPS = {
    'device': 'Environment',
    'save_model': 'Environment',
    'num_input_channels': 'Data',
    'num_target_channels': 'Data',
}
_GROUP_ORDER = ['Data', 'Model', 'Environment', 'Other']


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


def _print_two_columns(entries, col_width):
    """Splits entries evenly between a left and right column and prints
    them aligned to a shared column width."""
    half = (len(entries) + 1) // 2
    left_col, right_col = entries[:half], entries[half:]

    for i in range(half):
        left = left_col[i]
        right = right_col[i] if i < len(right_col) else ''
        print(f"{left.ljust(col_width)}{right}")


def _print_group_header(title):
    underline = '.' * max(len(title), 8)
    print(f"{_BOLD}{title}{_RESET}")
    print(underline)
    print()


def print_config_table(configs, groups=None, exclude=()):
    """Prints every attribute on `configs` as a clean two-column table,
    grouped into subsections (Data / Model / Environment / Other) with a
    bolded, underlined subsection heading. `groups` maps each key to its
    section name (see main.py:build_config); keys missing from it fall
    back to _DYNAMIC_KEY_GROUPS, then to "Other". `exclude` skips
    attributes not worth showing."""
    groups = groups or {}

    by_group = {}
    for key, value in vars(configs).items():
        if key.startswith('_') or key in exclude:
            continue
        group = groups.get(key) or _DYNAMIC_KEY_GROUPS.get(key) or 'Other'
        by_group.setdefault(group, []).append((key, _format_value(value)))

    if not by_group:
        return

    ordered_groups = [g for g in _GROUP_ORDER if g in by_group]
    ordered_groups += [g for g in by_group if g not in ordered_groups]

    # One shared column width for the whole table (not recomputed per
    # group), so every section lines up the same way regardless of how
    # long its longest entry happens to be.
    all_entries = [
        f"{key}: {value}"
        for group in ordered_groups
        for key, value in sorted(by_group[group])
    ]
    col_width = max(len(entry) for entry in all_entries) + 4

    print_section("CONFIGS")

    for i, group in enumerate(ordered_groups):
        items = sorted(by_group[group])
        entries = [f"{key}: {value}" for key, value in items]

        _print_group_header(group)
        _print_two_columns(entries, col_width)

        if i != len(ordered_groups) - 1:
            print()
            print()

    print()