#!/usr/bin/env python3
# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Batch runner for H9 example scripts.

By default the runner is *incremental*: it only runs examples that have
never been tested, that failed or timed out last time, or whose source has
changed since their last successful run. Results are cached in
``examples/output/run_status.json`` between invocations.

Usage:
    python run_examples.py            # incremental: only stale/failed/new
    python run_examples.py --all      # force the full set (after big changes)
    python run_examples.py --invalidate   # wipe the cache, then run all
    python run_examples.py --dev      # also include ex_10xxx examples
    python run_examples.py --timeout 120
    python run_examples.py --stamp         # update 'Last Tested' lines on success
    python run_examples.py ex0060 ex0062   # run specific examples (always runs them)

Examples are ordered oldest-tested-first, so an interrupted run does not keep
re-testing the early examples at the expense of the later ones.

Examples that require optional libraries (osgeo, cartopy, etc.) are
automatically skipped when the import fails — no hard-coded skip list needed.
Examples that need external data files can be listed in SKIP_DATA_DEPENDENT.
"""

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# Schema version for the status cache; bump to invalidate old caches.
STATUS_SCHEMA = 1

# Examples that can't run without specific data files not in the repo.
# Add filenames (without path) here as needed.
SKIP_DATA_DEPENDENT: set[str] = {
    # 'ex0076_stonehenge.py',    # needs re-consideration (noted in docstring)
    # 'ex0064_vertices.py'       # very slow.
}

# Examples that are intentionally slow and need a higher per-example timeout.
# Value is the suggested minimum timeout in seconds.
SLOW_EXAMPLES: dict[str, int] = {
    'ex0020_plate_glb.py': 300,
    'ex0063_grid.py': 400,
    'ex0063_gridb.py': 400,
    'ex0045_plate_net.py': 200,
    'ex0059_seamstitch.py': 200,
    'ex0060_addresses.py': 5000,  # >120s
    'ex0060u_addresses.py': 5000,  # >120s
    'ex0065_octproj_a.py': 5000,  # >120s
    'ex0065_octproj_c.py': 5000,  # >120s
    'ex0080w_warped_authalics.py': 5000,  # >120s
    'ex0081w_warped_authalics.py': 5000,  # >120s
    'ex0095_smp_grid.py': 400,
    'ex0098_zoom_sequence.py': 5000,  # >120s
    'ex0099_zoom_other.py': 5000,  # >120s
    'ex0101_h9_grid.py': 2000,
    'ex0115_hex_mesh.py': 5000,  # >120s
    'ex0117_hexgrid_data.py': 5000,  # >120s
    'ex0118_mollweide.py': 5000,  # >120s
    'ex0121_tissot_svg.py': 5000,  # >120s
    'ex0250_geotiff.py': 5000,  # >120s
    'ex0253_area.py': 5000,  # >120s
    'ex0260_polygrid.py': 5000,
    'ex0261_polygrid.py': 5000,
}

# ANSI colours — disabled automatically if not a TTY
_tty = sys.stdout.isatty()
def _c(code, text): return f'\033[{code}m{text}\033[0m' if _tty else text
GREEN  = lambda t: _c('32', t)
RED    = lambda t: _c('31', t)
YELLOW = lambda t: _c('33', t)
CYAN   = lambda t: _c('36', t)
BOLD   = lambda t: _c('1',  t)
DIM    = lambda t: _c('2',  t)

def discover(examples_dir: Path, include_dev: bool, filters: list[str]) -> list[Path]:
    pattern = 'ex*.py'
    scripts = sorted(examples_dir.glob(pattern))
    result = []
    for s in scripts:
        name = s.name
        is_dev = name.startswith('ex_')
        if is_dev and not include_dev:
            continue
        if filters and not any(f in name for f in filters):
            continue
        result.append(s)
    return result


def run_one(script: Path, cwd: Path, timeout: int) -> dict:
    """Run a single example. Returns a result dict."""
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = time.perf_counter() - t0
        stderr = proc.stderr.strip()
        stdout = proc.stdout.strip()

        # Auto-detect missing optional dependency
        if proc.returncode != 0:
            missing = _detect_missing_import(stderr)
            if missing:
                return dict(status='skip', reason=f'missing: {missing}',
                            elapsed=elapsed, stdout=stdout, stderr=stderr)
            return dict(status='fail', reason=f'exit {proc.returncode}',
                        elapsed=elapsed, stdout=stdout, stderr=stderr)

        return dict(status='pass', reason='', elapsed=elapsed,
                    stdout=stdout, stderr=stderr)

    except subprocess.TimeoutExpired:
        return dict(status='timeout', reason=f'>{timeout}s',
                    elapsed=timeout, stdout='', stderr='')


def _detect_missing_import(stderr: str) -> str | None:
    """Return the missing module name if stderr contains a ModuleNotFoundError."""
    m = re.search(r"ModuleNotFoundError: No module named '([^']+)'", stderr)
    return m.group(1) if m else None


def _write_log(log_dir: Path, stem: str, result: dict) -> None:
    """Write stdout+stderr to output/logs/<stem>.txt."""
    log_path = log_dir / f'{stem}.txt'
    with log_path.open('w') as f:
        f.write(f'status:  {result["status"]}\n')
        f.write(f'elapsed: {result.get("elapsed", 0):.2f}s\n')
        if result.get('reason'):
            f.write(f'reason:  {result["reason"]}\n')
        stdout = result.get('stdout', '').strip()
        stderr = result.get('stderr', '').strip()
        if stdout:
            f.write('\n--- stdout ---\n')
            f.write(stdout)
            f.write('\n')
        if stderr:
            f.write('\n--- stderr ---\n')
            f.write(stderr)
            f.write('\n')


def _stamp_last_tested(script: Path, version: str) -> None:
    """Update or insert a 'Last Tested' line in the script's docstring."""
    today = datetime.date.today().strftime('%d %b %Y')
    entry = f'{today} {version} (passed)'
    text = script.read_text()
    # If a Last Tested block exists, prepend a new line after it
    m = re.search(r'(Last Tested\s*\n)', text)
    if m:
        insert_pos = m.end()
        new_text = text[:insert_pos] + entry + '\n' + text[insert_pos:]
        script.write_text(new_text)


def _get_version() -> str:
    try:
        import importlib.metadata
        return importlib.metadata.version('hhg9')
    except Exception:
        return '?'


def _load_status(path: Path) -> dict:
    """Load the per-example status cache. Returns {} if missing/incompatible."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    if data.get('schema') != STATUS_SCHEMA:
        return {}
    return data.get('examples', {})


def _save_status(path: Path, examples: dict) -> None:
    """Persist the per-example status cache."""
    data = {'schema': STATUS_SCHEMA, 'examples': examples}
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def _needs_run(script: Path, record: dict | None) -> tuple[bool, str]:
    """Decide whether an example needs (re-)running. Returns (needed, why)."""
    if record is None:
        return True, 'new'
    if record.get('status') in ('fail', 'timeout'):
        return True, record['status']
    try:
        if script.stat().st_mtime > record.get('source_mtime', 0):
            return True, 'changed'
    except OSError:
        return True, 'new'
    return False, record.get('status', 'pass')


def main():
    parser = argparse.ArgumentParser(description='Run H9 example scripts in batch.')
    parser.add_argument('--dev', action='store_true',
                        help='Also run ex_10xxx development examples')
    parser.add_argument('--timeout', type=int, default=120,
                        help='Per-example timeout in seconds (default: 120)')
    parser.add_argument('--stamp', action='store_true',
                        help="Update 'Last Tested' in passing examples")
    parser.add_argument('--all', action='store_true',
                        help='Run the full set, ignoring the incremental cache')
    parser.add_argument('--invalidate', action='store_true',
                        help='Wipe the status cache before running (forces a full re-test)')
    parser.add_argument('filters', nargs='*',
                        help='Only run examples whose filename contains these strings '
                             '(named examples always run, ignoring the cache)')
    args = parser.parse_args()

    root = Path(__file__).parent
    examples_dir = root / 'examples'
    output_dir = examples_dir / 'output'
    output_dir.mkdir(exist_ok=True)
    log_dir = output_dir / 'logs'
    log_dir.mkdir(exist_ok=True)

    status_path = output_dir / 'run_status.json'
    if args.invalidate and status_path.exists():
        status_path.unlink()
        print(DIM(f'Invalidated status cache ({status_path.name})'))
    status = _load_status(status_path)

    scripts = discover(examples_dir, args.dev, args.filters)
    if not scripts:
        print('No matching examples found.')
        sys.exit(0)

    # Select which examples to run. Explicit filters and --all run everything
    # matching; otherwise run only stale/failed/new examples (incremental).
    explicit = bool(args.filters)
    full = explicit or args.all
    to_run: list[tuple[Path, str]] = []   # (script, reason-to-run)
    up_to_date = 0
    data_skipped = 0
    for script in scripts:
        if script.name in SKIP_DATA_DEPENDENT:
            data_skipped += 1
            continue
        if full:
            to_run.append((script, 'forced'))
            continue
        needed, why = _needs_run(script, status.get(script.stem))
        if needed:
            to_run.append((script, why))
        else:
            up_to_date += 1

    # Oldest-tested-first so interrupted runs don't favour the early examples.
    # Never-run examples (ran_at missing → 0.0) sort to the front.
    to_run.sort(key=lambda sw: status.get(sw[0].stem, {}).get('ran_at', 0.0))

    version = _get_version()
    counts = dict(pass_=0, fail=0, skip=0, timeout=0)

    total = len(to_run)
    mode = 'full' if full else 'incremental'
    note = []
    if up_to_date:
        note.append(f'{up_to_date} up-to-date')
    if data_skipped:
        note.append(f'{data_skipped} data-dependent')
    note_str = f'  ({", ".join(note)} skipped)' if note else ''
    print(BOLD(f'\nH9 examples — {mode}: {total} to run  '
               f'(timeout={args.timeout}s){note_str}\n'))
    if total == 0:
        print(DIM('  Nothing to do — everything is up to date.\n'))
        sys.exit(0)

    for i, (script, why) in enumerate(to_run, 1):
        name = script.stem
        why_tag = DIM(f'({why})') if not full else ''
        prefix = f'[{i:2d}/{total}] {name:<35}'

        print(f'{prefix} ', end='', flush=True)
        effective_timeout = max(args.timeout, SLOW_EXAMPLES.get(script.name, 0))
        result = run_one(script, examples_dir, effective_timeout)
        _write_log(log_dir, script.stem, result)
        st = result['status']

        if st == 'pass':
            elapsed_str = f'{result["elapsed"]:.1f}s'
            print(f'{GREEN("PASS")} {DIM(elapsed_str)} {why_tag}')
            counts['pass_'] += 1
            if args.stamp:
                _stamp_last_tested(script, version)
        elif st == 'skip':
            print(f'{YELLOW("SKIP")} {DIM(result["reason"])}')
            counts['skip'] += 1
        elif st == 'timeout':
            print(f'{YELLOW("TIME")} {DIM(result["reason"])}')
            counts['timeout'] += 1
        else:
            print(f'{RED("FAIL")} {DIM(result["reason"])}')
            counts['fail'] += 1
            # Show last few lines of stderr/stdout for quick diagnosis
            for stream in (result['stderr'], result['stdout']):
                if stream:
                    tail = '\n'.join(stream.splitlines()[-6:])
                    print(DIM('       ' + tail.replace('\n', '\n       ')))

        # Update and persist the cache after every example, so an interrupted
        # run keeps its progress and resumes where it left off next time.
        try:
            source_mtime = script.stat().st_mtime
        except OSError:
            source_mtime = 0.0
        status[name] = dict(status=st, elapsed=round(result.get('elapsed', 0.0), 3),
                            reason=result.get('reason', ''), ran_at=time.time(),
                            source_mtime=source_mtime, version=version)
        _save_status(status_path, status)

    # Summary
    p, f, s, t = counts['pass_'], counts['fail'], counts['skip'], counts['timeout']
    print(BOLD(f'\n{"─"*55}'))
    parts = [GREEN(f'{p} passed'), RED(f'{f} failed') if f else DIM('0 failed'),
             YELLOW(f'{s} skipped') if s else DIM('0 skipped'),
             YELLOW(f'{t} timed out') if t else DIM('0 timed out')]
    print('  ' + '  ·  '.join(parts))
    print()

    sys.exit(0 if f == 0 and t == 0 else 1)


if __name__ == '__main__':
    main()
