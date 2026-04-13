set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: $0 <grow.py args...> <remote_jobdir>" >&2
  exit 2
fi

remote_jobdir=${!#}
grow_arg_count=$(($# - 1))
grow_args=("${@:1:$grow_arg_count}")

GROW="code/grow.py"
python - "${grow_args[@]}" <<'PY'
import argparse
from pathlib import Path
import sys

grow_args = sys.argv[1:]

def existing_dir(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise argparse.ArgumentTypeError(f"directory does not exist: {path}")
    if not p.is_dir():
        raise argparse.ArgumentTypeError(f"not a directory: {path}")
    return path

def dinucleotide_sequence(seq: str) -> str:
    seq = seq.strip().upper()
    if len(seq) != 2:
        raise argparse.ArgumentTypeError("dinucleotide sequence must have length 2")
    allowed = set("ACGU")
    if any(ch not in allowed for ch in seq):
        raise argparse.ArgumentTypeError(
            "dinucleotide sequence must contain only A/C/G/U"
        )
    return seq

def pdb_code(code: str) -> str:
    code = code.strip()
    if len(code) != 4 or not code[0].isdigit() or not code[1:].isalnum():
        raise argparse.ArgumentTypeError(
            "PDB codes must be 4 chars: one digit + 3 alphanumeric characters"
        )
    return code.upper()

def resolve_growth_layout(source_sequence: str, target_sequence: str, direction: str):
    if direction == "forward":
        return source_sequence, target_sequence
    return target_sequence, source_sequence

parser = argparse.ArgumentParser(
    prog="grow",
    description="Grow a source pose pool into a target pose pool using pooled trace-SD matching.",
)
parser.add_argument("--debug", action="store_true", help="Show a full traceback on errors.")
parser.add_argument(
    "--source-poses",
    required=True,
    type=existing_dir,
    help="Source pose directory containing poses-*.npy(.zst) and offsets-*.dat.",
)
parser.add_argument(
    "--source-sequence",
    required=True,
    type=dinucleotide_sequence,
    help="Source dinucleotide sequence.",
)
parser.add_argument(
    "--target-sequence",
    required=True,
    type=dinucleotide_sequence,
    help="Target dinucleotide sequence.",
)
parser.add_argument(
    "--direction",
    required=True,
    choices=("forward", "backward"),
    help="Growth direction relative to the source fragment.",
)
parser.add_argument(
    "--crmsd",
    required=True,
    type=float,
    help="cRMSD threshold for conformer pre-filtering.",
)
parser.add_argument(
    "--ov-rmsd",
    required=True,
    type=float,
    help="Whole-pose RMSD threshold.",
)
parser.add_argument(
    "--output",
    required=True,
    help="Output directory where poses-*.npy.zst and offsets-*.dat are written.",
)
parser.add_argument(
    "--max-poses-per-chunk",
    type=int,
    default=30_000_000,
    metavar="N",
    help="Maximum number of poses per output file pair (default: 30000000).",
)
parser.add_argument(
    "--test-seed",
    type=int,
    default=0,
    help="Random seed for test options (default: 0).",
)
parser.add_argument(
    "--test-conformers",
    type=int,
    default=None,
    metavar="N",
    help="If set, reduce the target conformer library to N elements selected at random.",
)
parser.add_argument(
    "--test-rotamers",
    type=int,
    default=None,
    metavar="M",
    help="If set, reduce target conformer rotamer lists to M shared positions.",
)
parser.add_argument(
    "--pdb-exclude",
    nargs="+",
    default=[],
    type=pdb_code,
    metavar="PDB",
    help="One or more PDB codes to exclude from the target fragment library and cRMSD matrix.",
)
args = parser.parse_args(grow_args)

try:
    if args.max_poses_per_chunk <= 0:
        raise ValueError("--max-poses-per-chunk must be positive")
    if args.crmsd < 0.0:
        raise ValueError("--crmsd must be non-negative")
    if args.ov_rmsd <= 0.0:
        raise ValueError("--ov-rmsd must be positive")

    crmsd_ab, crmsd_bc = resolve_growth_layout(
        args.source_sequence,
        args.target_sequence,
        args.direction,
    )
    if crmsd_ab[1] != crmsd_bc[0]:
        raise ValueError(
            f"Source and target sequences do not overlap for {args.direction} growth: "
            f"{args.source_sequence}/{args.target_sequence}"
        )
except ValueError as exc:
    if getattr(args, "debug", False):
        raise
    print(f"error: {exc}", file=sys.stderr)
    raise SystemExit(1)
PY

seamless-run -vvv -y --dry --write-remote-job "$remote_jobdir" \
  -i "$GROW" \
  -I "${GROW}.DEPS.txt" \
  --conda alaric \
  python -u "$GROW" "${grow_args[@]}"
