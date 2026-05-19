"""CLI for the evaluation framework.

Run an evaluation:
    python -m docflow.eval run --dataset path/to/dataset --providers gemini-3.1-flash-lite,qwen-3-vl-235b
        [--output reports/] [--allow-fallback] [--quiet]

Compare two runs:
    python -m docflow.eval regression previous.json current.json [--output diff.md]

If --dataset has no .truth.json sidecars, the run produces an empty report
with a note. Add truth files (see docflow/eval/truth.py for schema) to grow
the golden dataset.
"""

import argparse
import sys
from pathlib import Path

from docflow.env import load_env_file
from docflow.eval.regression import compare_runs
from docflow.eval.report import write_all
from docflow.eval.runner import run_evaluation


def main(argv: list[str] | None = None) -> int:
    load_env_file(Path(__file__).parent.parent.parent / ".env")

    parser = argparse.ArgumentParser(prog="docflow.eval")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Evaluate a dataset against one or more providers")
    run_p.add_argument("--dataset", required=True, type=Path,
                       help="Directory with invoice files and matching .truth.json sidecars")
    run_p.add_argument("--providers", required=True,
                       help="Comma-separated provider aliases (e.g. 'gemini-3.1-flash-lite,qwen-3-vl-235b')")
    run_p.add_argument("--output", type=Path, default=Path("eval_output"),
                       help="Output directory for reports (default: ./eval_output)")
    run_p.add_argument("--basename", default="eval_report",
                       help="Filename prefix for the four output files")
    run_p.add_argument("--allow-fallback", action="store_true",
                       help="Allow pipeline to fall back when explicit provider hits quota/etc. "
                            "Off by default for benchmark integrity.")
    run_p.add_argument("--quiet", action="store_true", help="Suppress per-file progress lines")

    reg_p = sub.add_parser("regression", help="Diff two run JSON dumps")
    reg_p.add_argument("previous", type=Path)
    reg_p.add_argument("current", type=Path)
    reg_p.add_argument("--output", type=Path, default=None,
                       help="Write the markdown diff to this path. Default: stdout.")

    args = parser.parse_args(argv)

    if args.cmd == "run":
        providers = [p.strip() for p in args.providers.split(",") if p.strip()]
        if not providers:
            parser.error("--providers must list at least one provider alias")
        if not args.dataset.is_dir():
            parser.error(f"dataset not a directory: {args.dataset}")

        result = run_evaluation(
            args.dataset, providers,
            allow_fallback=args.allow_fallback,
            verbose=not args.quiet,
        )

        paths = write_all(result, args.output, basename=args.basename)
        print()
        print(f"Wrote: {paths['md']}")
        print(f"       {paths['csv']}")
        print(f"       {paths['xlsx']}")
        print(f"       {paths['json']}")
        print()
        if result.sample_count == 0:
            print(f"⚠️  No (invoice, truth) pairs found in {args.dataset}.")
            print(f"    Add invoice_001.truth.json sidecars next to invoice_001.pdf to grade them.")
            return 1
        for p, a in result.per_provider.items():
            print(f"  {p}: invoice_acc={a.invoice_accuracy:.0%} "
                  f"crit_acc={a.critical_field_accuracy:.0%} "
                  f"halluc={a.hallucination_rate:.0%} "
                  f"wbc={a.wrong_but_confident_rate:.0%} "
                  f"avg_score={a.avg_quality_score:.2f}")
        return 0

    if args.cmd == "regression":
        md = compare_runs(args.previous, args.current, out=args.output)
        if args.output:
            print(f"Wrote diff to {args.output}")
        else:
            print(md)
        return 0

    parser.error(f"Unknown command: {args.cmd}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
