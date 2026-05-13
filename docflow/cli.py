import argparse
import sys
from pathlib import Path

from docflow.env import load_env_file
from docflow.output import write_excel
from docflow.pipeline import NoExtractorFound, extract


def main() -> int:
    load_env_file()
    parser = argparse.ArgumentParser(prog="docflow")
    parser.add_argument("input", type=Path, nargs="?", help="PDF/image file or folder")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output file (single mode) or output folder (batch mode)",
    )
    parser.add_argument(
        "-p",
        "--provider",
        default="auto",
        help="Extraction provider: auto (cascade), azure, pdfplumber, mistral, claude, "
             "gemini, gemini-3.1-flash-lite, gemini-2.5-pro",
    )
    parser.add_argument(
        "--list-providers",
        action="store_true",
        help="Show all available providers and which are configured",
    )
    args = parser.parse_args()

    if args.list_providers:
        from docflow.pipeline import list_providers

        print("Provider          Status      Notes")
        print("-" * 60)
        for alias, ok, note in list_providers():
            mark = "✓ ready " if ok else "✗ missing"
            print(f"  {alias:<15} {mark}   {note}")
        return 0

    if args.input is None:
        parser.error("input is required (or use --list-providers)")

    if not args.input.exists():
        print(f"error: {args.input} not found", file=sys.stderr)
        return 2

    if args.input.is_dir():
        from docflow.batch import run_batch

        output_dir = args.output or Path("output")
        results = run_batch(args.input, output_dir, provider=args.provider)
        if not results:
            print(f"no supported files found in {args.input}")
            return 1
        ok = sum(1 for r in results if r.error is None)
        err = sum(1 for r in results if r.error is not None)
        print(f"\nbatch complete: {ok} ok, {err} errors")
        print(f"output: {output_dir / 'Фактури.xlsx'}")
        return 0 if err == 0 else 1

    try:
        doc = extract(args.input, provider=args.provider)
    except NoExtractorFound as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    from docflow.registry import SupplierRegistry
    from docflow.validators import validate

    registry = SupplierRegistry()
    enrich_findings = []
    if doc.invoice:
        doc.invoice, enrich_findings = registry.enrich(doc.invoice)

    output_path = args.output or Path("output") / f"{args.input.stem}.xlsx"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_excel(doc, output_path)

    print(f"extracted {len(doc.tables)} table(s) from {doc.page_count} page(s)")
    print(f"output: {output_path}")

    if enrich_findings:
        print("\nregistry enrichment:")
        for rf in enrich_findings:
            marker = {"ok": "✓", "warning": "!", "error": "✗"}.get(rf.level, "?")
            print(f"  {marker} {rf.code}: {rf.message}")

    findings = validate(doc)
    if findings:
        print("\nvalidation:")
        for f in findings:
            marker = {"ok": "✓", "warning": "!", "error": "✗"}.get(f.level, "?")
            val = f" [{f.value}]" if f.value else ""
            print(f"  {marker} {f.code}: {f.message}{val}")

    if doc.invoice:
        record_findings = registry.record(doc.invoice)
        if record_findings:
            print("\nregistry update:")
            for rf in record_findings:
                marker = {"ok": "✓", "warning": "!", "error": "✗"}.get(rf.level, "?")
                print(f"  {marker} {rf.code}: {rf.message}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
