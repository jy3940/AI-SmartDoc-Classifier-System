"AI SmartDoc Classifier System - Main application entry point"

import argparse


def main():
    "Parse args and run GUI/CLI"
    parser = argparse.ArgumentParser(
        description="AI SmartDoc Classifier - Intelligent Document Organization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="",
    )

    parser.add_argument('--gui', action='store_true', help='Launch GUI (default)')
    parser.add_argument('--cli', action='store_true', help='Command-line mode')
    parser.add_argument('--folder', type=str, help='Folder to process')
    parser.add_argument('--file', type=str, help='File to process')
    parser.add_argument('--output', type=str, help='Output directory for organized files (CLI mode)')
    parser.add_argument('--move', action='store_true', help='Move files (default: copy)')
    parser.add_argument('--model', type=str, default='ensemble', help='ML model to use')

    args = parser.parse_args()

    # Default to GUI if no mode specified
    if not args.cli:
        args.gui = True

    if args.gui:
        print("Launching AI SmartDoc Classifier System...")
        from src.gui import run_gui
        run_gui()
        return

    # CLI mode
    from src.classifier import DocClassifier

    classifier = DocClassifier()
    mode = "move" if args.move else "copy"

    if args.folder:
        print(f"Processing folder: {args.folder}")
        results = classifier.classify_folder(args.folder, mode=mode, output_dir=args.output)
    elif args.file:
        print(f"Processing file: {args.file}")
        result = classifier.classify_file(args.file, mode=mode, output_dir=args.output)
        status = result.get("status")
        category = result.get("category") or "Unknown"
        summary = {category: 1} if status == "success" else {"Error": 1}
        results = {
            "total": 1,
            "successful": 1 if status == "success" else 0,
            "failed": 0 if status == "success" else 1,
            "files": [result],
            "summary": summary,
        }
    else:
        parser.print_help()
        return

    total = int(results.get("total") or results.get("processed") or 0)
    successful = int(results.get("successful") or 0)
    failed = int(results.get("failed") or 0)

    print(f"\nTotal files: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    if results.get("cancelled"):
        print("Cancelled: Yes")
    print("Category breakdown:")
    for cat, count in sorted(results.get('summary', {}).items()):
        print(f"  - {cat}: {count}")

if __name__ == "__main__":
    main()
