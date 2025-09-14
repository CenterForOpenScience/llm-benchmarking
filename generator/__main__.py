# unified CLI
import argparse, os, sys
from .common.logging import setup_logger

def main():
    p = argparse.ArgumentParser("generate")
    p.add_argument("--stage", choices=["design", "execute"], required=True)
    p.add_argument("--tier", choices=["easy", "medium", "hard"], default="easy")
    p.add_argument("--study-path", required=True)
    p.add_argument("--templates-dir", default="./templates")
    p.add_argument("--show-prompt", action="store_true", default=False)
    args = p.parse_args()

    if args.stage == "design" and args.tier == "easy":
        from .design.easy import run_design_easy
        logger = setup_logger(os.path.join(args.study_path, "_logs", "design_easy.log"))
        run_design_easy(args.study_path, args.templates_dir, args.show_prompt, logger)
    elif args.stage == "execute" and args.tier == "easy":
        from .execute.easy import run_execute_easy
        run_execute_easy(args.study_path)
    else:
        sys.exit(f"Stage/tier not implemented yet: {args.stage}/{args.tier}")

if __name__ == "__main__":
    main()

