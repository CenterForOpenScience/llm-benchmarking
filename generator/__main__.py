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
        # Use your LLM agent in design_react/agent.py
        # It already configures its own file logging via _configure_file_logging().
        from generator.design_agent import run_design
        # Optional: still create a top-level log to mirror execute behavior:
        _ = setup_logger(os.path.join(args.study_path, "_logs", "design_easy.log"))
        # run the agent that generates the plan / prereg JSON
        run_design(args.study_path, show_prompt=args.show_prompt)

    elif args.stage == "execute" and args.tier == "easy":
        # Agent-driven, step-by-step with human confirmation before executing
        from generator.execute_agent import run_execute
        _ = setup_logger(os.path.join(args.study_path, "_logs", "execute_easy.log"))
        run_execute(
            study_path=args.study_path,
            show_prompt=args.show_prompt,
            templates_dir=args.templates_dir,
        )

    else:
        sys.exit(f"Stage/tier not implemented yet: {args.stage}/{args.tier}")

if __name__ == "__main__":
    main()
