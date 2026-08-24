"""Compatibility entry point for the reorganized Sunnybrook evaluator."""

if __name__ == "__main__":
    import runpy

    runpy.run_module("unified_latents.engineering.eval.eval_sunnybrook_engineering", run_name="__main__")
else:
    from unified_latents.engineering.eval.eval_sunnybrook_engineering import *  # noqa: F401,F403
