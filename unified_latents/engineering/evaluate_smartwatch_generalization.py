"""Compatibility entry point for the smartwatch generalization evaluator."""

if __name__ == "__main__":
    import runpy

    runpy.run_module("unified_latents.engineering.eval.evaluate_smartwatch_generalization", run_name="__main__")
else:
    from unified_latents.engineering.eval.evaluate_smartwatch_generalization import *  # noqa: F401,F403
