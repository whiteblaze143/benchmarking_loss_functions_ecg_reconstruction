"""Compatibility entry point for the reorganized Sunnybrook benchmark."""

if __name__ == "__main__":
    import runpy

    runpy.run_module("unified_latents.engineering.eval.benchmark_sunnybrook_all_features", run_name="__main__")
else:
    from unified_latents.engineering.eval.benchmark_sunnybrook_all_features import *  # noqa: F401,F403
