"""Compatibility entry point for Sunnybrook benchmark comparison."""

if __name__ == "__main__":
    import runpy

    runpy.run_module("unified_latents.engineering.eval.compare_sunnybrook_benchmarks", run_name="__main__")
else:
    from unified_latents.engineering.eval.compare_sunnybrook_benchmarks import *  # noqa: F401,F403
