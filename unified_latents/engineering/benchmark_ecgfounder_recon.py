"""Compatibility entry point for the reorganized ECGFounder benchmark."""

if __name__ == "__main__":
    import runpy

    runpy.run_module("unified_latents.engineering.eval.benchmark_ecgfounder_recon", run_name="__main__")
else:
    from unified_latents.engineering.eval.benchmark_ecgfounder_recon import *  # noqa: F401,F403
