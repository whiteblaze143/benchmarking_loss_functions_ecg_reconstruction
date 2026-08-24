"""Compatibility entry point for the reorganized stage-1 trainer."""

if __name__ == "__main__":
    import runpy

    runpy.run_module("unified_latents.engineering.trainers.train_stage1_ae", run_name="__main__")
else:
    from unified_latents.engineering.trainers.train_stage1_ae import *  # noqa: F401,F403
