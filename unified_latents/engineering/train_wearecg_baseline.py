"""Compatibility entry point for the reorganized WearECG baseline trainer."""

if __name__ == "__main__":
    import runpy

    runpy.run_module("unified_latents.engineering.trainers.train_wearecg_baseline", run_name="__main__")
else:
    from unified_latents.engineering.trainers.train_wearecg_baseline import *  # noqa: F401,F403
