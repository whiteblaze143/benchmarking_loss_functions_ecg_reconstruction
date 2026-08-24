"""Compatibility entry point for the reorganized memory-optimized trainer."""

if __name__ == "__main__":
    import runpy

    runpy.run_module("unified_latents.engineering.trainers.train_vae_fm_memopt", run_name="__main__")
else:
    from unified_latents.engineering.trainers.train_vae_fm_memopt import *  # noqa: F401,F403
