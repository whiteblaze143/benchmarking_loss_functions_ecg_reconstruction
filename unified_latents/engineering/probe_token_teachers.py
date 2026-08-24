"""Compatibility entry point for the token-teacher probe."""

if __name__ == "__main__":
    import runpy

    runpy.run_module("unified_latents.engineering.experimental.probe_token_teachers", run_name="__main__")
else:
    from unified_latents.engineering.experimental.probe_token_teachers import *  # noqa: F401,F403
