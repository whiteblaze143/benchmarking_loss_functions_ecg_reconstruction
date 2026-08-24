"""
Stub for MasonWrapper, CNVAEReconstructor, BNVAEArgs.
These classes are referenced by scripts but not yet present in the consolidated repo.
Restore from backup or re-implement to use.
"""
def __getattr__(name):
    if name in ("MasonWrapper", "CNVAEReconstructor", "BNVAEArgs"):
        raise ImportError(
            f"{name} is not yet migrated to src.reconstruction.learn_functions.wrappers. "
            "Restore from backup or implement to use this script."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
