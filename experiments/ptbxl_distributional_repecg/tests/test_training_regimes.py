from pathlib import Path


EXPERIMENT = Path(__file__).resolve().parents[1]


def test_trainers_do_not_advertise_unimplemented_training_regimes() -> None:
    for paper_id in range(1, 16):
        path = EXPERIMENT / f"scripts/paper{paper_id:02d}/train_paper{paper_id:02d}_shared_grid.py"
        source = path.read_text()
        assert 'choices=["full_only"]' in source, path
        for unsupported in ('"mask_aug"', '"finetune"', '"scratch"'):
            assert unsupported not in source, (path, unsupported)
