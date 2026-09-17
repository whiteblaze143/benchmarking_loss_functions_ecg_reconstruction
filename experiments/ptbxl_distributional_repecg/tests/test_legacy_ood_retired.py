from pathlib import Path
import subprocess
import sys


EXPERIMENT = Path(__file__).resolve().parents[1]


def test_fabricated_label_builder_is_retired() -> None:
    script = EXPERIMENT / "scripts/paper02/build_paper02_ood_representations.py"
    completed = subprocess.run([sys.executable, str(script)], text=True, capture_output=True)
    assert completed.returncode != 0
    assert "fabricated PTB-XL labels are forbidden" in completed.stderr
    assert "np.zeros(5" not in script.read_text()


def test_all_legacy_ood_evaluators_fail_closed() -> None:
    scripts = sorted((EXPERIMENT / "scripts").glob("paper*/evaluate_paper*_ood.py"))
    assert len(scripts) == 15
    for script in scripts:
        source = script.read_text()
        assert "retired unsafe OOD evaluator" in source
        assert "PhaseCNN" not in source
        completed = subprocess.run([sys.executable, str(script)], text=True, capture_output=True)
        assert completed.returncode != 0
        assert "checkpoint-backed native-task evaluation" in completed.stderr
