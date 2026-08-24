import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_book_seed_and_factor_claims_match_executable_manifest():
    manifest = json.loads((ROOT / "refine-logs/factorial_manifest.json").read_text())
    jobs = [job for phase in manifest["phases"] for job in phase["jobs"]]
    identities = []
    for job in jobs:
        match = re.fullmatch(r"f_(\d{7})_s(\d+)", job["id"])
        assert match is not None
        identities.append((match.group(1), int(match.group(2))))

    assert len(identities) == 480
    assert sorted({seed for _, seed in identities}) == [42, 200, 201]
    assert len({mask for mask, _ in identities}) == 160

    exhaustive = (ROOT / "book/12_exhaustive_model_analysis.qmd").read_text()
    proposal = (ROOT / "refine-logs/FINAL_PROPOSAL.md").read_text()
    assert "42, 123, and 456" not in exhaustive
    assert "42, 200, and 201" in exhaustive
    assert "Seven main effects" not in proposal
    assert "seven-bit" not in proposal
    assert "five binary main effects" in proposal
