import json
from pathlib import Path
import sys

_ROOT = Path(".").resolve()
sys.path.insert(0, str(_ROOT))
from scripts.experiment_provenance import code_provenance

source_paths = [
    "scripts/train_factorial.py",
    "scripts/train_mcma_3lead.py",
    "scripts/common_loss.py",
    "scripts/experiment_provenance.py",
]
source_provenance = code_provenance(_ROOT, source_paths)

contract_path = _ROOT / "refine-logs/factorial_training_contract.json"
contract = json.loads(contract_path.read_text())

contract["approved_source_bundle_sha256"] = source_provenance["source_bundle_sha256"]
contract["source_file_sha256"] = source_provenance["source_file_sha256"]

contract_path.write_text(json.dumps(contract, indent=2))
print("Contract updated!")
