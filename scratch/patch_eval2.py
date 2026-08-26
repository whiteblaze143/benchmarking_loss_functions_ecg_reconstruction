with open("scripts/evaluate_1lead_rdb_semiseg_blinded.py", "r") as f:
    code = f.read()

import re

# We will just patch ensure_original_ceiling to check ORIGINAL_MATCHES instead of ORIGINAL_PREDICTIONS
code = code.replace("if ORIGINAL_PREDICTIONS:", "if ORIGINAL_MATCHES:")

with open("scripts/evaluate_1lead_rdb_semiseg_blinded.py", "w") as f:
    f.write(code)
