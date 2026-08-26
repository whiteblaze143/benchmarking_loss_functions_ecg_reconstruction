with open("scripts/evaluate_1lead_rdb_semiseg_blinded.py", "r") as f:
    code = f.read()

code = code.replace("FROM checkpoints WHERE", "FROM models WHERE")
code = code.replace("status='remote_verified'", "status='completed'")

with open("scripts/evaluate_1lead_rdb_semiseg_blinded.py", "w") as f:
    f.write(code)
