with open("scripts/evaluate_1lead_rdb_semiseg_blinded.py", "r") as f:
    code = f.read()

code = code.replace("FROM models WHERE", "FROM checkpoints WHERE")
code = code.replace("status='completed'", "status='remote_verified'")

with open("scripts/evaluate_1lead_rdb_semiseg_blinded.py", "w") as f:
    f.write(code)
