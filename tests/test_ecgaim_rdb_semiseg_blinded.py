from pathlib import Path
import os
import subprocess
import sys

import numpy as np

from scripts.evaluate_ecgaim_rdb_semiseg_blinded import (
    BOUNDARIES, LEAD_GROUPS, connect, initialize, reference_boundaries,
)


def test_rdb_regions_map_to_six_inclusive_boundaries():
    intervals=np.asarray([[0,10,19],[1,30,44],[2,50,79]],dtype=np.int64)
    result=reference_boundaries(intervals)
    assert set(result)==set(BOUNDARIES)
    assert result["P_onset"].tolist()==[10]
    assert result["P_offset"].tolist()==[19]
    assert result["QRS_onset"].tolist()==[30]
    assert result["QRS_offset"].tolist()==[44]
    assert result["T_onset"].tolist()==[50]
    assert result["T_offset"].tolist()==[79]


def test_lead_groups_preserve_primary_and_derived_controls():
    assert LEAD_GROUPS["primary_missing_precordial"]==(6,8,9,10,11)
    assert LEAD_GROUPS["derived_limb_control"]==(2,3,4,5)
    assert set(LEAD_GROUPS["all_missing"])==set(range(12))-{0,1,7}


def test_compact_database_contains_only_aggregate_tables(tmp_path: Path):
    path=tmp_path/"compact.sqlite"; connection=connect(path)
    initialize(connection,{"test":True},"d"*64)
    tables={row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert tables=={"metadata","evaluations","boundary_summaries"}
    assert connection.execute("PRAGMA integrity_check").fetchone()[0]=="ok"
    connection.close()


def test_protocol_mismatch_refuses_database_reuse(tmp_path: Path):
    connection=connect(tmp_path/"compact.sqlite")
    initialize(connection,{"version":1},"a"*64)
    try:
        initialize(connection,{"version":2},"a"*64)
    except RuntimeError as error:
        assert "different protocol" in str(error)
    else:
        raise AssertionError("protocol drift was accepted")
    connection.close()


def test_production_entrypoint_help_runs_without_pythonpath():
    environment=os.environ.copy(); environment.pop("PYTHONPATH",None)
    completed=subprocess.run(
        [sys.executable,"scripts/evaluate_ecgaim_rdb_semiseg_blinded.py","--help"],
        cwd=Path(__file__).resolve().parents[1],env=environment,text=True,capture_output=True,
    )
    assert completed.returncode==0,completed.stderr
    assert "Compact external RDB blinded evaluation" in completed.stdout
