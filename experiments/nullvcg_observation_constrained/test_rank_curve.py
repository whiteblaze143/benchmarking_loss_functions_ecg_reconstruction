import numpy as np

from run_residual_rank_curve import gross_limb_mask, select_rank


def test_gross_limb_qc_does_not_flag_quantization_but_flags_spike():
    target = np.zeros((2, 12, 8), dtype=np.float32)
    target[0, 3] = 0.0015
    target[1, 4, 3] = 0.02
    np.testing.assert_array_equal(gross_limb_mask(target), [False, True])


def test_rank_selection_uses_smallest_energy_and_next_gain_candidate():
    rows = {
        rank: {
            "training_residual_energy_fraction": [0.55, 0.78, 0.88, 0.92, 0.95, 0.97][rank - 1],
            "marginal_oracle_gain": [0.20, 0.10, 0.03, 0.01, 0.004, 0.002][rank - 1],
        }
        for rank in range(1, 7)
    }
    selected, reason = select_rank(rows)
    assert selected == 4
    assert "G_5<0.005" in reason


def test_rank_selection_falls_back_to_first_ninety_percent_energy():
    rows = {
        rank: {
            "training_residual_energy_fraction": [0.55, 0.78, 0.88, 0.91, 0.95, 0.97][rank - 1],
            "marginal_oracle_gain": [0.20, 0.10, 0.03, 0.02, 0.01, 0.006][rank - 1],
        }
        for rank in range(1, 7)
    }
    selected, reason = select_rank(rows)
    assert selected == 4
    assert "had not met" in reason
