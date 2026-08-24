import sqlite3
import pandas as pd
import numpy as np

db_path = "results/clinical_biomarkers_multids/clinical_metrics.db"
conn = sqlite3.connect(db_path, timeout=60)
df = pd.read_sql_query("""
    SELECT model_id, evaluation_version,
           v3_r_presacan_r2, v3_r_presacan_slope, v3_r_var_ret_pct, v3_r_direct_r2, v3_r_direct_slope,
           v6_r_presacan_r2, v6_r_presacan_slope, v6_r_var_ret_pct, v6_r_direct_r2, v6_r_direct_slope,
           v3_t_presacan_r2, v3_t_presacan_slope, v3_t_var_ret_pct,
           interlead_r2_real_I_V3, interlead_r2_recon_I_V3,
           interlead_r2_real_I_V6, interlead_r2_recon_I_V6,
           spurious_coupling_ratio_v3, avg_precordial_var_ret_pct
    FROM presacan_model_summary
""", conn)
conn.close()

print(f"=== TOTAL MODELS IN PRESACAN BENCHMARK: {len(df)} ===\n")

if len(df) > 0:
    def get_mask(mid):
        parts = mid.split('_')
        for p in parts:
            if len(p) == 7 and p.isdigit():
                return p
        return mid

    df['mask'] = df['model_id'].apply(get_mask)
    df['arch'] = df['model_id'].apply(lambda x: 'msvae' if 'msvae' in x else ('ecg_aim' if 'aim' in x else 'unet'))

    # Rank 1: Top Models by Lead V3 R-peak Variance Retention (%)
    print("=== TOP 10 MODELS BY V3 VARIANCE RETENTION (%) ===")
    top_var = df.sort_values(by='v3_r_var_ret_pct', ascending=False).head(10)
    print(top_var[['mask', 'arch', 'model_id', 'v3_r_var_ret_pct', 'v3_r_direct_r2', 'v3_r_presacan_r2', 'v3_r_presacan_slope', 'interlead_r2_recon_I_V3']].to_string(index=False))

    print("\n=== TOP 10 MODELS BY AVERAGE PRECORDIAL (V1-V6) VARIANCE RETENTION (%) ===")
    top_avg_var = df.sort_values(by='avg_precordial_var_ret_pct', ascending=False).head(10)
    print(top_avg_var[['mask', 'arch', 'model_id', 'avg_precordial_var_ret_pct', 'v3_r_var_ret_pct', 'v6_r_var_ret_pct', 'v3_r_presacan_slope', 'spurious_coupling_ratio_v3']].to_string(index=False))

    print("\n=== BOTTOM 10 MODELS (MOST SEVERE VARIANCE COLLAPSE) ===")
    bot_var = df.sort_values(by='v3_r_var_ret_pct', ascending=True).head(10)
    print(bot_var[['mask', 'arch', 'model_id', 'v3_r_var_ret_pct', 'v3_r_presacan_r2', 'v3_r_presacan_slope', 'interlead_r2_recon_I_V3']].to_string(index=False))

    print("\n=== SPURIOUS COUPLING: TOP 10 MODELS WITH LOWEST RECONSTRUCTED INTERLEAD R^2(I, V3) ===")
    # Ideal is real ground truth ~0.0207, whereas GAN/U-Net jumps to ~0.49
    low_spurious = df.sort_values(by='interlead_r2_recon_I_V3', ascending=True).head(10)
    print(low_spurious[['mask', 'arch', 'model_id', 'interlead_r2_recon_I_V3', 'spurious_coupling_ratio_v3', 'v3_r_var_ret_pct']].to_string(index=False))

    # Factorial Analysis on 7-digit masks
    mask_df = df[df['mask'].str.len() == 7].copy()
    if len(mask_df) > 0:
        mask_df['corr_loss'] = mask_df['mask'].apply(lambda m: int(m[1]))
        mask_df['deriv_loss'] = mask_df['mask'].apply(lambda m: int(m[2]))
        mask_df['vcg_loss'] = mask_df['mask'].apply(lambda m: int(m[3]))
        mask_df['energy_loss'] = mask_df['mask'].apply(lambda m: int(m[4]))
        mask_df['lead_cons_loss'] = mask_df['mask'].apply(lambda m: int(m[5]))
        mask_df['mmd_kernel'] = mask_df['mask'].apply(lambda m: int(m[6]))

        print("\n=== FACTORIAL MAIN EFFECTS ON V3 VARIANCE RETENTION (%) ===")
        for col in ['corr_loss', 'deriv_loss', 'vcg_loss', 'energy_loss', 'lead_cons_loss', 'mmd_kernel']:
            grp = mask_df.groupby(col)[['v3_r_var_ret_pct', 'v3_r_presacan_slope', 'v3_r_direct_r2', 'interlead_r2_recon_I_V3']].agg(['mean', 'count'])
            print(f"\n--- Factor: {col} ---")
            print(grp.to_string())
