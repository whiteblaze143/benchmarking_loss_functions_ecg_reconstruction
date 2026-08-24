import pandas as pd
import numpy as np
import pingouin as pg

sub = pd.DataFrame({
    'true': [100, 200, 300, 400],
    'pred': [101, 199, 302, 395]
})

icc_df = pd.DataFrame({
    'Target': np.repeat(np.arange(len(sub)), 2),
    'Rater': np.tile(['Physician', 'NK2'], len(sub)),
    'Score': np.column_stack((sub['true'].values, sub['pred'].values)).flatten()
})
icc = pg.intraclass_corr(data=icc_df, targets='Target', raters='Rater', ratings='Score')
print(icc)
