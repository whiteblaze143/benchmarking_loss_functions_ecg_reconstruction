import io,json,sys,zipfile
import h5py,numpy as np,sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from rr_features import features
rng=np.random.default_rng(42);x=rng.normal(size=(128,16));y=(x[:,0]>0).astype(int)
a=HistGradientBoostingClassifier(max_iter=3,early_stopping=False,random_state=42).fit(x,y)
b=HistGradientBoostingClassifier(max_iter=3,early_stopping=False,random_state=42).fit(x,y)
np.testing.assert_array_equal(a.predict_proba(x),b.predict_proba(x))
p='/data/mithunmanivannan/datasets/iridia_af_8405941_v1.0.1/iridia-af-records-v1.0.1.zip'
with zipfile.ZipFile(p) as z:
    name='iridia-af-records-v1.0.1/record_000/record_000_rr_00.h5'
    with h5py.File(io.BytesIO(z.read(name))) as h:rr=h['rr'][31:511]
f,valid=features(rr.reshape(8,60));assert valid.all() and np.isfinite(f).all()
print('WITNESS_AF_RR_CPU',json.dumps({'shape':list(f.shape),'python':sys.version.split()[0],'numpy':np.__version__,'sklearn':sklearn.__version__,'gpu_used':False}))
