
import numpy as np
import neurokit2 as nk

def extract_features(sig, fs=500):
    """
    Extract features from a single ECG lead (numpy array).
    """
    # If explicit lead II selection is needed, handle outside or assume sig is 1D.
    # The original script took [12, L] tensor.
    # We should make this robust to input shape.
    
    try:
        if isinstance(sig, np.ndarray):
            if sig.ndim == 2:
                # Assume [12, L], take Lead II (idx 1)
                lead_ii = sig[1, :]
            else:
                lead_ii = sig
        else:
            # Tensor
            if sig.dim() == 2:
                lead_ii = sig[1, :].cpu().numpy()
            else:
                lead_ii = sig.cpu().numpy()

        # Clean
        clean = nk.ecg_clean(lead_ii, sampling_rate=fs, method='neurokit')
        
        # Peaks
        # Find R-peaks
        try:
            r_peaks = nk.ecg_findpeaks(clean, sampling_rate=fs)['ECG_R_Peaks']
        except Exception:
            return None
            
        if len(r_peaks) < 2:
            return None
            
        # Delineate
        try:
            waves = nk.ecg_delineate(clean, r_peaks, sampling_rate=fs, method='dwt')[0]
        except Exception:
            pass # Delineation optional for some feats
        
        # Calculate HRV & Intervals via ecg_analyze
        df_analyze = nk.ecg_analyze(nk.ecg_process(clean, sampling_rate=fs)[0], sampling_rate=fs)
        
        # Flatten and Clean
        rec_feat = df_analyze.iloc[0].to_dict()
        clean_feat = {}
        
        for k, v in rec_feat.items():
            try:
                # Unwrap array/list
                if isinstance(v, (np.ndarray, list)):
                    v = np.array(v).flatten()
                    if len(v) == 1:
                        v = v[0]
                    else:
                        v = np.mean(v)
                        
                # Check numeric
                if isinstance(v, (int, float, np.number)):
                    if np.isinf(v):
                         clean_feat[k] = np.nan # Imputer will handle
                    elif not np.isnan(v):
                        clean_feat[k] = float(v)
            except Exception:
                pass
                
        return clean_feat

    except Exception as e:
        print(f"Error extracting features: {e}")
        return {}
