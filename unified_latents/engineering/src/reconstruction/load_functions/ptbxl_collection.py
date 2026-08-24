import os
import pandas as pd
import wfdb
import numpy as np
import torch
from scipy.signal import resample
from util_functions.general import get_twelve_keys

class PTBXLCollectionProxy:
    """
    Mock MongoDB Collection for PT-BXL data.
    Provides GE MUSE-style documents to satisfy Mason et al. (2024) scripts.
    """
    def __init__(self, root_dir, csv_file, target_fs=100):
        self.root_dir = root_dir
        self.df = pd.read_csv(csv_file)
        self.target_fs = target_fs
        # Map ecg_id as string for easier querying
        self.df['ecg_id_str'] = self.df['ecg_id'].astype(str)
        # Set ecg_id as index for fast find_one
        self.indexed_df = self.df.set_index('ecg_id_str')

    def find_one(self, query):
        """
        Mocks MongoDB find_one. Supports query by _id or ElementID.
        """
        # Mason scripts use both '_id' and 'ElementID'
        element_id = query.get('_id') or query.get('ElementID')
        if element_id is None:
             # Handle empty query (rare in Mason scripts but possible)
             return None
        
        element_id = str(element_id)
        if element_id not in self.indexed_df.index:
            return None
            
        row = self.indexed_df.loc[element_id]
        
        # Load WFDB (500Hz)
        # filename_hr example: records500/00000/00001_hr
        rel_path = row['filename_hr']
        abs_path = os.path.join(self.root_dir, rel_path)
        
        data, header = wfdb.rdsamp(abs_path) # [samples, 12]
        fs = header['fs']
        
        # Resample if needed
        if self.target_fs and fs != self.target_fs:
            num_samples = int(len(data) * self.target_fs / fs)
            data = resample(data, num_samples, axis=0)
            fs = self.target_fs

        # PTB-XL Lead Order: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
        # Mason Lead Order:  I, II, III, aVL, aVR, aVF, V1, V2, V3, V4, V5, V6
        # We need to swap index 3 (aVR) and 4 (aVL)
        mason_data = data.copy()
        mason_data[:, 3] = data[:, 4] # aVL
        mason_data[:, 4] = data[:, 3] # aVR
        
        twelve_keys = get_twelve_keys()
        
        # Mason's extract_twelve_leads expects element['lead'][key].to_numpy() / 1000
        # This implies lead values are in microvolts (uV) since it divides by 1000 to get mV
        # PTB-XL data from wfdb.rdsamp is usually in mV.
        # So we multiply by 1000.
        lead_dict = {}
        for i, key in enumerate(twelve_keys):
            # We use a Series to provide .to_numpy()
            lead_dict[key] = pd.Series(mason_data[:, i] * 1000)

        # Split report into GE MUSE style lines
        report = str(row['report']) if pd.notna(row['report']) else ""
        diag_lines = [{"StmtText": line.strip(), "StmtFlag": ["ENDSLINE"]} for line in report.split('\n') if line.strip()]

        # Construct the "MUSE" document
        doc = {
            '_id': element_id,
            'ElementID': element_id,
            'MeasureID': element_id,
            'PatientID': str(row['patient_id']),
            'RestingECG': {
                'PatientDemographics': {
                    'PatientID': str(row['patient_id']),
                    'PatientAge': int(row['age']) if pd.notna(row['age']) else 0,
                    'Gender': 'MALE' if row['sex'] == 0 else 'FEMALE', # 0=M, 1=F in PTB-XL
                    'Race': 'UNKNOWN'
                },
                'TestDemographics': {
                    'AcquisitionDate': str(row['recording_date'])[:10].replace('-', ''), # YYYYMMDD
                },
                'Diagnosis': {
                    'DiagnosisStatement': diag_lines
                },
                'OriginalDiagnosis': report
            },
            'lead': lead_dict,
            # Add top-level keys for classify_dataset.py
            'Diagnosis': diag_lines,
            'Demographic': {
                'PatientAge': int(row['age']) if pd.notna(row['age']) else 0,
                'Gender': 'MALE' if row['sex'] == 0 else 'FEMALE',
                'Race': 'UNKNOWN'
            }
        }
        
        return doc

    def find(self, query=None, projection=None, limit=0):
        """
        Mocks MongoDB find. Supports simple queries like {'ElementID': {'$in': [...]}}
        """
        target_ids = None
        if query and 'ElementID' in query and '$in' in query['ElementID']:
            target_ids = [str(x) for x in query['ElementID']['$in']]
        elif query and '_id' in query and '$in' in query['_id']:
            target_ids = [str(x) for x in query['_id']['$in']]
            
        if target_ids:
            # Filter df
            subset = self.df[self.df['ecg_id_str'].isin(target_ids)]
        else:
            subset = self.df
            
        if limit > 0:
            subset = subset.head(limit)
            
        for _, row in subset.iterrows():
            yield self.find_one({'_id': row['ecg_id_str']})

    def count_documents(self, query):
        return len(self.df) # Simplified
