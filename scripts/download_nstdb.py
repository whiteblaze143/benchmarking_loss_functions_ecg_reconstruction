
import wfdb
import os
import argparse

def download_nstdb(data_dir="data/mit-bih-noise-stress-test-database-1.0.0"):
    print(f"Downloading NSTDB to {data_dir}...")
    os.makedirs(data_dir, exist_ok=True)
    
    # PhysioNet DB name is 'nstdb'
    # We only need the noise records: 'bw', 'em', 'ma'
    # But wfdb.dl_database downloads everything usually.
    # Let's try downloading specific files if dl_database is too heavy (it contains full holters).
    # actually nstdb is small enough.
    
    try:
        wfdb.dl_database('nstdb', data_dir)
        print("Download complete.")
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False

if __name__ == "__main__":
    download_nstdb()
