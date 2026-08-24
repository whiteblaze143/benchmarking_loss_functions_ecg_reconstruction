import sys
import os

project_root_dir = "/data1/1shared/jinjiarui/run/Diffusion-TS"
sys.path.append(project_root_dir)
os.chdir(project_root_dir)

from utils.data_utils.mimic_dataset import MIMICDataset


if __name__ == "__main__":

    dataset = MIMICDataset(
        "MIMIC",
        "/data1/1shared/jinjiarui/datasets/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0",
    )
