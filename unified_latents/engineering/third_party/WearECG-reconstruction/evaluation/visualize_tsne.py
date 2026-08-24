import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from pathlib import Path
from datetime import datetime

# It's assumed that ECGFounder and metric_utils are in the Python path
# or in a location discoverable by Python's import system.
try:
    from ECGFounder import ft_ECGFounder
    from metric_utils import seed_everything, compute_representations_in_batches
except ImportError as e:
    print(f"Error importing ECGFounder or metric_utils: {e}")
    print("Please ensure these modules are installed and accessible in your PYTHONPATH.")
    # As a fallback, define stubs if main script logic needs them to exist
    # This allows the script to be parsed, but it will fail at runtime if they are used.
    def seed_everything(seed, use_cuda): pass
    def compute_representations_in_batches(model, data, batch_size, device):
        raise NotImplementedError("compute_representations_in_batches is not available due to import error")
    class ft_ECGFounder:
        def __init__(self, device):
            raise NotImplementedError("ft_ECGFounder is not available due to import error")
        def eval(self): pass


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize ECG data with t-SNE")
    parser.add_argument(
        "--data_folder",
        type=str,
        default="results/vae_500/PTBXL",
        help="Path to folder containing generated and real samples. Expects 'samples/overall_fake_data.npy' and 'samples/ground_truth_5000_test.npy' within this folder.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="results/tsne_visualizations",
        help="Path to folder for saving t-SNE plots",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for computation (cuda or cpu)",
    )
    parser.add_argument(
        "--cuda_device", type=int, default=0, help="CUDA device index to use if device is cuda"
    )
    parser.add_argument(
        "--perplexity", type=float, default=30.0, help="Perplexity for t-SNE"
    )
    parser.add_argument(
        "--n_iter", type=int, default=1000, help="Number of iterations for t-SNE"
    )
    parser.add_argument(
        "--learning_rate", type=str, default='auto', help="Learning rate for t-SNE (float or 'auto')"
    )
    parser.add_argument(
        "--notes", type=str, default="", help="Additional notes for appending to the plot filename"
    )
    parser.add_argument(
        "--use_raw_data",
        action="store_true",
        help="Use raw data for t-SNE instead of extracted features. Warning: can be slow and memory-intensive for high-dimensional raw data.",
    )
    parser.add_argument(
        "--real_data_filename",
        type=str,
        default="ground_truth_5000_test.npy",
        help="Filename of the real (ground truth) ECG data .npy file within the 'samples' subdirectory of data_folder."
    )
    parser.add_argument(
        "--fake_data_filename",
        type=str,
        default="overall_fake_data.npy",
        help="Filename of the fake (generated) ECG data .npy file within the 'samples' subdirectory of data_folder."
    )
    return parser.parse_args()


def extract_metadata_from_path(path_str):
    """Extract dataset and synthesis type from path string"""
    path_parts = Path(path_str).parts
    # Assuming path structure like .../synthesis_type/dataset
    dataset_name = path_parts[-1]
    synthesis_type = path_parts[-2] if len(path_parts) > 1 else "unknown"
    return {
        "dataset": dataset_name,
        "synthesis_type": synthesis_type,
    }


def main():
    args = parse_args()
    
    try:
        seed_everything(42, True if args.device == "cuda" else False)
    except NameError: # In case seed_everything was not imported
        print("Warning: seed_everything not found. Proceeding without setting a fixed seed.")
        
    if args.device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
        if not torch.cuda.is_available():
            print("Warning: CUDA selected, but not available. Falling back to CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Selected device: {device}")
    print(f"Starting t-SNE visualization for data in: {args.data_folder}")
    Path(args.output_folder).mkdir(parents=True, exist_ok=True)

    metadata = extract_metadata_from_path(args.data_folder)
    
    # Load data
    try:
        fake_samples_path = os.path.join(args.data_folder, "samples", args.fake_data_filename)
        real_samples_path = os.path.join(args.data_folder, "samples", args.real_data_filename)
        
        print(f"Loading fake samples from: {fake_samples_path}")
        fake_samples = np.load(fake_samples_path)
        print(f"Loading real samples from: {real_samples_path}")
        real_samples = np.load(real_samples_path)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}.")
        print("Please ensure --data_folder is correct and the .npy files ('overall_fake_data.npy', 'ground_truth_5000_test.npy' by default or as specified) exist in the 'samples' subdirectory.")
        return
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        return

    print(f"Real samples original shape: {real_samples.shape}, Fake samples original shape: {fake_samples.shape}")

    # Ensure samples are not excessively large to avoid memory issues, especially for raw data
    # This is a heuristic, might need adjustment based on typical data sizes and available memory
    MAX_SAMPLES_FOR_TSNE = 10000 
    if real_samples.shape[0] > MAX_SAMPLES_FOR_TSNE:
        print(f"Warning: Real samples ({real_samples.shape[0]}) exceed MAX_SAMPLES_FOR_TSNE ({MAX_SAMPLES_FOR_TSNE}). Subsampling to prevent memory issues.")
        indices = np.random.choice(real_samples.shape[0], MAX_SAMPLES_FOR_TSNE, replace=False)
        real_samples = real_samples[indices]
    if fake_samples.shape[0] > MAX_SAMPLES_FOR_TSNE:
        print(f"Warning: Fake samples ({fake_samples.shape[0]}) exceed MAX_SAMPLES_FOR_TSNE ({MAX_SAMPLES_FOR_TSNE}). Subsampling to prevent memory issues.")
        indices = np.random.choice(fake_samples.shape[0], MAX_SAMPLES_FOR_TSNE, replace=False)
        fake_samples = fake_samples[indices]
    
    print(f"Using {real_samples.shape[0]} real samples and {fake_samples.shape[0]} fake samples for t-SNE.")


    all_data_for_tsne = None
    labels = None
    data_source_name = ""

    if args.use_raw_data:
        print("Using raw data for t-SNE.")
        # Reshape data if it's multi-lead/multi-channel: (n_samples, n_timesteps, n_leads) -> (n_samples, n_timesteps * n_leads)
        # Or if it's (n_samples, n_leads, n_timesteps) -> (n_samples, n_leads * n_timesteps)
        # Assuming the last two dimensions need to be flattened.
        try:
            real_data_flat = real_samples.reshape(real_samples.shape[0], -1)
            fake_data_flat = fake_samples.reshape(fake_samples.shape[0], -1)
        except Exception as e:
            print(f"Error reshaping raw data: {e}. Ensure data has at least 2 dimensions.")
            return
        
        all_data_for_tsne = np.concatenate([real_data_flat, fake_data_flat], axis=0)
        labels = np.concatenate([np.zeros(real_data_flat.shape[0], dtype=int), 
                                 np.ones(fake_data_flat.shape[0], dtype=int)])
        data_source_name = "raw_data"
    else:
        print("Extracting features using ft_ECGFounder for t-SNE.")
        try:
            model = ft_ECGFounder(device=device)
            model.eval()

            print("Computing representations for real samples...")
            real_ecg_representation = compute_representations_in_batches(
                model, real_samples, batch_size=128, device=device
            )
            print(f"Real representations shape: {real_ecg_representation.shape}")
            
            print("Computing representations for fake samples...")
            fake_ecg_representation = compute_representations_in_batches(
                model, fake_samples, batch_size=128, device=device
            )
            print(f"Fake representations shape: {fake_ecg_representation.shape}")
            
            all_data_for_tsne = np.concatenate([real_ecg_representation, fake_ecg_representation], axis=0)
            labels = np.concatenate([np.zeros(real_ecg_representation.shape[0], dtype=int), 
                                     np.ones(fake_ecg_representation.shape[0], dtype=int)])
            data_source_name = "features"
        except NameError:
            print("Error: ft_ECGFounder or compute_representations_in_batches not available due to import error. Cannot extract features.")
            print("Try running with --use_raw_data or ensure ECGFounder and metric_utils are correctly installed.")
            return
        except Exception as e:
            print(f"An error occurred during feature extraction: {e}")
            return

    if all_data_for_tsne is None:
        print("No data prepared for t-SNE. Exiting.")
        return

    print(f"Total data points for t-SNE: {all_data_for_tsne.shape[0]}, Dimensions: {all_data_for_tsne.shape[1]}")

    # Perform t-SNE
    print(f"Running t-SNE (perplexity={args.perplexity}, n_iter={args.n_iter}, learning_rate='{args.learning_rate}')... This may take a while.")
    
    current_lr_tsne = args.learning_rate
    if args.learning_rate.lower() != 'auto':
        try:
            current_lr_tsne = float(args.learning_rate)
        except ValueError:
            print(f"Warning: Invalid learning_rate '{args.learning_rate}'. Using scikit-learn's default or 'auto' if applicable by version.")
            # For older scikit-learn, 'auto' might not be valid, default is 200.0
            # For newer, 'auto' becomes (n_samples - 1) / (4 * n_components) / 3 if perplexity is high, else 200.0
            # We'll pass the string 'auto' and let sklearn handle it or error if unsupported version.
            current_lr_tsne = 'auto' 

    tsne_model = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        n_iter=args.n_iter,
        learning_rate=current_lr_tsne,
        random_state=42,
        verbose=1,
        init='pca', # PCA initialization can be faster and more stable
        n_jobs=-1 # Use all available cores
    )
    tsne_results = tsne_model.fit_transform(all_data_for_tsne)
    print("t-SNE completed.")

    # Plotting
    plt.figure(figsize=(14, 12))
    scatter_plot = sns.scatterplot(
        x=tsne_results[:, 0],
        y=tsne_results[:, 1],
        hue=labels,
        palette=sns.color_palette(['#1f77b4', '#ff7f0e'], 2), # Blue for real, Orange for fake
        legend="full",
        alpha=0.6
    )
    
    handles, _ = scatter_plot.get_legend_handles_labels()
    scatter_plot.legend(handles, ['Real Samples (0)', 'Fake Samples (1)'], title='Sample Type', loc='best')
    
    plot_title = (f't-SNE: {metadata["dataset"]} ({metadata["synthesis_type"]})'
                  f'Data: {data_source_name}, Perplexity: {args.perplexity}, Iterations: {args.n_iter}')
    plt.title(plot_title)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    notes_suffix = f"_{args.notes.replace(' ', '_')}" if args.notes else ""
    
    plot_filename = f'tsne_{metadata["dataset"]}_{metadata["synthesis_type"]}_{data_source_name}{notes_suffix}_{timestamp}.png'
    plot_path = os.path.join(args.output_folder, plot_filename)
    
    try:
        plt.savefig(plot_path, bbox_inches='tight')
        print(f"t-SNE plot saved to: {plot_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")

    # plt.show() # Uncomment if you want to display the plot interactively

if __name__ == "__main__":
    main() 