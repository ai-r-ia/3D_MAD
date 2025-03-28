from genericpath import isdir
import os
import numpy as np
import matplotlib.pyplot as plt
from configs.config import get_logger
from scipy.stats import norm
from tqdm import tqdm
import matplotlib.ticker as mticker
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import pickle
import logging
from torch.nn.functional import cosine_similarity
from tqdm import tqdm
import random
from itertools import combinations

from configs.seed import set_seed

# Logging setup
log_file = "logs/score_apcer.log"
logging.basicConfig(
    filename=log_file,
    level=logging.DEBUG,  # Changed to DEBUG for more detailed logs
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Enable CuDNN for optimal performance
torch.backends.cudnn.benchmark = True

printer = "digital"
morph_types = [
     "lma",  "mipgan1",  "mordiff", "greedy",  "mipgan2", "pipe"
    #   'CASIA_MAD_DB',  'Caucasian_MAD_DB',  
    #   'Indian_MAD_DB', 
    #   'African_MAD_DB', 
    #  "lma",  "mipgan1",  "mordiff", "greedy", "lmaubo", "mipgan2", "pipe"
            #    "lmaubo"
               ]
root_dirs = [
    "/mnt/extravolume/data/frill/digital", 
    # "/mnt/extravolume/data/ethnicity/digital", 
            #  "/mnt/extravolume/data/feret/digital"
            # "/mnt/extravolume/data/synonot/digital"
             ]


def save_apcer_bpcer_values(apcer_dict, bpcer_dict, save_dir, db_name, pct = 0.1, avg = False):
    """
    Save APCER and BPCER values to separate files in the specified directory.

    Parameters:
    - apcer_dict: Dictionary with model names and their corresponding APCER values
    - bpcer_dict: Dictionary with model names and their corresponding BPCER values
    - save_dir: Directory where the files should be saved
    """
    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Save APCER and BPCER values
    apcer_file = os.path.join(save_dir, f"apcer_values_{db_name}.pkl")
    bpcer_file = os.path.join(save_dir, f"bpcer_values_{db_name}.pkl")
    # bpcer_file = os.path.join(save_dir, "bpcer_at_apcer_values.pkl")
    if avg:
        apcer_file = os.path.join(save_dir, f"avg_apcer_values_{db_name}_{pct}.pkl")
        bpcer_file = os.path.join(save_dir, f"avg_bpcer_values__{db_name}_{pct}.pkl")
        

    # Save APCER values
    with open(apcer_file, "wb") as f:
        pickle.dump(apcer_dict, f)
    
    # Save BPCER values
    with open(bpcer_file, "wb") as f:
        pickle.dump(bpcer_dict, f)

    print(f"APCER and BPCER values saved to {save_dir}")
 
def get_far_frr_thresholds(
    genuine: np.ndarray, impostor: np.ndarray, bins: int = 10_001
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mi = min(np.min(impostor), np.min(genuine))
    mx = max(np.max(impostor), np.max(genuine))
 
    # Following 3 lines are optional
    # Normalize the scores
    impostor = (impostor - mi) / (mx - mi)
    genuine = (genuine - mi) / (mx - mi)
    thresholds = np.linspace(0, 1, bins)
 
    far = np.zeros(bins)
    frr = np.zeros(bins)
    for i, threshold in tqdm(enumerate(thresholds)):
        far[i] = np.sum(impostor >= threshold) / len(impostor)
        frr[i] = np.sum(genuine < threshold) / len(genuine)
 
    return far, frr, thresholds
 
 
def get_apcer_at_given_bpcer(
    far: np.ndarray, frr: np.ndarray, thresholds: np.ndarray, bpcer: float
) -> tuple[float, float]:
    idx = np.argmin(np.abs(frr - bpcer))  
    apcer = far[idx]  
    threshold = thresholds[idx]
    return apcer, threshold

def get_bpcer_at_given_apcer(
    far: np.ndarray, frr: np.ndarray, thresholds: np.ndarray, apcer: float
) -> tuple[float, float]:
    idx = np.argmin(np.abs(far - apcer)) 
    bpcer = frr[idx]  
    threshold = thresholds[idx]
    return bpcer, threshold



def compute_apcer_bpcer_for_all_models(models, morph_types, root_dirs, printer, save_dir):
    logging.debug("Starting APCER and BPCER computation for all models...")

    # Initialize dictionaries to store APCER and BPCER values for all models
    apcer_dict = {}
    bpcer_dict = {}
    db_name = "frill"
    for root_dir in root_dirs:
        dataset_name = os.path.basename(os.path.dirname(root_dir)) 
        db_name = dataset_name
        for morph_type in morph_types:
            for model_name, model in models.items():
                logging.debug(f"Computing APCER and BPCER for model: {model_name}, morph: {morph_type}, root: {root_dir}")
    
                # Load the saved genuine and imposter scores
                save_dir_for_scores = f"scores/{model_name}/{dataset_name}/{morph_type}"
                genuine_scores_path = f"{save_dir_for_scores}/genuine.npy"
                imposter_scores_path = f"{save_dir_for_scores}/imposter.npy"
                
                if os.path.exists(genuine_scores_path) and os.path.exists(imposter_scores_path):
                    genuine_scores = np.load(genuine_scores_path)
                    imposter_scores = np.load(imposter_scores_path)
                    
                    # Load FAR, FRR, and thresholds using get_far_frr_thresholds
                    far, frr, thresholds = get_far_frr_thresholds(genuine_scores, imposter_scores)

                    # You can now calculate APCER and BPCER using the threshold value
                    # Example: Assuming you want to compute APCER at 5% BPCER and BPCER at 5% APCER
                    apcer_at_5_bpcer, threshold_for_apcer = get_apcer_at_given_bpcer(far, frr, thresholds, 0.05)
                    print(f"APCER at 5% BPCER: {apcer_at_5_bpcer:.4} or {apcer_at_5_bpcer * 100:.4}%, Threshold: {threshold_for_apcer:.4}")
                    logging.info(f"APCER at 5% BPCER: {apcer_at_5_bpcer:.4} or {apcer_at_5_bpcer * 100:.4}%, Threshold: {threshold_for_apcer:.4}")

                    bpcer_at_5_apcer, threshold_for_bpcer = get_bpcer_at_given_apcer(far, frr, thresholds, 0.05)
                    print(f"BPCER at 5% APCER: {bpcer_at_5_apcer:.4} or {bpcer_at_5_apcer * 100:.4}%, Threshold: {threshold_for_bpcer:.4}")
                    logging.info(f"BPCER at 5% APCER: {bpcer_at_5_apcer:.4} or {bpcer_at_5_apcer * 100:.4}%, Threshold: {threshold_for_bpcer:.4}")

                    
                    # apcer_at_5_bpcer, threshold_for_apcer = get_apcer_at_given_bpcer(far, frr, thresholds, 0.1)
                    # print(f"APCER at 10% BPCER: {apcer_at_5_bpcer:.4} or {apcer_at_5_bpcer * 100:.4}%, Threshold: {threshold_for_apcer:.4}")
                    # logging.info(f"APCER at 10% BPCER: {apcer_at_5_bpcer:.4} or {apcer_at_5_bpcer * 100:.4}%, Threshold: {threshold_for_apcer:.4}")

                    # bpcer_at_5_apcer, threshold_for_bpcer = get_bpcer_at_given_apcer(far, frr, thresholds, 0.1)
                    # print(f"BPCER at 10% APCER: {bpcer_at_5_apcer:.4} or {bpcer_at_5_apcer * 100:.4}%, Threshold: {threshold_for_bpcer:.4}")
                    # logging.info(f"BPCER at 10% APCER: {bpcer_at_5_apcer:.4} or {bpcer_at_5_apcer * 100:.4}%, Threshold: {threshold_for_bpcer:.4}")

                    # Store APCER and BPCER values in their respective dictionaries
                    apcer_dict[f"{model_name}_{dataset_name}_{morph_type}"] = apcer_at_5_bpcer
                    bpcer_dict[f"{model_name}_{dataset_name}_{morph_type}"] = bpcer_at_5_apcer
                    
                else:
                    logging.error(f"Skipping APCER/BPCER calculation for {model_name}, {morph_type} due to missing scores.")

    # Save all APCER and BPCER values to files
    save_apcer_bpcer_values(apcer_dict, bpcer_dict, save_dir, db_name)
    
    
def compute_avg_apcer_bpcer_for_all_models(models, morph_types, root_dirs, printer, save_dir, pct):
    logging.debug("Starting APCER and BPCER computation for all models...")

    # Initialize dictionaries to store APCER and BPCER values for all models
    apcer_dict = {}
    bpcer_dict = {}
    db_name = "frill"
    for root_dir in root_dirs:
        dataset_name = os.path.basename(os.path.dirname(root_dir)) 
        db_name = dataset_name 
        
        for model_name, model in models.items():
            logging.debug(f"Computing APCER and BPCER for model: {model_name},  root: {root_dir}")

            # Load the saved genuine and imposter scores
            save_dir_for_scores = f"scores/{model_name}/{dataset_name}"
            genuine_scores_path = f"{save_dir_for_scores}/genuine.npy"
            imposter_scores_path = f"{save_dir_for_scores}/imposter.npy"
            
            if os.path.exists(genuine_scores_path) and os.path.exists(imposter_scores_path):
                genuine_scores = np.load(genuine_scores_path)
                imposter_scores = np.load(imposter_scores_path)
                
                # Load FAR, FRR, and thresholds using get_far_frr_thresholds
                far, frr, thresholds = get_far_frr_thresholds(genuine_scores, imposter_scores)

                far_m = far*100
                frr_m = frr*100
                plt.plot(far_m, frr_m,  label="Student", color='blue')

                plt.xlabel('APCER (%)')
                plt.ylabel('BPCER (%)')
                plt.title('DET Curve')
                # plt.xscale('log')
                # plt.yscale('log')
                # plt.xlim(0.1, 110)
                # plt.ylim(0.1, 110)
                plt.gca().xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}" if x >= 1 else f"{x:.1f}"))
                plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{int(y)}" if y >= 1 else f"{y:.1f}"))
                plt.tick_params(axis="both", which="both", labelsize=10)
                plt.minorticks_on()
                plt.grid(which='both', linestyle="--", linewidth = 0.5, alpha=0.5)
                plt.legend()

                plt.savefig(f"det_curve_{model_name}.png", dpi=300, bbox_inches='tight')  # High-res save
                plt.close()

                # You can now calculate APCER and BPCER using the threshold value
                # Example: Assuming you want to compute APCER at 5% BPCER and BPCER at 5% APCER
                # apcer_at_5_bpcer, threshold_for_apcer = get_apcer_at_given_bpcer(far, frr, thresholds, 0.05)
                # print(f"APCER at 5% BPCER: {apcer_at_5_bpcer:.4} or {apcer_at_5_bpcer * 100:.4}%, Threshold: {threshold_for_apcer:.4}")
                # logging.info(f"APCER at 5% BPCER: {apcer_at_5_bpcer:.4} or {apcer_at_5_bpcer * 100:.4}%, Threshold: {threshold_for_apcer:.4}")

                # bpcer_at_5_apcer, threshold_for_bpcer = get_bpcer_at_given_apcer(far, frr, thresholds, 0.05)
                # print(f"BPCER at 5% APCER: {bpcer_at_5_apcer:.4} or {bpcer_at_5_apcer * 100:.4}%, Threshold: {threshold_for_bpcer:.4}")
                # logging.info(f"BPCER at 5% APCER: {bpcer_at_5_apcer:.4} or {bpcer_at_5_apcer * 100:.4}%, Threshold: {threshold_for_bpcer:.4}")

                apcer_at_5_bpcer, threshold_for_apcer = get_apcer_at_given_bpcer(far, frr, thresholds, pct)
                print(f"APCER at {pct*100} BPCER: {apcer_at_5_bpcer:.4} or {apcer_at_5_bpcer * 100:.4}%, Threshold: {threshold_for_apcer:.4}")
                logging.info(f"APCER at 10% BPCER: {apcer_at_5_bpcer:.4} or {apcer_at_5_bpcer * 100:.4}%, Threshold: {threshold_for_apcer:.4}")

                bpcer_at_5_apcer, threshold_for_bpcer = get_bpcer_at_given_apcer(far, frr, thresholds, pct)
                print(f"BPCER at {pct*100} APCER: {bpcer_at_5_apcer:.4} or {bpcer_at_5_apcer * 100:.4}%, Threshold: {threshold_for_bpcer:.4}")
                logging.info(f"BPCER at {pct*100} APCER: {bpcer_at_5_apcer:.4} or {bpcer_at_5_apcer * 100:.4}%, Threshold: {threshold_for_bpcer:.4}")

                # Store APCER and BPCER values in their respective dictionaries
                apcer_dict[f"{model_name}_{dataset_name}"] = apcer_at_5_bpcer
                bpcer_dict[f"{model_name}_{dataset_name}"] = bpcer_at_5_apcer
                
            else:
                logging.error(f"Skipping APCER/BPCER calculation for {model_name},  due to missing scores.")

    # Save all APCER and BPCER values to files
    save_apcer_bpcer_values(apcer_dict, bpcer_dict, save_dir, db_name, pct, avg = True)



models = {
    "spatial_attn" : None,
    # "channel_attn" : None,
    "self_attn" : None,
}


# save_dir = "scores/"
# # compute_apcer_bpcer_for_all_models(models, morph_types, root_dirs, printer, save_dir)
# compute_avg_apcer_bpcer_for_all_models(models, morph_types, root_dirs, printer, save_dir, 0.6)



def calculate_eer(genuine, imposter, bins=10_001, batch_size=5000):
    if not len(genuine) or not len(imposter):
        logging.error("Genuine or imposter scores are empty!")
        return None

    genuine = np.array(genuine)
    imposter = np.array(imposter)

    min_threshold = min(genuine.min(), imposter.min())
    max_threshold = max(genuine.max(), imposter.max())
    thresholds = np.linspace(min_threshold, max_threshold, bins)

    far = np.zeros(bins)
    frr = np.zeros(bins)

    
    num_batches = len(imposter) // batch_size + (1 if len(imposter) % batch_size > 0 else 0)
    for i in tqdm(range(num_batches), desc = "FAR"):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(imposter))
        imposter_batch = imposter[start:end]

        far += np.sum(imposter_batch[:, None] >= thresholds, axis=0) / len(imposter)

    num_batches = len(genuine) // batch_size + (1 if len(genuine) % batch_size > 0 else 0)
    for i in tqdm(range(num_batches), desc = "FRR"):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(genuine))
        genuine_batch = genuine[start:end]

        frr += np.sum(genuine_batch[:, None] < thresholds, axis=0) / len(genuine)

    diff = np.abs(far - frr)
    min_diff_idx = np.argmin(diff)
    eer = (far[min_diff_idx] + frr[min_diff_idx]) / 2

    logging.info(f"EER calculated: {eer}")
    print(f"EER calculated: {eer}")
    return eer


# # sota_name = 'Zhang_etal'
# # sota_name = 'Tapia_etal'
# # sota_name = 'clip'
# sota_name = 'self_attn'
# sota_name = 'spatial_channel_mult_12'
# sota_name = 'spatial_channel_12'
# sota_name = 'spatial_channel_12_depth'
# sota_name = 'dgcnn_simpleview_svm'
from pathlib import Path
import pandas as pd
import re

def natural_sort_key(key):
    """Extracts numeric parts from protocol names for proper sorting."""
    match = re.search(r'(\d+)', key)  # Extract the numeric part
    return int(match.group(1)) if match else float('inf')  # Default to inf for non-numeric keys


def main():
    logger = get_logger(filename = "metrics")
   
    # all_files = [f for f in path.rglob("*") if f.is_file()]

    # for file in all_files:
    #     print(file)

    scores_dir = Path("scores")

    results = {}
    
    for protocol_path in scores_dir.iterdir():
        if not protocol_path.is_dir(): 
            continue
        
        protocol_number = protocol_path.name  
        
        for model_path in protocol_path.rglob("*"):
            if not model_path.is_dir():
                continue
            
            parts = model_path.relative_to(protocol_path).parts
            print(f"parts: {parts}")
            if len(parts) < 3:  
                print("incorrect structre")
                continue  
            
            model_name, dataset, morph_type = parts[:3]
            logger.info(f"calculating metrics for {model_name} on {dataset}")
            
            genuine_path = model_path / "genuine.npy"
            imposter_path = model_path / "imposter.npy"
            
            if genuine_path.exists() and imposter_path.exists():
                genuine_scores = np.load(genuine_path)
                imposter_scores = np.load(imposter_path)

                eer = calculate_eer(genuine_scores, imposter_scores)
                eer = eer*100
                logger.info(f"eer: {eer}")
                far, frr, thresholds = get_far_frr_thresholds(genuine_scores, imposter_scores)
                
                bpcer1, threshold = get_bpcer_at_given_apcer(far, frr, thresholds, 0.01)
                logger.info(f"BPCER at 1% APCER: {bpcer1:.4} or {bpcer1 * 100:.4}%, Threshold: {threshold:.4}")
                bpcer2, threshold = get_bpcer_at_given_apcer(far, frr, thresholds, 0.05)
                logger.info(f"BPCER at 5% APCER: {bpcer2:.4} or {bpcer2 * 100:.4}%, Threshold: {threshold:.4}")
                bpcer3, threshold = get_bpcer_at_given_apcer(far, frr, thresholds, 0.1)
                logger.info(f"BPCER at 10% APCER: {bpcer3:.4} or {bpcer3 * 100:.4}%, Threshold: {threshold:.4}")
    
                key = (model_name)
                if key not in results:
                    results[key] = {}
                
                results[key][protocol_number] = (eer, bpcer1*100, bpcer2*100, bpcer3*100)

                
        metrics_df = pd.DataFrame.from_dict(results, orient="index")

        # Flatten tuples into separate columns
        # metrics_df = metrics_df.apply(lambda row: pd.Series({f"{protocol}_{metric}": value 
                                                            # for protocol, values in row.items() 
                                                            # for metric, value in zip(["EER", "BPCER@1", "BPCER@5", "BPCER@10"], values)}), axis=1)
        metrics_df = metrics_df.apply(lambda row: pd.Series({
            f"{protocol}_{metric}": value 
            for protocol, values in row.items()  
            if isinstance(values, (list, tuple))  # Ensure values is iterable
            for metric, value in zip(["EER", "BPCER@1", "BPCER@5", "BPCER@10"], values)
        }), axis=1)

        # Reset index and add column names
        # metrics_df.index = pd.MultiIndex.from_tuples(metrics_df.index, names=["Model"])
        metrics_df.index.name = "Model"

        # sorted_columns = sorted(metrics_df.columns, key=natural_sort_key)
        # metrics_df = metrics_df[sorted_columns]  # Reorder columns

        # Save to CSV
        metrics_df.to_csv("metrics_summary.csv")

        # Print table preview
        print(metrics_df.head())





    
# sota_name = 'vit_svm_12'


# # save_dir = f"scores/{sota_name}/iPhone11_filled/lmaubo"
# save_dir = f"scores/{sota_name}/iPhone12_filled/lmaubo"
# # save_dir = f"scores/sota/{sota_name}/digital/lmaubo"
# genuine_path = f"{save_dir}/genuine.npy"
# imposter_path = f"{save_dir}/imposter.npy" 

    

# # EG:
# genuine, impostor = np.load(genuine_path), np.load(imposter_path)
# far, frr, thresholds = get_far_frr_thresholds(genuine, impostor)

# print(sota_name)
# # # apcer at 5% bpcer
# # apcer, threshold = get_apcer_at_given_bpcer(far, frr, thresholds, 0.1)
# # print(f"APCER: {apcer:.4} or {apcer * 100:.4}%, Threshold: {threshold:.4}")

# # # bpcer at 5% apcer
# bpcer, threshold = get_bpcer_at_given_apcer(far, frr, thresholds, 0.01)
# print(f"BPCER at 1% APCER: {bpcer:.4} or {bpcer * 100:.4}%, Threshold: {threshold:.4}")
# bpcer, threshold = get_bpcer_at_given_apcer(far, frr, thresholds, 0.05)
# print(f"BPCER at 5% APCER: {bpcer:.4} or {bpcer * 100:.4}%, Threshold: {threshold:.4}")
# bpcer, threshold = get_bpcer_at_given_apcer(far, frr, thresholds, 0.1)
# print(f"BPCER at 10% APCER: {bpcer:.4} or {bpcer * 100:.4}%, Threshold: {threshold:.4}")

# calculate_eer(genuine, impostor)

if __name__ == '__main__':
    set_seed()
    main()
    
