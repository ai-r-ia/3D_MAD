import os
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import pickle
import logging
from torch.nn.functional import cosine_similarity
from tqdm import tqdm
import random
import torch.nn.functional as F
from itertools import combinations
from typing import List


from datasets.datasetwrapper import DatasetWrapper
from models.dual_attn import DualAttentionModel, SingleAttentionModel
from models.resnet_attn import AttentionResNet, AttentionResNet2


SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Parameters
AUGMENT_TIMES = 2
num_epochs = 300
patience = 5
batch_size = 128  # Adjust as needed
num_workers = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Logging setup
log_file = "logs/scores.log"

# Enable CuDNN for optimal performance
torch.backends.cudnn.benchmark = True

# Morph types and root directornies to process
DATASET_NAME = "iPhone12_filled"
# DATASET_NAME = "3D_Morphing_DB_Jag"
printer = "digital"
morph_types = [
               "lmaubo"
            #    "3D_morph"
               ]
root_dirs = [
    f"/mnt/extravolume/data/{DATASET_NAME}/color/digital/"
        # f"/home/ubuntu/cluster/nbl-users/Shreyas-Sushrut-Raghu/ria/3DFace_DB/{DATASET_NAME}/color/digital/"
        # f"/home/ubuntu/cluster/nbl-users/Shreyas-Sushrut-Raghu/ria/3DFace_DB/3D_Morphing_DB_Jag/3DNTNU_Morphing_DB/"
             ]

def compute_deer(root_dirs = root_dirs, morph_types = morph_types) -> None:
    # Differential equal eror rate-DEER between dataset pairs
    logging.debug("Starting D-EER computation for all models...")
    for root_dir in root_dirs:
        deer_records = []
        dataset_name = os.path.basename(os.path.dirname(root_dir)) 
        eer_table = {}

        for morph_type in morph_types:
            eer_file = f"scores/eer_table_{dataset_name}_{morph_type}_{printer}.pkl"
            with open(eer_file, "rb") as handle:
                eer_data = pickle.load(handle)
                for model_name, eer in eer_data.items():
                    if model_name not in eer_table:
                        eer_table[model_name] = {}
                    eer_table[model_name][morph_type] = eer

            # for model_name, eers in eer_table.items():
            #     for d1, d2 in combinations(morph_types, 2):
            #         eer1 = eer_table[model_name][d1]
            #         eer2 = eer_table[model_name][d2]
            #         deer = abs(eer1 - eer2)
            #         deer_records.append(
            #             {"Model": model_name, "Dataset_Pair": f"{d1} vs {d2}", "DEER": deer}
            #         )

            # logger.info("DEER Table:")
            # for record in deer_records:
            #     logger.info(
            #         f"{record['Model']} on {record['Dataset_Pair']}: DEER = {record['DEER']:.4f}"
            #     )

            # with open(f"scores/deer_table_{dataset_name}_{morph_type}_{printer}.pkl", "wb") as file:
            #     pickle.dump(deer_records, file)     
                
        for model_name, eers in eer_table.items():
            for d1, d2 in combinations(eers.keys(), 2):  # Compare all morph_types
                deer = abs(eers[d1] - eers[d2])
                deer_records.append(
                    {"Model": model_name, "Dataset_Pair": f"{d1} vs {d2}", "DEER": deer}
                )

        logging.info("DEER Table:")
        for record in deer_records:
            logging.info(f"{record['Model']} on {record['Dataset_Pair']}: DEER = {record['DEER']:.4f}")

        with open(f"scores/deer_table_{dataset_name}_{printer}.pkl", "wb") as file:
            pickle.dump(deer_records, file)           

    return


def calculate_eer(genuine, imposter, bins=10_001, batch_size=5000):
    if not len(genuine) or not len(imposter):
        logging.error("Genuine or imposter scores are empty!")
        return None

    genuine = np.array(genuine)
    imposter = np.array(imposter)

    min_threshold = min(genuine.min(), imposter.min())
    max_threshold = max(genuine.max(), imposter.max())
    thresholds = np.linspace(min_threshold, max_threshold, bins)

    # Initialize FAR and FRR arrays
    far = np.zeros(bins)
    frr = np.zeros(bins)

    # Process in batches to avoid memory issues
    num_batches = len(imposter) // batch_size + (1 if len(imposter) % batch_size > 0 else 0)
    for i in tqdm(range(num_batches), desc = "FAR"):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(imposter))
        imposter_batch = imposter[start:end]

        # Calculate FAR for the batch
        far += np.sum(imposter_batch[:, None] >= thresholds, axis=0) / len(imposter)

    # Process genuine scores similarly
    num_batches = len(genuine) // batch_size + (1 if len(genuine) % batch_size > 0 else 0)
    for i in tqdm(range(num_batches), desc = "FRR"):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(genuine))
        genuine_batch = genuine[start:end]

        # Calculate FRR for the batch
        frr += np.sum(genuine_batch[:, None] < thresholds, axis=0) / len(genuine)

    # Compute EER
    diff = np.abs(far - frr)
    min_diff_idx = np.argmin(diff)
    eer = (far[min_diff_idx] + frr[min_diff_idx]) / 2

    logging.info(f"EER calculated: {eer}")
    return eer


def classify(enrollfeat, probefeat, chunk_size=512):
    """Compute cosine similarity in chunks to reduce memory overhead."""
    
    # Normalize features for cosine similarity
    enrollfeat = F.normalize(enrollfeat, p=2, dim=1)
    probefeat = F.normalize(probefeat, p=2, dim=1)

    num_enroll = enrollfeat.size(0)
    num_probe = probefeat.size(0)

    # Pre-allocate similarity matrix
    similarity_matrix = torch.empty((num_enroll, num_probe), device=enrollfeat.device, dtype=enrollfeat.dtype)

    for i in range(0, num_enroll, chunk_size):
        chunk = enrollfeat[i : i + chunk_size]  # Process chunk
        similarity_matrix[i : i + chunk_size] = torch.matmul(chunk, probefeat.T)  # Faster than pairwise cosine_similarity

    # Log similarity matrix for debugging
    # logging.debug(f"Similarity matrix: {similarity_matrix}")
    
    return similarity_matrix



# def classify(enrollfeat, probefeat, chunk_size=512):
#     """Compute cosine similarity in chunks to reduce memory overhead."""
#     similarities = []
#     for i in range(0, enrollfeat.size(0), chunk_size):
#         chunk = enrollfeat[i : i + chunk_size]
#         similarities.append(cosine_similarity(chunk.unsqueeze(1), probefeat.unsqueeze(0), dim=2))
#     return torch.cat(similarities, dim=0)

def eval_and_get_eer(model_name, model, morph_type:str, root_dir):
    logging.debug(f"Starting evaluation for model: {model_name}, morph: {morph_type}, root: {root_dir}")
    
    dataset_wrapper = DatasetWrapper(root_dir=root_dir)

    trainds = dataset_wrapper.get_train_dataset(
        augment_times=AUGMENT_TIMES,
        batch_size=batch_size,
        morph_types=[morph_type],
        num_models=1,
        shuffle=True,
        
    )
    # testds = dataset_wrapper.get_test_dataset(
    #     augment_times=AUGMENT_TIMES,
    #     batch_size=batch_size,
    #     morph_types=[morph_type],
    #     num_models=1,
    #     shuffle=True,
    # )
    testds = []
    # dataset_wrapper = DatasetWrapper(root_dir=root_dir, morph_type=morph_type, printer = printer)
    
    # trainds = dataset_wrapper.get_train_dataset(
    #     augment_times=AUGMENT_TIMES,
    #     batch_size=batch_size,
    #     morph_type=morph_type,
    #     shuffle=False,
    #     num_workers=num_workers,
    # )
    # testds = dataset_wrapper.get_test_dataset(
    #     augment_times=AUGMENT_TIMES,
    #     batch_size=batch_size,
    #     morph_type=morph_type,
    #     shuffle=False,
    #     num_workers=num_workers,
    # )

    model.to(device)
    model.eval()

    eer = None
    genuine_scores, imposter_scores = [], []

    # dataset_name = os.path.basename(os.path.dirname(root_dir)) 
    dataset_name = DATASET_NAME 
    save_dir = f"scores/{model_name}/{dataset_name}/{morph_type}"
    os.makedirs(save_dir, exist_ok=True)

    genuine_path = f"{save_dir}/genuine.npy"
    imposter_path = f"{save_dir}/imposter.npy"
    temp = False
    if temp and os.path.exists(genuine_path) and os.path.exists(imposter_path):
        logging.debug("Loading saved scores...")
        try:
            logging.info(f"Loading saved scores from {genuine_path} and {imposter_path}")
            genuine_scores = np.load(genuine_path)
            imposter_scores = np.load(imposter_path)

            if genuine_scores.size == 0 or imposter_scores.size == 0:
                logging.error(f"Loaded scores are empty! Genuine: {genuine_scores.size}, Imposter: {imposter_scores.size}")
            else:
                logging.info(f"Loaded {len(genuine_scores)} genuine and {len(imposter_scores)} imposter scores.")
        except Exception as e:
            logging.error(f"Error while loading scores: {e}")
            genuine_scores, imposter_scores = [], []
        
    else:
        with torch.no_grad():
            for batch_color, batch_depth, labels in tqdm(trainds, desc=f"Processing train for {model_name}, {morph_type}"):
                try:
                    batch_color, batch_depth = batch_color.to(device), batch_depth.to(device)
                    if(model_name.endswith("_color")):
                        preds = model(batch_color)
                    elif(model_name.endswith("_depth")):
                        preds = model(batch_depth)
                    else:
                        preds = model(batch_color, batch_depth)
                    # print("fesdilj")
                    preds = F.softmax(preds, dim=1)
                    # print(f"{model_name} | {morph_type} | preds.shape={preds.shape} | labels.shape={labels.shape}")
                    for pred, label in zip(preds, labels):
                        # print(label)  
                        label = label[0].item()

                        pred = pred.cpu().detach().view(-1)  
                        # print(pred)
                        if label == 1:
                            genuine_scores.append(pred[0])
                        else:
                            imposter_scores.append(pred[0])

                    del batch_color, batch_depth, preds
                    torch.cuda.empty_cache()
                except Exception as e:
                    logging.error(f"Error processing trainds batch: {e}")
                    continue

            for  batch_color, batch_depth, labels in tqdm(testds, desc=f"Processing test for {model_name}, {morph_type}"):
                try:
                    batch_color, batch_depth, =  batch_color.to(device), batch_depth.to(device)
                    if(model_name.endswith("_color")):
                        preds = model(batch_color)
                    elif(model_name.endswith("_depth")):
                        preds = model(batch_depth)
                    else:
                        preds = model( batch_color, batch_depth,)
                    preds = F.softmax(preds, dim=1)

                    for pred, label in zip(preds, labels):
                        label = label[0].item()

                        pred = pred.cpu().detach().view(-1)  
                        if label == 1:
                            genuine_scores.append(pred[0])
                        else:
                            imposter_scores.append(pred[0])

                    del  batch_color, batch_depth, preds
                    torch.cuda.empty_cache()
                except Exception as e:
                    logging.error(f"Error processing testds batch: {e}")
                    continue


    # Convert lists to tensors, then to NumPy before saving
    # genuine_scores = [torch.tensor(score) for score in genuine_scores]
    # imposter_scores = [torch.tensor(score) for score in imposter_scores]
    genuine_scores = [score.clone().detach() if isinstance(score, torch.Tensor) else torch.tensor(score) for score in genuine_scores]
    imposter_scores = [score.clone().detach() if isinstance(score, torch.Tensor) else torch.tensor(score) for score in imposter_scores]
    np.save(genuine_path, torch.stack(genuine_scores).numpy())
    np.save(imposter_path, torch.stack(imposter_scores).numpy())


    eer = calculate_eer(genuine_scores, imposter_scores)
    logging.info(f"EER: {eer} for model: {model_name}, morph: {morph_type}, root: {root_dir}")
    return eer

def compute_eer(models, morph_types, root_dirs):
    logging.debug("Starting EER computation for all models...")
    for root_dir in root_dirs:
        dataset_name = os.path.basename(os.path.dirname(root_dir)) 
        for morph_type in morph_types:
            eer_table = {}
            for model_name, model in models.items():
                logging.debug(f"Computing EER for model: {model_name}, morph: {morph_type}, root: {root_dir}")
                eer_table[model_name] = eval_and_get_eer(model_name, model, morph_type, root_dir)
            save_path = f"scores/eer_table_{dataset_name}_{morph_type}_{printer}.pkl"
            logging.debug(f"Saving EER table to {save_path}")
            with open(save_path, "wb") as f:
                pickle.dump(eer_table, f)

# def compute_eer(models, morph_types, root_dirs):
#     logging.debug("Starting EER computation for all models...")
#     for root_dir, morph_type in zip(root_dirs, morph_types):
#         dataset_name = os.path.basename(os.path.dirname(root_dir)) 
#         # for morph_type in morph_types:
#         eer_table = {}
#         for model_name, model in models.items():
#             logging.debug(f"Computing EER for model: {model_name}, morph: {morph_type}, root: {root_dir}")
#             eer_table[model_name] = eval_and_get_eer(model_name, model, morph_type, root_dir)
#         save_path = f"logs/scores/final/eer_table_{dataset_name}_{morph_type}.pkl"
#         logging.debug(f"Saving EER table to {save_path}")
#         with open(save_path, "wb") as f:
#             pickle.dump(eer_table, f)


# model3 = AttentionResNet(attention_types=["channel"])
# model4 = AttentionResNet(attention_types=["channel"])
# model_c = DualAttentionModel(model1=model3, model2=model4)
# pretrained_weights = torch.load('checkpoints/channel_11/channel_11_epoch_5.pth')
# model_c.load_state_dict(pretrained_weights, strict = False)
# model_c.eval()
# model5 = AttentionResNet(attention_types=["spatial"])
# model6 = AttentionResNet(attention_types=["spatial"])
# model_s = DualAttentionModel(model1=model5, model2=model6)
# pretrained_weights = torch.load('checkpoints/spatial_11/spatial_11_epoch_8.pth')
# model_s.load_state_dict(pretrained_weights, strict = False)
# model_s.eval()


# model3_12 = AttentionResNet(attention_types=["channel"])
# model4_12 = AttentionResNet(attention_types=["channel"])
# model_c_12 = DualAttentionModel(model1=model3_12, model2=model4_12)
# pretrained_weights = torch.load('checkpoints/channel_12/channel_12_epoch_6.pth')
# model_c_12.load_state_dict(pretrained_weights, strict = False)
# model_c_12.eval()
# model5_12 = AttentionResNet(attention_types=["spatial"])
# model6_12 = AttentionResNet(attention_types=["spatial"])
# model_s_12 = DualAttentionModel(model1=model5_12, model2=model6_12)
# pretrained_weights = torch.load('checkpoints/spatial_12/spatial_12_epoch_12.pth')
# model_s_12.load_state_dict(pretrained_weights, strict = False)
# model_s_12.eval()


# model7 = AttentionResNet(attention_types=[])
# model_cross = DualAttentionModel(model1=model7, model2=model7, cross = True)
# pretrained_weights = torch.load('checkpoints/cross/cross_epoch_15.pth')
# model_cross.load_state_dict(pretrained_weights, strict = False)
# model_cross.eval()


# model16 = AttentionResNet2(attention_types=["spatial","channel"])
# model17 = AttentionResNet2(attention_types=["spatial", "channel"])
# model_1 = DualAttentionModel(model1=model16, model2=model17)
# pretrained_weights = torch.load('checkpoints/spatial_channel_11/spatial_channel_11_epoch_5.pth')
# model_1.load_state_dict(pretrained_weights, strict = False)
# model_1.eval()
model16_12 = AttentionResNet2(attention_types=["spatial","channel"])
model17_12 = AttentionResNet2(attention_types=["spatial", "channel"])
model_1_12 = DualAttentionModel(model1=model16_12, model2=model17_12)
pretrained_weights = torch.load('checkpoints/spatial_channel_mult_12/spatial_channel_mult_12_epoch_14.pth')
model_1_12.load_state_dict(pretrained_weights, strict = False)
model_1_12.eval()


# single
# model3 = AttentionResNet(attention_types=["channel"])
# model_c = SingleAttentionModel(model=model3)
# pretrained_weights = torch.load('checkpoints/channel_12_color/channel_12_color_epoch_11.pth')
# model_c.load_state_dict(pretrained_weights, strict = False)
# model_c.eval()
# model_c_depth = SingleAttentionModel(model=model3)
# pretrained_weights = torch.load('checkpoints/channel_12_depth/channel_12_depth_epoch_12.pth')
# model_c_depth.load_state_dict(pretrained_weights, strict = False)
# model_c_depth.eval()
# model5 = AttentionResNet(attention_types=["spatial"])
# model_s = SingleAttentionModel(model=model5)
# pretrained_weights = torch.load('checkpoints/spatial_12_color/spatial_12_color_epoch_11.pth')
# model_s.load_state_dict(pretrained_weights, strict = False)
# model_s.eval()
# model_s_depth = SingleAttentionModel(model=model5)
# pretrained_weights = torch.load('checkpoints/spatial_12_depth/spatial_12_depth_epoch_8.pth')
# model_s_depth.load_state_dict(pretrained_weights, strict = False)
# model_s_depth.eval()
# model16 = AttentionResNet2(attention_types=["spatial","channel"])
# model_1 = SingleAttentionModel(model=model16)
# pretrained_weights = torch.load('checkpoints/spatial_channel_12_color/spatial_channel_12_color_epoch_10.pth')
# model_1.load_state_dict(pretrained_weights, strict = False)
# model_1.eval()
# model_1_depth = SingleAttentionModel(model=model16)
# pretrained_weights = torch.load('checkpoints/spatial_channel_12_depth/spatial_channel_12_depth_epoch_10.pth')
# model_1_depth.load_state_dict(pretrained_weights, strict = False)
# model_1_depth.eval()

# attn_types = [["channel"], ["spatial"],["self"],["channel", "spatial"], ["channel", "self"], ["spatial", "channel", "self"], ["spatial", "self"]]
attn_types = [["channel", "spatial"], ["channel", "self"], ["spatial", "channel", "self"], ["spatial", "self"]]
models = {}

# for attn_type in attn_types:
#     model1 = AttentionResNet(attention_types=attn_type)
#     model2 = AttentionResNet(attention_types=attn_type)
#     model = DualAttentionModel(model1=model1, model2=model2)
#     model_name = "_".join(attn_type)
#     pretrained_weights = torch.load(f'checkpoints/{model_name}/{model_name}_best.pth')
#     model.load_state_dict(pretrained_weights, strict = True)
#     model.eval()
#     models[model_name] = model    

models = {
    # "spatial_channel_12_color": model_1,
    # "spatial_channel_12_depth": model_1_depth,
    # "channel_12_color" : model_c,
    # "spatial_12_color" : model_s,
    # "channel_12_depth" : model_c_depth,
    # "spatial_12_depth" : model_s_depth,
    # "channel_12" : model_c_12,
    # "spatial_12" : model_s_12,
    "spatial_channel_mult_12": model_1_12,
}

compute_eer(models, morph_types, root_dirs)
# compute_deer()


# _12 refers to models testsed on iPhone12_filled
# _11 refers to models testsed on iPhone11_filled 