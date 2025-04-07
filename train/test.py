import os
from configs.config import create_parser, get_logger, protocol_dict
from configs.seed import set_seed
import torch
import torch.nn as nn
# import torchvision.models as models
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


# Parameters
AUGMENT_TIMES = 2
num_epochs = 300
patience = 5
batch_size = 128  # Adjust as needed
num_workers = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

printer = "digital"
morph_types = ["lmaubo"]

def calculate_eer(genuine, imposter, logger, bins=10_001, batch_size=5000):
    if not len(genuine) or not len(imposter):
        logger.error("Genuine or imposter scores are empty!")
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

    logger.info(f"EER calculated: {eer}")
    return eer


def classify(enrollfeat, probefeat, chunk_size=512):
    """Compute cosine similarity in chunks to reduce memory overhead."""
    
    enrollfeat = F.normalize(enrollfeat, p=2, dim=1)
    probefeat = F.normalize(probefeat, p=2, dim=1)

    num_enroll = enrollfeat.size(0)
    num_probe = probefeat.size(0)

    similarity_matrix = torch.empty((num_enroll, num_probe), device=enrollfeat.device, dtype=enrollfeat.dtype)

    for i in range(0, num_enroll, chunk_size):
        chunk = enrollfeat[i : i + chunk_size]  
        similarity_matrix[i : i + chunk_size] = torch.matmul(chunk, probefeat.T)  
        
    return similarity_matrix



def eval_and_get_eer(model_name, model, morph_type:str, root_dir, args, logger):
    logger.debug(f"Starting evaluation for model: {model_name}, morph: {morph_type}, root: {root_dir}/{args.testds}")
    
    dataset_wrapper = DatasetWrapper(root_dir= f"{root_dir}/{args.testds}_filled/color/digital/")

    testds = dataset_wrapper.get_test_dataset(
        augment_times=AUGMENT_TIMES,
        batch_size=batch_size,
        morph_types=[morph_type],
        num_models=1,
        shuffle=True,
    )
    
    model.to(device)
    model.eval()

    eer = None
    genuine_scores, imposter_scores = [], []

    # dataset_name = os.path.basename(os.path.dirname(root_dir)) 
    dataset_name =args.testds 
    protocol_num = protocol_dict[f"{args.trainds}_{args.testds}"] 
    save_dir = f"scores/Protocol_{protocol_num}/{model_name}/{dataset_name}/{morph_type}"
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
                    logger.error(f"Error processing testds batch: {e}")
                    continue


    genuine_scores = [score.clone().detach() if isinstance(score, torch.Tensor) else torch.tensor(score) for score in genuine_scores]
    imposter_scores = [score.clone().detach() if isinstance(score, torch.Tensor) else torch.tensor(score) for score in imposter_scores]
    np.save(genuine_path, torch.stack(genuine_scores).numpy())
    np.save(imposter_path, torch.stack(imposter_scores).numpy())

    eer = calculate_eer(genuine_scores, imposter_scores, logger)
    logger.info(f"EER: {eer} for model: {model_name}, morph: {morph_type}, root: {root_dir}")
    return eer

def compute_eer(models, morph_types, root_dir, args, logger):
    logger.debug("Starting EER computation for all models...")
    dataset_name = os.path.basename(os.path.dirname(root_dir)) 
    for morph_type in morph_types:
        eer_table = {}
        for model_name, model in models.items():
            logger.debug(f"Computing EER for model: {model_name} on {args.testds}")
            eer_table[model_name] = eval_and_get_eer(model_name, model, morph_type, root_dir, args, logger)
        save_path = f"scores/eer_table_{dataset_name}_imgtype_ablation.pkl" #Note: change based on experiment
        logger.debug(f"Saving EER table to {save_path}")
        with open(save_path, "wb") as f:
            pickle.dump(eer_table, f)

def main(args):
    
    protocol_num = protocol_dict[f"{args.trainds}_{args.testds}"]
    logger = get_logger(filename = "scores_imgtype_ablation2", protocol = protocol_num) #Note: change based on experiment
    # logger = get_logger(filename = "scores_attn_ablation", protocol = protocol_num) #Note: change based on experiment
    # logger = get_logger(filename = "scores_params_ablation2", protocol = protocol_num) #Note: change based on experiment

    single = False
    models = {}
    reductions = [4]
    kernel_sizes = [5]    
    
    if single:
        attn_types = [ ["channel"], ["spatial"]]
        # img_types = ["color", "depth", "cmbd"]
        # attn_types = [  ["spatial"]]
        img_types = [ "cmbd"]
        for attn_type in attn_types:
            for img_type in img_types:
                for reduction in reductions:
                    for kernel_size in kernel_sizes: 
                        model_name = "_".join(attn_type)
                        model_name = f"{model_name}_{args.trainds}_{img_type}"
                        model1 = AttentionResNet2(attention_types=attn_type, reduction= reduction, kernel_size=kernel_size)
                        model2 = AttentionResNet2(attention_types=attn_type, reduction= reduction, kernel_size=kernel_size)
                        
                        if img_type == "cmbd":
                            model = DualAttentionModel(model1=model1, model2=model2)
                        else:
                            model = SingleAttentionModel(model=model1)
                            
                        pretrained_weights = torch.load(f'checkpoints/Protocol_0/{model_name}/{model_name}_best.pth')
                        model.load_state_dict(pretrained_weights, strict = False)
                        model.eval() 
                        models[model_name] = model
        
    else:
        attn_types = [["spatial", "channel"]]
        # reductions = [4, 8, 16]
        # kernel_sizes = [3,5,7] 
        img_types = ["color", "depth"]
        for attn_type in attn_types:
            # for img_type in img_types:
            for reduction in reductions:
                for kernel_size in kernel_sizes:  
                    model_name = "_".join(attn_type)
                    # model_name = f"{model_name}_{args.trainds}_{reduction}_{kernel_size}_add"
                    # model_name = f"{model_name}_{args.trainds}_{reduction}_{kernel_size}_mult"
                    model_name = f"{model_name}_{args.trainds}_final"
                    # model_name = f"{model_name}_{args.trainds}_{img_type}"
                    model1 = AttentionResNet2(attention_types=attn_type, reduction= reduction, kernel_size=kernel_size)
                    model2 = AttentionResNet2(attention_types=attn_type, reduction= reduction, kernel_size=kernel_size)
                    model = DualAttentionModel(model1=model1, model2=model2)
                    # model = SingleAttentionModel(model=model1)
                    pretrained_weights = torch.load(f'checkpoints/Protocol_0/{model_name}/{model_name}_best.pth')
                    model.load_state_dict(pretrained_weights, strict = False)
                    model.eval() 
                    models[model_name] = model
                        
    compute_eer(models, morph_types, args.root_dir, args, logger) 

if __name__ == "__main__":
    set_seed()
    parser = create_parser()
    args = parser.parse_args()
    main(args) 