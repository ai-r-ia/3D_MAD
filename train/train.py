import os
from configs.config import create_parser, get_logger
from configs.seed import set_seed
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split
from models.dual_attn import DualAttentionModel, SingleAttentionModel
import torch
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm
import random
import numpy as np
from datasets.datasetwrapper import DatasetWrapper
from typing import List
from torch.optim.adam import Adam
from torch.optim.adamw import AdamW
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
import multiprocessing
from multiprocessing import Pool

from train_classifier import train_classifier
from utils.early_stopping import EarlyStopping
from utils.save_plots import save_plots
from models.resnet_attn import AttentionResNet2, AttentionResNet

SINGLE = 1
DUAL = 2

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_CONCURRENT_PROCESSES = 2
AUGMENT_TIMES = 2
alpha = 0.5  
num_epochs = 300
batch_size = 64
learning_rate = 1e-4
patience = 5

def train_model(model, model_name: str, trainds, testds, logger, modeltype:int = DUAL, imagetype = None)-> int: 
    
    os.makedirs(f"checkpoints/Protocol_0/{model_name}", exist_ok=True)
    
    # logging.basicConfig(filename=log_file, level=logger.INFO, format="%(asctime)s - %(message)s")
    # logging = logger.getLogger()
    optimizer = Adam([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(logger, patience=patience)
    validation_after = 1
    scaler = torch.cuda.amp.GradScaler()  
    
    logger.info(f"Training {model_name}")
    model.to(device)
    model.train()
    train_accuracies = []
    losses = []
    test_accuracy_list = []
    testing_loss_list = []
    plot_epoch_train = 0
    plot_epoch_test = 0
    for epoch in range(num_epochs):
        total_loss = 0
        bon_correct = 0
        bon_incorrect = 0
        mor_correct = 0
        mor_incorrect = 0
        total_train_samples = 0
        

        for color_imgs, depth_imgs, labels in tqdm(trainds, desc="Training"):
            color_imgs, depth_imgs, labels = color_imgs.to(device, non_blocking=True), depth_imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            if labels.dim() > 1:
                y = labels
                # print(labels)
                labels = torch.argmax(labels, dim=1)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                if(modeltype == SINGLE):
                    if(imagetype == 'color'):
                        outputs = model(color_imgs)
                    elif(imagetype == 'depth'):
                        outputs = model(depth_imgs)
                else:
                    outputs = model(color_imgs, depth_imgs)
                loss = criterion(outputs, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            bcorrect, bincorrect, mcorrect, mincorrect = train_classifier(outputs, y)
            # loss.backward()
            # optimizer.step()
            total_loss += loss.item()

            bon_correct += bcorrect
            bon_incorrect += bincorrect
            mor_correct += mcorrect
            mor_incorrect += mincorrect
            total_train_samples += len(labels)
            # total_loss /= total_train_samples

        accuracy = (bon_correct + mor_correct) / total_train_samples * 100
        train_accuracies.append(accuracy)
        losses.append(total_loss)
        plot_epoch_train = epoch
        
        scheduler.step()
        del color_imgs, depth_imgs, labels, outputs, loss
        torch.cuda.empty_cache()
        logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%")
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
        if not epoch % validation_after:
            validation_loss = 0.0
            total_test_samples = 0
            bon_correct = 0
            bon_incorrect = 0
            mor_correct = 0
            mor_incorrect = 0
            model.eval()

            with torch.no_grad():
                for  x1, x2, y in tqdm(testds, desc="Testing"):
                    x1,x2, y = x1.to(device),x2.to(device), y.to(device)
                    
                    if(modeltype == SINGLE):
                        if(imagetype == 'color'):
                            preds = model(x1)
                        elif(imagetype == 'depth'):
                            preds = model(x2)
                    else:
                        preds = model(x1,x2)
                        
                    bcorrect, bincorrect, mcorrect, mincorrect = train_classifier(preds, y)
                    labels = torch.argmax(y, dim=1)
                    batchloss = criterion(preds, labels)

                    validation_loss += batchloss.detach().cpu().item()
                    bon_correct += bcorrect
                    bon_incorrect += bincorrect
                    mor_correct += mcorrect
                    mor_incorrect += mincorrect
                    total_test_samples += len(y)
                    # validation_loss /= total_test_samples


            test_accuracy = (bon_correct + mor_correct) / total_test_samples * 100
            test_accuracy_list.append(test_accuracy)
            testing_loss_list.append(validation_loss)
            plot_epoch_test = epoch


            # Early stopping check
            early_stopping(validation_loss, model, f"checkpoints/Protocol_0/{model_name}/{model_name}_best.pth")
            if early_stopping.early_stop:
                logger.info("Early stopping triggered. Training stopped.")
                break

            logger.info(f"Epoch [{epoch+1}/{num_epochs}], test Loss: {validation_loss:.4f}, test Accuracy: {test_accuracy:.2f}%")
            print(f"Epoch [{epoch+1}/{num_epochs}], test Loss: {validation_loss:.4f}, test Accuracy: {test_accuracy:.2f}%")
        
        torch.save(model.state_dict(), f"checkpoints/Protocol_0/{model_name}/{model_name}_epoch_{epoch+1}.pth")
        logger.info(f"Checkpoint saved for {model_name} at epoch {epoch+1}")

        model.train()  # Ensure model is back in training mode after validation
        
    save_plots(list(range(0, plot_epoch_train + 1)), train_accuracies, losses, f"{model_name}_train")
    save_plots(list(range(0, plot_epoch_test + 1)), test_accuracy_list, testing_loss_list, f"{model_name}_test")

    return plot_epoch_train
    


def run_model(attn_type, trainds_name, train_dataset, test_dataset, reduction, kernel_size, logger):
# def run_model(args):
    torch.cuda.set_device(0)  
    # attn_type, train_dataset, test_dataset, reduction, kernel_size = args  # Unpack here
    logger.info(f"Running model with attn_type={attn_type}, reduction={reduction}, kernel_size={kernel_size}")
    logger.info(f"Training {attn_type}")
    model_name = "_".join(attn_type)

    if False and len(attn_type) > 1:
        model1 = AttentionResNet2(attention_types=attn_type, reduction= reduction, kernel_size=kernel_size)
        model2 = AttentionResNet2(attention_types=attn_type, reduction= reduction, kernel_size=kernel_size)
        model = DualAttentionModel(model1=model1, model2=model2)
   
        train_model(model=model, model_name= f"{model_name}_{trainds_name}_{reduction}_{kernel_size}", trainds=train_dataset, testds=test_dataset, logger = logger)
        # train_model(model=model, model_name= f"{model_name}_{trainds_name}_{reduction}_{kernel_size}_add", trainds=train_dataset, testds=test_dataset, logger = logger)
   
    else:
        model1 = AttentionResNet2(attention_types=attn_type, reduction= reduction, kernel_size=kernel_size)
        model = SingleAttentionModel(model=model1)
        
        train_model(model=model, model_name= f"{model_name}_{trainds_name}_color", trainds=train_dataset, testds=test_dataset, logger = logger,modeltype = SINGLE, imagetype = 'color')
        
        model1_depth = AttentionResNet2(attention_types=attn_type, reduction= reduction, kernel_size=kernel_size)
        depth_model = SingleAttentionModel(model=model1_depth)
        
        train_model(model=depth_model, model_name= f"{model_name}_{trainds_name}_depth", trainds=train_dataset, testds=test_dataset, logger = logger, modeltype = SINGLE, imagetype = 'depth')
        
        # model1_cmbd = AttentionResNet2(attention_types=attn_type, reduction= reduction, kernel_size=kernel_size)
        # model2_cmbd = AttentionResNet2(attention_types=attn_type, reduction= reduction, kernel_size=kernel_size)
        # model_cmbd = DualAttentionModel(model1=model1_cmbd, model2=model2_cmbd)
        # train_model(model=model_cmbd, model_name= f"{model_name}_{trainds_name}_cmbd", trainds=train_dataset, testds=test_dataset, logger = logger)

    logger.info(f"{attn_type} training done")
    
   
def main(args):
    protocol_num = 0
    # logger = get_logger(filename = "proposed_params_ablation2", protocol = protocol_num)
    logger = get_logger(filename = "proposed_imgtype_ablation2", protocol = protocol_num)
    # logger = get_logger(filename = "proposed_attn_ablation2", protocol = protocol_num)
    logger.info(f"training proposed on {args.trainds}")
    dataset_wrapper = DatasetWrapper(root_dir=f"{args.root_dir}/{args.trainds}_filled/color/digital/")

    train_dataset = dataset_wrapper.get_train_dataset(
        augment_times=AUGMENT_TIMES,
        batch_size=batch_size,
        morph_types=["lmaubo"],
        num_models=1,
        shuffle=True,
    )

    full_dataset = train_dataset.dataset

    one_hot_labels = np.array([full_dataset[i][2] for i in range(len(full_dataset))]) 

    labels = np.argmax(one_hot_labels, axis=1)

    train_indices, val_indices = train_test_split(
        np.arange(len(full_dataset)), test_size=0.3, stratify=labels, random_state=42
    )

    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    train_labels, val_labels = one_hot_labels[train_indices], one_hot_labels[val_indices]  

    logger.info(f"Total samples: {len(full_dataset)}")
    logger.info(f"Train samples: {len(train_indices)} ({len(train_indices)/len(full_dataset)*100:.2f}%)")
    logger.info(f"Validation samples: {len(val_indices)} ({len(val_indices)/len(full_dataset)*100:.2f}%)")
   
    unique_classes, train_counts = np.unique(np.argmax(train_labels, axis=1), return_counts=True)
    unique_classes, val_counts = np.unique(np.argmax(val_labels, axis=1), return_counts=True)

    logger.info(f"Class distribution in Train set: {dict(zip(unique_classes, train_counts))}")
    logger.info(f"Class distribution in Validation set: {dict(zip(unique_classes, val_counts))}")
    
    multiprocessing.set_start_method("spawn", force=True)
    # attn_types = [ ["channel"], ["spatial"]]
    reductions = [4]
    kernel_sizes = [5]    
    attn_types = [["spatial", "channel"]]
    # reductions = [4, 8, 16]
    # kernel_sizes = [3,5,7]    
    processes = []
   
    for reduction in reductions:
        for kernel_size in kernel_sizes: 
            for attn_type in attn_types:
               run_model(attn_type=attn_type, trainds_name=args.trainds, train_dataset= train_loader, test_dataset=val_loader, reduction=reduction, kernel_size=kernel_size, logger = logger)
   
    # CROSS ATTN TRAINING
    # model1 = AttentionResNet2(attention_types=[])
    # model2 = AttentionResNet2(attention_types=[])
    # model = DualAttentionModel(model1=model1, model2=model2, cross = True)
    # train_model(model=model, model_name= "cross", trainds=train_dataset, testds=test_dataset)


    # for reduction in reductions:
    #     for kernel_size in kernel_sizes:
    #         for attn_type in attn_types:
    #             p = multiprocessing.Process(target=run_model, args=(attn_type, train_dataset, test_dataset, reduction, kernel_size))
    #             processes.append(p)
    #             p.start()

    # for p in processes:
    #     p.join()

    # with Pool(processes=os.cpu_count()) as pool:
    #     args_list = [(attn_type, train_dataset, test_dataset, reduction, kernel_size) 
    #                  for reduction in reductions 
    #                  for kernel_size in kernel_sizes 
    #                  for attn_type in attn_types]

    #     pool.map(run_model, args_list)
        
    logger.info("All proposed models finished training.")
     

if __name__ == "__main__":
    set_seed()
    parser = create_parser()
    args = parser.parse_args()
    main(args)
    