import os
from models.dual_attn import DualAttentionModel, SingleAttentionModel
import torch
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm
import random
import numpy as np
from datasets.datasetwrapper import DatasetWrapper
import logging
from typing import List
from torch.optim.adam import Adam
from torch.optim.adamw import AdamW
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
import multiprocessing

from train_classifier import train_classifier
from utils.early_stopping import EarlyStopping
from utils.save_plots import save_plots
from models.resnet_attn import AttentionResNet2, AttentionResNet

SINGLE = 1
DUAL = 2

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


AUGMENT_TIMES = 2
alpha = 0.5  
num_epochs = 300
batch_size = 64
learning_rate = 1e-4
patience = 5

def train_model(model, model_name: str, trainds, testds, modeltype:int = DUAL, imagetype = None)-> int: 
    
    os.makedirs(f"checkpoints/{model_name}", exist_ok=True)
    log_file = f"logs/{model_name}.log"
    logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(message)s")
    logger = logging.getLogger()
    optimizer = Adam([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(logger, patience=patience)
    validation_after = 1
    scaler = torch.cuda.amp.GradScaler()  
    
    print(f"Training {model_name}")
    logging.info(f"Training {model_name}")
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
        logging.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%")
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
            early_stopping(validation_loss, model, f"checkpoints/{model_name}/{model_name}_best.pth")
            if early_stopping.early_stop:
                logging.info("Early stopping triggered. Training stopped.")
                break

            logging.info(f"Epoch [{epoch+1}/{num_epochs}], test Loss: {validation_loss:.4f}, test Accuracy: {test_accuracy:.2f}%")
            print(f"Epoch [{epoch+1}/{num_epochs}], test Loss: {validation_loss:.4f}, test Accuracy: {test_accuracy:.2f}%")
        
        torch.save(model.state_dict(), f"checkpoints/{model_name}/{model_name}_epoch_{epoch+1}.pth")
        logging.info(f"Checkpoint saved for {model_name} at epoch {epoch+1}")

        model.train()  # Ensure model is back in training mode after validation
        
    save_plots(list(range(0, plot_epoch_train + 1)), train_accuracies, losses, f"{model_name}_train")
    save_plots(list(range(0, plot_epoch_test + 1)), test_accuracy_list, testing_loss_list, f"{model_name}_test")

    return plot_epoch_train
    


def run_model(attn_type, train_dataset, test_dataset):
    torch.cuda.set_device(0)  
    print(f"Training {attn_type}")
    model1 = AttentionResNet2(attention_types=[attn_type])
    model2 = AttentionResNet2(attention_types=[attn_type])
    model = DualAttentionModel(model1=model1, model2=model2)
    train_model(model=model, model_name=attn_type, trainds=train_dataset, testds=test_dataset)
    print(f"{attn_type} training done")
    
   
def main():
    
    dataset_wrapper = DatasetWrapper(root_dir="/mnt/extravolume/data/iPhone11_filled/color/digital/")

    train_dataset = dataset_wrapper.get_train_dataset(
        augment_times=AUGMENT_TIMES,
        batch_size=batch_size,
        morph_types=["lmaubo"],
        num_models=1,
        shuffle=True,
    )
    test_dataset = dataset_wrapper.get_test_dataset(
        augment_times=AUGMENT_TIMES,
        batch_size=batch_size,
        morph_types=["lmaubo"],
        num_models=1,
        shuffle=True,
    )

    multiprocessing.set_start_method("spawn", force=True)
    # attn_types = [["spatial", "channel"], ["channel"], ["spatial"]]
    attn_types = [["spatial", "channel"]]
    
    processes = []
    
    for attn_type in attn_types:
        if len(attn_type) > 1:
            model1 = AttentionResNet2(attention_types=attn_type)
            model2 = AttentionResNet2(attention_types=attn_type)
        else:
            model1 = AttentionResNet(attention_types=attn_type)
            model2 = AttentionResNet(attention_types=attn_type)
            
        model = DualAttentionModel(model1=model1, model2=model2)
        # model = SingleAttentionModel(model=model1)
        model_name = "_".join(attn_type)
        train_model(model=model, model_name= f"{model_name}_mult_12", trainds=train_dataset, testds=test_dataset)
        # train_model(model=model, model_name= f"{model_name}_12_color", trainds=train_dataset, testds=test_dataset, modeltype = SINGLE, imagetype = 'color')
        # train_model(model=model, model_name= f"{model_name}_12_depth", trainds=train_dataset, testds=test_dataset, modeltype = SINGLE, imagetype = 'depth')

    # CROSS ATTN TRAINING
    # model1 = AttentionResNet2(attention_types=[])
    # model2 = AttentionResNet2(attention_types=[])
    # model = DualAttentionModel(model1=model1, model2=model2, cross = True)
    # train_model(model=model, model_name= "cross", trainds=train_dataset, testds=test_dataset)

    for attn_type in attn_types:
        p = multiprocessing.Process(target=run_model, args=(attn_type, train_dataset, test_dataset))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print("All models finished training.")
     

if __name__ == "__main__":
    SEED = 42
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    main()
    