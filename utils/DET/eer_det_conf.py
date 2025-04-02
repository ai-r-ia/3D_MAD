import numpy as np
import matplotlib.pyplot as plt

def param_conf_inter(FAR, FRR, num_imposteurs, num_clients):
    """
    Calculates a 90% confidence interval for each value of FAR and FRR using a parametric method.

    Parameters:
    FAR (numpy array): FAR vector
    FRR (numpy array): FRR vector
    num_imposteurs (int): Number of impostor tests
    num_clients (int): Number of client tests

    Returns:
    tuple: (FARconfMIN, FRRconfMIN, FARconfMAX, FRRconfMAX)
    """
    numErr = len(FAR)
    
    FRRconfMIN = np.zeros(numErr)
    FRRconfMAX = np.zeros(numErr)
    FARconfMIN = np.zeros(numErr)
    FARconfMAX = np.zeros(numErr)
    
    for i in range(numErr):
        varFRR = np.sqrt(FRR[i] * (1 - FRR[i]) / num_clients)
        FRRconfMIN[i] = FRR[i] - 1.645 * varFRR
        FRRconfMAX[i] = FRR[i] + 1.645 * varFRR
        
        varFAR = np.sqrt(FAR[i] * (1 - FAR[i]) / num_imposteurs)
        FARconfMIN[i] = FAR[i] - 1.645 * varFAR
        FARconfMAX[i] = FAR[i] + 1.645 * varFAR
    
    return FARconfMIN, FRRconfMIN, FARconfMAX, FRRconfMAX


def EER_DET_conf(clients, impostors, OPvalue, pas0):
    m0 = max(clients)
    num_clients = len(clients)
    m1 = min(impostors)
    num_impostors = len(impostors)
    pas1 = (m0 - m1) / pas0
    x = np.arange(m1, m0, pas1)
    
    FAR = []
    FRR = []
    
    for threshold in x:
        FRR.append(100 * sum(c < threshold for c in clients) / num_clients)
        FAR.append(100 * sum(i >= threshold for i in impostors) / num_impostors)
    
    FRR = np.array(FRR)
    FAR = np.array(FAR)
    
    diff = FRR - FAR
    tmps = np.where(diff <= 0)[0][-1]
    
    if (FAR[tmps] - FRR[tmps]) <= (FRR[tmps + 1] - FAR[tmps + 1]):
        EER = (FAR[tmps] + FRR[tmps]) / 2
        tmpEER = tmps
    else:
        EER = (FRR[tmps + 1] + FAR[tmps + 1]) / 2
        tmpEER = tmps + 1
    
    tmpOP = np.where(OPvalue - FAR <= 0)[0][-1]
    OP = FRR[tmpOP]
    
    FARconfMIN, FRRconfMIN, FARconfMAX, FRRconfMAX = param_conf_inter(FAR / 100, FRR / 100, num_impostors, num_clients)
    
    confInterEER = EER - 100 * (FARconfMIN[tmpEER] + FRRconfMIN[tmpEER]) / 2
    confInterOP = OP - 100 * FRRconfMIN[tmpOP]
    
    plt.figure(1)
    plt.plot(x, FRR, 'r', label='FRR')
    plt.plot(x, FAR, 'b', label='FAR')
    plt.xlabel('Threshold')
    plt.ylabel('Error')
    plt.title('FAR vs FRR graph')
    plt.legend()
    plt.savefig("far_vs_frr.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    plt.figure(2)
    plt.plot(FAR, 100 - FRR, 'r')
    plt.xlabel('Impostor Attempts Accepted = FAR (%)')
    plt.ylabel('Genuine Attempts Accepted = 1-FRR (%)')
    plt.title('ROC curve')
    plt.scatter(EER, 100 - EER, color='k', label='EER')
    plt.scatter(FAR[tmpOP], 100 - FRR[tmpOP], color='g', label='OP')
    plt.legend()
    plt.savefig("roc_curve.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    return EER, FRR, FAR, confInterEER, OP, confInterOP

    
sota_name = 'Tapia_etal'
 
save_dir = f"scores/sota/{sota_name}"
genuine_path = f"{save_dir}/genuine.npy"
imposter_path = f"{save_dir}/imposter.npy" 
 
genuine, impostor = np.load(genuine_path), np.load(imposter_path)

EER_DET_conf(genuine, impostor, OPvalue = 0, pas0 = 10001)
