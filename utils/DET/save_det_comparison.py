from re import I
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def param_conf_inter(FAR, FRR, num_imposteurs, num_clients):
    """
    Calculates a 90% confidence interval for each value of FAR and FRR using a parametric method.
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

# def compute_far_frr(clients, impostors, pas0):
#     """
#     Computes FAR and FRR over a range of thresholds while smoothly adjusting EER to 50%.
#     """
#     m0 = max(clients)
#     num_clients = len(clients)
#     m1 = min(impostors)
#     num_impostors = len(impostors)
#     pas1 = (m0 - m1) / pas0
#     thresholds = np.arange(m1, m0, pas1)
    
#     FAR, FRR, valid_thresholds = [], [], []
    
#     for threshold in thresholds:
#         frr_value = 100 * sum(c < threshold for c in clients) / num_clients
#         far_value = 100 * sum(i >= threshold for i in impostors) / num_impostors

#         FRR.append(frr_value)
#         FAR.append(far_value)
#         valid_thresholds.append(threshold)

#     FAR = np.array(FAR)
#     FRR = np.array(FRR)
#     valid_thresholds = np.array(valid_thresholds)

#     # Find the threshold where FAR and FRR are closest
#     eer_index = np.argmin(np.abs(FAR - FRR))

#     # Shift EER to be exactly 50% by adjusting threshold
#     scale_factor = 50 / max(FAR[eer_index], FRR[eer_index])  # Rescale to 50%
#     FAR = np.clip(FAR * scale_factor, 0, 50)
#     FRR = np.clip(FRR * scale_factor, 0, 50)

#     return FAR, FRR, valid_thresholds


def compute_far_frr(clients, impostors, pas0):
    """
    Computes FAR and FRR over a range of thresholds.
    """
    m0 = max(clients)
    num_clients = len(clients)
    m1 = min(impostors)
    num_impostors = len(impostors)
    pas1 = (m0 - m1) / pas0
    thresholds = np.arange(m1, m0, pas1)
    
    FAR, FRR = [], []
    for threshold in thresholds:
        FRR.append(100 * sum(c < threshold for c in clients) / num_clients)
        FAR.append(100 * sum(i >= threshold for i in impostors) / num_impostors)

    return np.array(FAR), np.array(FRR), thresholds


def plot_det(ax, pmiss, pfa, plot_code='-', opt_thickness=0.5, label = 'model_name'):
    """
    Plot DET curve for detection performance tradeoff.
    
    Parameters:
        pmiss (array-like): Miss probabilities.
        pfa (array-like): False alarm probabilities.
        plot_code (str): Line color/style (default 'y').
        opt_thickness (float): Line thickness (default 0.5).
    
    Returns:
        Line2D object of the plot.
    """
    # print(len(pmiss), len(pfa))
    if len(pmiss) != len(pfa):
        raise ValueError("Vector size of Pmiss and Pfa must be equal in call to plot_det")
    
    set_det_limits()
    
    h, = ax.plot(norm.ppf(pfa), norm.ppf(pmiss), plot_code, linewidth=opt_thickness, label=label)
    
    make_det(ax)
    # plt.savefig("det_matlab.png", dpi=300, bbox_inches='tight')
    # plt.show()
    return h

def make_det(ax):
    """
    Creates a DET plot with logarithmic-scaled axes.
    """
    pticks = np.array([
        0.00001, 0.00002, 0.00005, 0.0001,  0.0002, 0.0005,
        0.001, 0.002, 0.005, 0.01, 0.02, 0.05,
        0.1, 0.2, 0.4, 0.6, 0.8, 0.9,
        0.95, 0.98, 0.99, 0.995, 0.998, 0.999,
        0.9995, 0.9998, 0.9999, 0.99995, 0.99998, 0.99999
    ])
    xlabels = ["{:.3g}".format(p*100) for p in pticks]
    ylabels = xlabels.copy()
    
    global DET_limits
    if DET_limits is None:
        set_det_limits()
    
    pmiss_min, pmiss_max, pfa_min, pfa_max = DET_limits
    
    tmin_miss = np.searchsorted(pticks, pmiss_min)
    tmax_miss = np.searchsorted(pticks, pmiss_max, side='right')
    tmin_fa = np.searchsorted(pticks, pfa_min)
    tmax_fa = np.searchsorted(pticks, pfa_max, side='right')
    
    ax.set_xlim(norm.ppf([pfa_min, pfa_max]))
    ax.set_xticks(norm.ppf(pticks[tmin_fa:tmax_fa]))
    ax.set_xticklabels(xlabels[tmin_fa:tmax_fa], rotation=45)
    ax.set_xlabel("False Acceptance Rate (in %)")
    
    ax.set_ylim(norm.ppf([pmiss_min, pmiss_max]))
    ax.set_yticks(norm.ppf(pticks[tmin_miss:tmax_miss]))
    ax.set_yticklabels(ylabels[tmin_miss:tmax_miss])
    ax.set_ylabel("False Reject Rate (in %)")
    
    ax.grid(True)
    ax.set_box_aspect(1)

def set_det_limits():
    """Sets the default DET limits."""
    global DET_limits
    DET_limits = (0.0005, 0.7, 0.0005, 0.7)
    # DET_limits = (0.0005, 0.5, 0.0005, 0.5) #preferred - used in last paper
    # DET_limits = (0.005, 0.5, 0.005, 0.5)


# plt.legend(['DET Curve'])

# # FAR vs FRR plot
# plt.subplot(1, 2, 1)
# for (sota_name, save_dir), color, linestyle in zip(datasets.items(), colors, linestyles):
#     genuine_path = f"{save_dir}/genuine.npy"
#     imposter_path = f"{save_dir}/imposter.npy"

#     genuine = np.load(genuine_path)
#     impostor = np.load(imposter_path)

#     FAR, FRR, thresholds = compute_far_frr(genuine, impostor, pas0=10001)
    
#     plt.plot(thresholds, FAR, linestyle=linestyle, color=color, label=f'FAR - {sota_name}')
#     plt.plot(thresholds, FRR, linestyle=linestyle, color=color, label=f'FRR - {sota_name}', alpha=0.6)

# plt.xlabel('Threshold')
# plt.ylabel('Error Rate (%)')
# plt.title('FAR vs FRR')
# plt.legend()

# # ROC curve
# plt.subplot(1, 2, 2)
# for (sota_name, save_dir), color, linestyle in zip(datasets.items(), colors, linestyles):
#     genuine_path = f"{save_dir}/genuine.npy"
#     imposter_path = f"{save_dir}/imposter.npy"

#     genuine = np.load(genuine_path)
#     impostor = np.load(imposter_path)

#     FAR, FRR, thresholds = compute_far_frr(genuine, impostor, pas0=10001)
    
#     plt.plot(FAR, 100 - FRR, linestyle=linestyle, color=color, label=sota_name)

# plt.xlabel('FAR (%)')
# plt.ylabel('Genuine Acceptance Rate (%)')
# plt.title('ROC Curve')
# plt.legend()

# plt.tight_layout()
# plt.savefig(f"det_combined_{protocol}.png", dpi=300, bbox_inches='tight')
# plt.savefig(f"det_combined_{protocol}.pdf", dpi=300, bbox_inches='tight')
# plt.show()

def save_det_plot(ax, datasets: dict, protocol: str):
    colors = ['r', 'b', 'g', 'm', 'y', 'c', 'orange']  # Colors for different datasets

    for (sota_name, save_dir), color in zip(datasets.items(), colors):
        print(sota_name)
        genuine_path = f"{save_dir}/genuine.npy"
        imposter_path = f"{save_dir}/imposter.npy"

        genuine = np.load(genuine_path)
        impostor = np.load(imposter_path)

        FAR, FRR, _ = compute_far_frr(genuine, impostor, pas0=10001)
        FAR = np.clip(FAR / 100, 1e-6, 1 - 1e-6)
        FRR = np.clip(FRR / 100, 1e-6, 1 - 1e-6)

        plot_det(ax, FAR, FRR, color, label=sota_name)

    ax.set_xlabel('MACER (%)', fontsize = 19)
    ax.set_ylabel('BPCER (%)', fontsize = 19)
    
    # Replace underscores with spaces for a more readable protocol title
    readable_protocol = protocol.replace('_', ' ')
    ax.set_title(f'DET Curve - {readable_protocol}', fontsize = 19)
    ax.legend(fontsize = "13")

    # Save individual figure for each protocol
    fig_filename = f"det_{protocol}.png"
    ax.figure.savefig(fig_filename, dpi=300, bbox_inches='tight')
    fig_filename_pdf = f"det_{protocol}.pdf"
    ax.figure.savefig(fig_filename_pdf, dpi=300, bbox_inches='tight')

def main():
    protocols = ["Protocol_1", "Protocol_2", "Protocol_3", "Protocol_4"]

    for protocol in protocols:
        fig, ax = plt.subplots(figsize=(6, 5))  # Create a single plot for each protocol

        dataset = "iPhone12" if protocol in ["Protocol_2", "Protocol_3"] else "iPhone11"
        model_dataset = "iPhone11" if protocol in ["Protocol_3", "Protocol_1"] else "iPhone12"
        datasets = {
            'SimpleView': f"scores/{protocol}/pointnet2_simpleview/{dataset}/lmaubo",
            'LBP-SVM': f"scores/{protocol}/lbp_svm/{dataset}/lmaubo",
            'ResNet50-SVM': f"scores/{protocol}/resnet_svm/{dataset}/lmaubo",
            'ViT-SVM': f"scores/{protocol}/vit_svm/{dataset}/lmaubo",
            'PointNet++': f"scores/{protocol}/pointnet2/{dataset}/lmaubo",
            'PointNet': f"scores/{protocol}/pointnet/{dataset}/lmaubo",
            '3DMDRNet (Proposed)': f"scores/{protocol}/spatial_channel_{model_dataset}_4_5/{dataset}/lmaubo",
        }
        save_det_plot(ax, datasets, protocol)

        # plt.tight_layout()
        # No need for `plt.show()` as each plot is saved individually
        plt.close()  # Close the figure after saving to free up memory

if __name__ == '__main__':
    main()
