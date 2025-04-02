import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


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

def plot_det(pmiss, pfa, plot_code='y', opt_thickness=0.5):
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
    print(len(pmiss), len(pfa))
    if len(pmiss) != len(pfa):
        raise ValueError("Vector size of Pmiss and Pfa must be equal in call to plot_det")
    
    set_det_limits()
    
    fig, ax = plt.subplots()
    h, = ax.plot(norm.ppf(pfa), norm.ppf(pmiss), plot_code, linewidth=opt_thickness)
    
    make_det(ax)
    plt.savefig("det_matlab.png", dpi=300, bbox_inches='tight')
    plt.show()
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
    DET_limits = (0.0005, 0.2, 0.0005, 0.2)

sota_name = 'Tapia_etal'
 
save_dir = f"scores/sota/{sota_name}"
genuine_path = f"{save_dir}/genuine.npy"
imposter_path = f"{save_dir}/imposter.npy" 
 
genuine, impostor = np.load(genuine_path), np.load(imposter_path)
FAR, FRR, thresholds = compute_far_frr(genuine, impostor, pas0=10001)
FAR = np.clip(FAR / 100, 1e-6, 1 - 1e-6)
FRR = np.clip(FRR / 100, 1e-6, 1 - 1e-6)
plot_det(FAR, FRR, 'r')