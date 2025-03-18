import matplotlib.pyplot as plt 
import os

def save_plots(epochs, accuracies, losses, model):
    os.makedirs("plots", exist_ok = True) 
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, losses, label="Loss", color="blue", marker="o")
    plt.title(" Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"plots/{model}_loss.png", dpi=300)
    plt.close()  # Close the figure to avoid overlap

    # Plot and save Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, accuracies, label="Accuracy", color="green", marker="o")
    plt.title(" Accuracy Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"plots/{model}_accuracy.png", dpi=300)
    # plt.savefig(f"/plots/{model}_accuracy.png", dpi=300)÷
    plt.close()  # Close the figure to avoid overlap

    print(f"Plots saved as '{model}_loss.png' and '{model}_accuracy.png'")
