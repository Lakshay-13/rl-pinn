import matplotlib.pyplot as plt
import numpy as np
import torch
import os

def plot_loss(metrics_history, save_path):
    plt.figure(figsize=(10, 6))
    plotted = False
    if 'train_loss' in metrics_history and len(metrics_history['train_loss']) > 0:
        plt.plot(np.log10(metrics_history['train_loss']), label='Train Loss')
        plotted = True
    if 'valid_loss' in metrics_history and len(metrics_history['valid_loss']) > 0:
        plt.plot(np.log10(metrics_history['valid_loss']), label='Valid Loss')
        plotted = True
    
    plt.xlabel('Epochs')
    plt.ylabel('log10(Loss)')
    plt.title('Training and Validation Loss')
    if plotted:
        plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(save_path)
    plt.close()

def plot_residuals(residuals, save_path):
    # residuals shape: (7, 16, 41)
    fig, axes = plt.subplots(4, 2, figsize=(15, 20))
    axes = axes.flatten()
    
    for i in range(7):
        im = axes[i].imshow(np.abs(residuals[i]), aspect='auto', interpolation='bilinear')
        axes[i].set_title(f'Equation {i+1} Residuals')
        fig.colorbar(im, ax=axes[i])
    
    # Hide the 8th subplot
    axes[7].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_potential(phi_range, v_vals, v_theory_vals, save_path, v_best_vals=None):
    plt.figure(figsize=(10, 6))
    plt.plot(phi_range, v_theory_vals, label='Theory', color='blue', linestyle='--')
    plt.plot(phi_range, v_vals, label='NN (best=False)', color='orange')
    if v_best_vals is not None:
        plt.plot(phi_range, v_best_vals, label='NN (best=True)', color='green')
    plt.xlabel('phi')
    plt.ylabel('V(phi)')
    plt.title('Learned Potential vs Theory')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()
