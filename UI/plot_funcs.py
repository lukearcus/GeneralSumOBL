import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
from matplotlib import cm
from matplotlib.colors import Normalize

LABEL_SETS = {
        'kuhn_policy': {'x_ticks':("bet","check"),'y_ticks': ("1, no bets", "2, no bets", "3, no bets", "1, bet", "2, bet", "3, bet"),'x_label':"action",'y_label':"state"},
        'kuhn_belief': {'x_ticks':("1","2","3"),'y_ticks':("1, no bets", "2, no bets", "3, no bets", "1, bet", "2, bet", "3, bet"),'x_label':"opponent card",'y_label':"state"},
        }

def plot_everything(pols, bels, game, reward, exploitability):
    fig = plt.figure(figsize=[16, 12])
    if bels is not None:
        subfigs = fig.subfigures(1,3, width_ratios=[3,3,1])
        multiple_heatmaps(pols, subfigs[0], game + "_policy", labels=True)
        subfigs[0].suptitle('Policies', fontsize=32)
        multiple_heatmaps(bels, subfigs[1], game+ "_belief")
        subfigs[1].suptitle('Beliefs', fontsize = 32)
        cbar_ax = subfigs[2].add_axes([0.15, 0.15, 0.05, 0.7])
        subfigs[2].colorbar(cm.ScalarMappable(norm=Normalize(vmin=0,vmax=1)), cax=cbar_ax)
        #fig.subplots_adjust(right=0.8)
        #cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        #fig.colorbar(cm.ScalarMappable(norm=Normalize(vmin=0,vmax=1)), cax=cbar_ax)
        fig2 = plt.figure()
        ax = fig2.subplots()
        reward_smoothed(reward, ax)
        fig3 = plt.figure()
        ax = fig3.subplots()
        ax.plot(exploitability)
    else:
        multiple_heatmaps(pols, fig, game + "_policy")
        fig.suptitle('Policies', fontsize=32)

    plt.show()

def reward_smoothed(reward, ax):
    ax.plot(gaussian_filter1d(reward,1000))


def multiple_heatmaps(im_list, fig, label_name, labels = False, overlay_vals=False, max_ims=10):
    
    if len(im_list) > max_ims:
        im_list = im_list[-max_ims:]

    num_ims = len(im_list)
    axs = fig.subplots(num_ims,2)
    if num_ims > 1:
        if labels:
            big_axes = fig.subplots(nrows=num_ims, ncols=1, sharey=True) 

            for row, big_ax in enumerate(big_axes, start=0):
                big_ax.set_ylabel("Level %s" % row, fontsize=16)
                big_ax.set_facecolor('0.85')

                big_ax.tick_params(colors=(1.,1.,1., 0.0), top='off', bottom='off', left='off', right='off')
                big_ax._frameon = False
    
        plt.subplots_adjust( 
                    hspace=0.4)
        for i, im in enumerate(im_list):
            plot_heatmap(im[0], axs[i,0], label_name, overlay_vals)
            plot_heatmap(im[1], axs[i,1], label_name, overlay_vals)
        axs[0, 0].set_title("agent 1")
        axs[0, 1].set_title("agent 2")
    else:
        for i, im in enumerate(im_list):
            plot_heatmap(im[0], axs[0], label_name, overlay_vals)
            plot_heatmap(im[1], axs[1], label_name, overlay_vals)
        axs[0].set_title("agent 1")
        axs[1].set_title("agent 2")

    fig.set_facecolor('w')


def plot_heatmap(im, ax, label_name, overlay_vals=False):
    ax.imshow(im, vmin=0,vmax=1)
    x_label_list = LABEL_SETS[label_name]["x_ticks"]
    y_label_list = LABEL_SETS[label_name]["y_ticks"]
    ax.set_xticks(np.arange(len(x_label_list)),labels=x_label_list)
    ax.set_yticks(np.arange(len(y_label_list)),labels=y_label_list)
    ax.set_xlabel(LABEL_SETS[label_name]["x_label"])
    ax.set_ylabel(LABEL_SETS[label_name]["y_label"])
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    if overlay_vals:
        for (j,i),label in np.ndenumerate(im):
            ax.text(i,j,np.round(label,2),ha='center',va='center')

def exploitability_plot(exploitability, exploit_freq):
    plt.plot(np.arange(0, len(exploitability)*exploit_freq, exploit_freq), exploitability)
    plt.xlabel("Iteration")
    plt.ylabel("Exploitability")
    plt.title("Exploitability vs Iteration")

def FSP_plots(exploitability, exploit_freq, pols, game):
    exploitability_plot(exploitability, exploit_freq)
    fig = plt.figure()
    multiple_heatmaps(pols, fig, game+'_policy', True)
    plt.show()
    

