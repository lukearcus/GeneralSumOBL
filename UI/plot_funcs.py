import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d

LABEL_SETS = {
        'kuhn_policy': {'x':("bet","check"),'y': ("1 low", "2 low", "3 low", "1 high", "2 high", "3 high")},
        'kuhn_belief': {'x':("1","2","3"),'y':("1 low", "2 low", "3 low", "1 high", "2 high", "3 high")},
        }

def plot_everything(pols, bels, game, reward):
    fig = plt.figure()
    if bels is not None:
        subfigs = fig.subfigures(1,2)
        multiple_heatmaps(pols, subfigs[0], game + "_policy")
        subfigs[0].suptitle('Policies', fontsize=32)
        multiple_heatmaps(bels, subfigs[1], game+ "_belief")
        subfigs[1].suptitle('Beliefs', fontsize = 32)
        fig2 = plt.figure()
        ax = fig2.subplots()
        reward_smoothed(reward, fig2)
    else:
        multiple_heatmaps(pols, fig, game + "_policy")
        fig.suptitle('Policies', fontsize=32)

    plt.show()

def reward_smoothed(reward, ax):
    ax.plot(gaussian_filter1d(reward,1000))


def multiple_heatmaps(im_list, fig, label_name, overlay_vals=False):

    num_ims = len(im_list)
    axs = fig.subplots(num_ims,2)
    if num_ims > 1: 
        big_axes = fig.subplots(nrows=num_ims, ncols=1, sharey=True) 

        for row, big_ax in enumerate(big_axes, start=0):
            big_ax.set_title("Level %s \n" % row, fontsize=16)

            big_ax.tick_params(colors=(1.,1.,1., 0.0), top='off', bottom='off', left='off', right='off')
            big_ax._frameon = False
    
        for i, im in enumerate(im_list):
            plot_heatmap(im[0], axs[i,0], label_name, overlay_vals)
            plot_heatmap(im[1], axs[i,1], label_name, overlay_vals)
    else:
        for i, im in enumerate(im_list):
            plot_heatmap(im[0], axs[0], label_name, overlay_vals)
            plot_heatmap(im[1], axs[1], label_name, overlay_vals)

    fig.set_facecolor('w')


def plot_heatmap(im, ax, label_name, overlay_vals=False):
    ax.imshow(im, vmin=0,vmax=1)
    x_label_list = LABEL_SETS[label_name]["x"]
    y_label_list = LABEL_SETS[label_name]["y"]
    ax.set_xticks(np.arange(len(x_label_list)),labels=x_label_list)
    ax.set_yticks(np.arange(len(y_label_list)),labels=y_label_list)
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
    

