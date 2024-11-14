# -*- coding: utf-8 -*-
"""
Visualization utility - AdaBoost-FKD
============

This file contains some utilities for results visualization

Authors:
    Mario Chacon-Falcon <mario.chacon.falcon@gmail.com>
    Jose M. Moyano <jmoyano1@us.es>

Created on:
    2024-06-23

Version:
    1.0.1

License:
    GNU General Public License v3
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def difference_errorbar_plot(score_frames, labels, n_clients):
    """
    Plots an error bar plot representing the average absolute improvement of each client's FL model
    against the local model, both on global and local data, for every parameter considered.

    Args:
        score_frames (list): list of lists of pandas dataframes. Each list of dataframes stores the scores 
        of every clients' models (local and FL) on both local and global data of a FL model. The number of 
        dataframes correspond to the number of data splits used. Note that it is thought to be used with 
        the output accDataFrames, from the function evaluation.fl_cross_val_score().
        labels (list): list of strings to use as labels in the error bar plot for all clients.
        n_clients (int): Number of clients used in the training process.
    """
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    plt.setp((ax1, ax2), xticks=np.arange(0, 10 * n_clients, 10), xticklabels=np.arange(0, n_clients))
    ax1.set_title('Avg local_difference of each client')
    ax2.set_title('Avg global_difference of each client')
    ax1.set_xlabel('clients')
    ax1.set_ylabel('(FL_local-own_local)')
    ax2.set_xlabel('clients')
    ax2.set_ylabel('(FL_global-own_global)')
    markers = ['.', '+', 'v', 's', 'd']
    x = np.linspace(-4.2, 4.2, len(score_frames)) + 0.5
    x_data = np.zeros(n_clients)
    avg_lc_data = np.zeros(n_clients)
    std_lc_data = np.zeros(n_clients)
    avg_gc_data = np.zeros(n_clients)
    std_gc_data = np.zeros(n_clients)

    for i, data in enumerate(score_frames):
        dataframe = pd.concat(data, axis=0)
        for j in range(n_clients):
            client_data = dataframe[dataframe.index == j].describe()
            x_data[j] = x[i] + 10 * j
            avg_lc_data[j] = client_data.loc['mean']['local_difference']
            std_lc_data[j] = client_data.loc['std']['local_difference']
            avg_gc_data[j] = client_data.loc['mean']['global_difference']
            std_gc_data[j] = client_data.loc['std']['global_difference']
            ax1.axvline(10 * j - 5, color='lightgrey', linestyle='dotted')
            ax2.axvline(10 * j - 5, color='lightgrey', linestyle='dotted')
        ax1.errorbar(x_data, avg_lc_data, yerr=std_lc_data, fmt=markers[i], label=labels[i], linewidth=1, capsize=3,
                     ecolor='darkgray')
        ax2.errorbar(x_data, avg_gc_data, yerr=std_gc_data, fmt=markers[i], label=labels[i], linewidth=1, capsize=3,
                     ecolor='darkgray')

    ax1.legend()
    ax2.legend()


def splits_scores_bar_plot(describe_frames, avg_frames, centralized_scores, labels):
    """
    Bar plot that represents the average result (+-std) over all clients of every parameter for every data split. It 
    also plots the average result across all clients and all datas splits. 

    Args:
        describe_frames (list): list of lists of dataframes. Each list of dataframes stores the stats of the scores 
        of every clients' models (local and FL) on both local and global data of a FL model. The number of 
        dataframes correspond to the number of data splits used. Note that it is thought to be used with 
        the output describeDataFrames, from the function evaluation.fl_cross_val_score().
        avg_frames (list): list of dataframes. The datafaframe in position j stores the stats of the 
        average scores of all clients in all splits for the parameter labels[j]. It is thought to be used 
        with the output avg_result, from the function evaluation.fl_cross_val_score().
        centralized_scores (list): list centralized model scores throughout data splits.
        labels (list): list of strings to use as labels in the bar plot for all data splits.
    """
    fig, axs = plt.subplots(3, 2, figsize=(12, 21))
    axs = axs.flatten()

    #xtick_shift = (len(labels)-1)/2

    x = np.arange(len(labels))  # the label locations
    width = 0.10  # the width of the bars
    multiplier = 0
    names = ['FL_own_score', 'FL_gl_score', 'lc_own_score', 'lc_gl_score']
    for i in range(len(describe_frames[0])):
        multiplier = 0
        df = pd.Series.to_frame(describe_frames[0][i].loc['mean'][
                                           ['FL_acc_own_data', 'FL_acc_global_data', 'local_acc_own_data',
                                            'local_acc_global_data']])
        std_df = pd.Series.to_frame(describe_frames[0][i].loc['std'][
                                              ['FL_acc_own_data', 'FL_acc_global_data', 'local_acc_own_data',
                                               'local_acc_global_data']])
        for listOfFrames in describe_frames[1:]:
            series = listOfFrames[i].loc['mean'][
                ['FL_acc_own_data', 'FL_acc_global_data', 'local_acc_own_data', 'local_acc_global_data']]
            std_series = listOfFrames[i].loc['std'][
                ['FL_acc_own_data', 'FL_acc_global_data', 'local_acc_own_data', 'local_acc_global_data']]
            df = pd.concat((df, series), axis=1)
            std_df = pd.concat((std_df, std_series), axis=1)
        df = df.T.reset_index(drop=True)
        std_df = std_df.T.reset_index(drop=True)

        for j, column in enumerate(df.columns):
            offset = width * multiplier
            z = x + offset
            rects = axs[i].bar(z, df[column].to_numpy(), width, label=names[j])
            rects = axs[i].errorbar(z, df[column].to_numpy(), fmt='.', yerr=std_df[column].to_numpy(),
                                    linewidth=1, capsize=1, ecolor='k')
            #ax.bar_label(rects, padding=3)
            multiplier += 1
            offset = width * multiplier
        rects = axs[i].bar(x + offset, centralized_scores[i], width, label='centr_score')
        # Add some text for labels, title and custom x-axis tick labels, etc.
        axs[i].set_ylabel('Score')
        axs[i].set_title(f'split {i}')

        axs[i].set_xticks(x + 2 * width, labels)
        axs[i].legend(loc='upper left')
        axs[i].set_ylim(0, 1.5)

    multiplier = 0
    df = pd.Series.to_frame(avg_frames[0].loc['mean'][
                                       ['FL_acc_own_data', 'FL_acc_global_data', 'local_acc_own_data',
                                        'local_acc_global_data']])
    std_df = pd.Series.to_frame(avg_frames[0].loc['std'][
                                          ['FL_acc_own_data', 'FL_acc_global_data', 'local_acc_own_data',
                                           'local_acc_global_data']])
    for listOfFrames in avg_frames[1:]:
        series = listOfFrames.loc['mean'][
            ['FL_acc_own_data', 'FL_acc_global_data', 'local_acc_own_data', 'local_acc_global_data']]
        std_series = listOfFrames.loc['std'][
            ['FL_acc_own_data', 'FL_acc_global_data', 'local_acc_own_data', 'local_acc_global_data']]
        df = pd.concat((df, series), axis=1)
    df = df.T.reset_index(drop=True)
    std_df = std_df.T.reset_index(drop=True)

    for j, column in enumerate(std_df.columns):
        offset = width * multiplier
        z = x + offset
        rects = axs[-1].bar(z, df[column].to_numpy(), width, label=names[j])
        rects = axs[-1].errorbar(z, df[column].to_numpy(), fmt='.', yerr=std_df[column].to_numpy(),
                                 linewidth=1, capsize=1, ecolor='k')
        #ax.bar_label(rects, padding=3)
        multiplier += 1
        offset = width * multiplier
    rects = axs[-1].bar(x + offset, np.array(centralized_scores).mean(), width, label='avg_centr_score')
    rects = axs[-1].errorbar(x + offset, np.repeat(np.array(centralized_scores).mean(), len(x + offset)), fmt='.',
                             yerr=np.repeat(np.array(centralized_scores).std(), len(x + offset)), linewidth=1,
                             capsize=1, ecolor='k')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    axs[-1].set_ylabel('Score')
    axs[-1].set_title('Avg scores in splits')
    axs[-1].set_xticks(x + 2 * width, labels)
    axs[-1].legend(loc='upper left')
    axs[-1].set_ylim(0, 1.5)


def parameter_global_score_overview(splits, params):
    """
    Figure with number_of_splits subplots, representing pairs (x,y), where x are the values of params tested and 
    y are the performance of the algorithm for a given metric (between 0 and 1). Every subplot plots the average 
    performance of the FL model throughout the values of the given parameter and through data splits on local and global data.

    Args:
        splits (dict): dictionary with keys 0,...,number_of_splits and values lists of dataframes, where the i-th pair
        corresponds to a list of describeDataframes where position j of the list is the result of an FLmodel trained with split i
        dataset and parameter params[j].
        params (np.array): 1-d numpy array with the values of the parameter to be tested.
    """
    fig, axs = plt.subplots(3, 2, figsize=(12, 21))
    axs = axs.flatten()
    labels = ['FL_acc_global_data', 'local_acc_global_data']

    for i in splits:
        split0 = splits[i]
        merged_data = pd.concat(split0, axis=0)
        avg_stats = merged_data[merged_data.index == 'mean'][['FL_acc_global_data', 'local_acc_global_data']]
        for label in labels:
            acc = avg_stats[label].to_numpy()
            axs[i].plot(params, acc, linewidth=2, label=label)

        axs[i].set_ylabel('Score')
        axs[i].set_title(f'split {i}')
        axs[i].legend(loc='upper left')
        axs[i].set_ylim(0, 1.5)
    plt.show()
