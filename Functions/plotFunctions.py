from .metrics import *
import numpy as np
import color_dtu_design as colorl
import matplotlib.pyplot as plt

def plot_erros(title, errors, number_of_agents,legend, marker,col):
    x = np.arange(1,7)
    m = number_of_agents.shape[0]
    for j in range(4):
        plt.figure(j)
        for i in range(m):
            if number_of_agents[i] == 12:
                temp1 = errors[i].T[j,[0,2,4,6,8,10]]
                temp2 = errors[i].T[j,[1,3,5,7,9,11]]
                plt.plot(x,temp1, marker = marker[i],color=col[i], linestyle = 'None')
                plt.plot(x,temp2, marker = marker[i],color=col[i], linestyle = 'None')
            else:
                temp = errors[i][:].T[j]
                plt.plot(x,temp, marker=marker[i],color=col[i], linestyle = 'None')
        labels = ['C', 'A', 'R', 'M', 'T', 'B']

        plt.xticks(x, labels)
        plt.legend(legend)
        plt.ylabel('[mm]')
        plt.xlabel('Agents for each landmark')
        plt.title(title[j])
        plt.show()


def diffangels(angles_naiv_index, angles_PCA_index, angles_naiv_physical, angles_PCA_physical):
    diff_naiv_index = np.reshape(angles_naiv_index[:, 0] - angles_naiv_index[:, 1],(13,1))
    diff_PCA_index = np.reshape(angles_PCA_index[:, 0] - angles_PCA_index[:, 1],(13,1))
    diff_naiv_physical = np.reshape(angles_naiv_physical[:, 0] - angles_naiv_physical[:, 1],(13,1))
    diff_PCA_physical = np.reshape(angles_PCA_physical[:, 0] - angles_PCA_physical[:, 1],(13,1))

    angels = np.abs(np.concatenate([diff_naiv_index, diff_PCA_index,diff_naiv_physical,diff_PCA_physical],axis = 1))
    return angels

def plotdifang(title, angels, number_of_agents,legend, marker,col):
    x = np.arange(1,14)
    m = len(number_of_agents)
    for j in range(4):
        plt.figure(j+4)
        for i in range(m):
            temp = angels[i][:].T[j]
            plt.plot(x,temp, marker=marker[i],color=col[i], linestyle = 'None')
        plt.legend(legend)
        plt.ylabel('[mm]')
        plt.xlabel('Test Image')
        plt.title(title[j])
        plt.show()


def plot_all_plots(title_erros, title_angels, legend, marker, models, number_of_agents,col=None):
    m = len(models)
    errors = [[0] for i in number_of_agents]
    angels = [[0] for i in range(len(models))]
    for i in range(m):
        errors[i],naiv_i,pca_i,naiv_p,pca_p = performence_metric(models[i],number_of_agents[i])
        angels[i] = diffangels(naiv_i,pca_i,naiv_p,pca_p)
    if col==None:
        te = [[0] for i in number_of_agents]
        for i in range(m):
            te[i] = colorl.color_design.color_design(i+1).color
        col = te
    plot_erros(title_erros, errors, number_of_agents, legend, marker,col)
    plotdifang(title_angels, angels, number_of_agents, legend, marker,col)