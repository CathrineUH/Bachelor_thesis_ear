import numpy as np
import pandas as pd 
from computeAngle import *

def RMSE(df, number_of_agents):
    """
    Function that computes the Root mean square error.
    Input:
        df: dataframe of the results
        number_of_agents: the number of agents
    Output:8
        The root mean sqaure error (mm)
    """
    rmse = np.zeros((number_of_agents,1))
    m, _ = df.shape 
    for i in range(number_of_agents):
        temp_sum = 0
        for j in range(m):
             temp_sum += ((df.loc[j,"Distance "+ str(i)])**2)
        rmse[i] = np.sqrt(temp_sum / m)
    return rmse

def max_distance(df, number_of_agents):
    """
    Function that computes the maximum distance of each agent
    Input:
        df: dataframe of the results
        number_of_agents: the number of agents
    Output:8
        The maximum distance of each agent (mm)
    """
    max_dis = np.zeros((number_of_agents,1))
    for i in range(number_of_agents):
      max_dis[i] = max(df.loc[:,"Distance "+ str(i)])
    return max_dis

def min_distance(df, number_of_agents):
    """
    Function that computes the minimum distance of each agent
    Input:
        df: dataframe of the results
        number_of_agents: the number of agents
    Output:8
        The minimum distance of each agent (mm)
    """
    min_dis = np.zeros((number_of_agents,1))
    for i in range(number_of_agents):
      min_dis[i] = min(df.loc[:,"Distance "+ str(i)])
    return min_dis

def performence_metric(results,number_of_agents):
    """
    Function that computes the performence metrics and the angels
    Input:
        results: the path to the results.csv file from evaluation
        number_of_agents: the number of agents
    Output:8
        The performence metrics(mm) and the angels in radians/degrees
    """
    df = pd.read_csv(results, delimiter=',')
    min_dis = min_distance(df,number_of_agents)
    max_dis = max_distance(df,number_of_agents)
    rmse = RMSE(df, number_of_agents)
    angles_ann_naiv, angles_est_naiv = compute_angle_naiv(df,number_of_agents)
    angles_ann_pca, angles_est_pca = compute_angle_pca(df,number_of_agents)
    
    angels_naiv = np.concatenate([angles_ann_naiv, angles_est_naiv],axis = 1)
    angels_pca = np.concatenate([angles_ann_pca, angles_est_pca], axis = 1)
    errors = np.concatenate([ min_dis, max_dis, rmse], axis = 1)

    return errors, angels_naiv, angels_pca
