import numpy as np
import pandas as pd 
from .computeAngle import *

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


def estimates_error(df,number_of_agents):
    estimates_error = np.zeros((number_of_agents,1))
    standard_deviation = np.zeros((number_of_agents,1))
    m, _ = df.shape 

    for i in range(number_of_agents):
        estimates_error[i] = np.mean((df.loc[:,"Distance "+ str(i)]))
        standard_deviation[i] = np.std((df.loc[:,"Distance "+ str(i)]))

    res = np.concatenate([estimates_error,standard_deviation], axis = 1)
    return res


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
    res = estimates_error(df,number_of_agents)
    # angles_ann_index_naiv, angles_est_index_naiv = compute_angle_index_naiv(df,number_of_agents)
    # angles_ann_index_pca, angles_est_index_pca = compute_angle_index_pca(df,number_of_agents)

    # angles_ann_physical_naiv, angles_est_physical_naiv = compute_angle_physical_naiv(df, number_of_agents)
    # angles_ann_physical_pca, angles_est_physical_pca = compute_angle_physical_pca(df, number_of_agents)
    
    # angels_naiv_index = np.concatenate([angles_ann_index_naiv, angles_est_index_naiv],axis = 1)
    # angels_pca_index = np.concatenate([angles_ann_index_pca, angles_est_index_pca], axis = 1)
    # angles_naiv_physical = np.concatenate([angles_ann_physical_naiv, angles_est_physical_naiv],axis = 1)
    # angles_pca_physical = np.concatenate([angles_ann_physical_pca, angles_est_physical_pca],axis = 1)
    errors = np.concatenate([ min_dis, max_dis, rmse, res], axis = 1)

    return errors
