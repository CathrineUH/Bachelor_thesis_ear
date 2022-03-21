import numpy as np
import pandas as pd 

def RMSE(df, number_of_landmarks):
    rmse = np.zeros((number_of_landmarks,1))
    m, _ = df.shape 
    for i in range(number_of_landmarks):
        temp_sum = 0
        for j in range(m):
             temp_sum += ((df.loc[j,"Distance "+ str(i)])**2)
        rmse[i] = np.sqrt(temp_sum / m)
    return rmse

def compute_angle(df, is_degrees=True):
    """
    Function that computes the angle between the facial nerve and chorda tympani
    The calculations are based on the landmarks: C, A, T 
    Input:
        results: the path to the results.csv file from evaluation
        is_degrees: no input for result in degrees. Input for radians 
    Output:
        The angle in radians/degrees 
    """
    if is_degrees != True:
        is_degrees = is_degrees
    
    m, _ = df.shape 
    angles_est = np.zeros((m, 1))
    angles_ann = np.zeros((m, 1))
    for i in range(m):
        ########################## estimated angle ##########################
        angles_est[i] = get_angle(i, df, "Agent")

        ########################## angle found from annotation ##########################
        angles_ann[i] = get_angle(i, df, "Landmark")

    
    if is_degrees:
        angles_est *= 180 / np.pi 
        angles_ann *= 180 / np.pi 
    return angles_ann, angles_est

def get_angle(i, df, which):
    c1, c2, c3 = df.loc[i, which + " 0 pos x"], df.loc[i, which + " 0 pos y"], df.loc[i, which + " 0 pos z"]
    a1, a2, a3 = df.loc[i, which + " 1 pos x"], df.loc[i, which + " 1 pos y"], df.loc[i, which + " 1 pos z"]
    r1, r2, r3 = df.loc[i, which + " 2 pos x"], df.loc[i, which + " 2 pos y"], df.loc[i, which + " 2 pos z"]
    m1, m2, m3 = df.loc[i, which + " 3 pos x"], df.loc[i, which + " 3 pos y"], df.loc[i, which + " 3 pos z"]
    t1, t2, t3 = df.loc[i, which + " 4 pos x"], df.loc[i, which + " 4 pos y"], df.loc[i, which + " 4 pos z"]

    # chorda vector 
    v1 = np.array([c1 - a1, c2 - a2, c3 - a3])

    # fit vector for mastoid segment using SVD 
    data = np.array([[r1, r2, r3], [m1, m2, m3], [t1, t2, t3]])
    mu = np.mean(data, axis = 0)
    _, _, V = np.linalg.svd(data - mu)
    V = V.T 
    v2 = V[:, 0]

    # check for opposite signs 
    if np.sign(v1[2]) != np.sign(v2[2]):
        v1 = -v1 

    # compute angle 
    angle = np.arccos(np.dot(v1, v2)/(np.linalg.norm(v1) * np.linalg.norm(v2)))
    if angle >= np.pi:   
        angle = np.pi - angle 
    return angle 
    

def max_distance(df, number_of_landmarks):
    max_dis = np.zeros((number_of_landmarks,1))
    for i in range(number_of_landmarks):
      max_dis[i] = max(df.loc[:,"Distance "+ str(i)])
    return max_dis

def min_distance(df, number_of_landmarks):
    min_dis = np.zeros((number_of_landmarks,1))
    for i in range(number_of_landmarks):
      min_dis[i] = min(df.loc[:,"Distance "+ str(i)])
    return min_dis

def performence_metric(results,number_of_landmarks):
    df = pd.read_csv(results, delimiter=',')
    min_dis = min_distance(df,number_of_landmarks)
    max_dis = max_distance(df,number_of_landmarks)
    rmse = RMSE(df, number_of_landmarks)
    angles_ann, angles_est = compute_angle(df)
    return min_dis,max_dis,rmse, angles_ann,angles_est
