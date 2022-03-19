import numpy as np
import pandas as pd 
def compute_angle(results, is_degrees=True):
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
    
    df = pd.read_csv(results, delimiter=',')
    m, _ = df.shape 
    angles_est = np.zeros((m, 1))
    angles_ann = np.zeros((m, 1))
    for i in range(m):
        ########################## estimated angle ##########################
        c1, c2, c3 = df.loc[i, "Agent 0 pos x"], df.loc[i, "Agent 0 pos y"], df.loc[i, "Agent 0 pos z"]
        a1, a2, a3 = df.loc[i, "Agent 1 pos x"], df.loc[i, "Agent 1 pos y"], df.loc[i, "Agent 1 pos z"]
        r1, r2, r3 = df.loc[i, "Agent 2 pos x"], df.loc[i, "Agent 2 pos y"], df.loc[i, "Agent 2 pos z"]
        m1, m2, m3 = df.loc[i, "Agent 3 pos x"], df.loc[i, "Agent 3 pos y"], df.loc[i, "Agent 3 pos z"]
        t1, t2, t3 = df.loc[i, "Agent 4 pos x"], df.loc[i, "Agent 4 pos y"], df.loc[i, "Agent 4 pos z"]

        # chorda vector 
        v1 = np.array([c1 - a1, c2 - a2, c3 - a3])

        # fit vector for mastoid segment using SVD 
        data = np.array([[r1, r2, r3], [m1, m2, m3], [t1, t2, t3]])
        mu = np.mean(data, axis = 0)
        _, _, V = np.linalg.svd(data - mu)
        V = V.T 
        v2 = V[:, 0]

        # compute angle 
        angle = np.arccos(np.dot(v1, v2)/(np.linalg.norm(v1) * np.linalg.norm(v2)))
        if angle >= np.pi:   
            angle = np.pi - angle 
        angles_est[i] = angle 

        ########################## angle found from annotation ##########################
        c1, c2, c3 = df.loc[i, "Landmark 0 pos x"], df.loc[i, "Landmark 0 pos y"], df.loc[i, "Landmark 0 pos z"]
        a1, a2, a3 = df.loc[i, "Landmark 1 pos x"], df.loc[i, "Landmark 1 pos y"], df.loc[i, "Landmark 1 pos z"]
        r1, r2, r3 = df.loc[i, "Landmark 2 pos x"], df.loc[i, "Landmark 2 pos y"], df.loc[i, "Landmark 2 pos z"]
        m1, m2, m3 = df.loc[i, "Landmark 3 pos x"], df.loc[i, "Landmark 3 pos y"], df.loc[i, "Landmark 3 pos z"]
        t1, t2, t3 = df.loc[i, "Landmark 4 pos x"], df.loc[i, "Landmark 4 pos y"], df.loc[i, "Landmark 4 pos z"]

        # chorda vector 
        v1 = np.array([c1 - a1, c2 - a2, c3 - a3])
        # fit vector for mastoid segment using SVD 
        data = np.array([[r1, r2, r3], [m1, m2, m3], [t1, t2, t3]])
        mu = np.mean(data, axis = 0)
        _, _, V = np.linalg.svd(data - mu)
        V = V.T 
        v2 = V[:, 0]

        # compute angle 
        angle = np.arccos(np.dot(v1, v2)/(np.linalg.norm(v1) * np.linalg.norm(v2)))
        if angle >= np.pi:   
            angle = np.pi - angle 
        angles_ann[i] = angle 

    
    if is_degrees:
        angles_est *= 180 / np.pi 
        angles_ann *= 180 / np.pi 
    return angles_ann, angles_est

    
    