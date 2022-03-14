import numpy as np
import pandas as pd 
def compute_angle(results, is_degrees=True):
    """
    Function that computes the angle between the facial nerve and chorda tympani
    The calculations are based on the landmarks: C, A, T 
    Input:
        the results.csv file from evaluation
        is_degrees: no input for result in degrees. Input for radians 
    Output:
        The angle in radians/degrees 
    """
    if is_degrees != True:
        is_degrees = False
    
    df = pd.read_csv(file, delimiter=',')
    m, _ = df.shape 
    angles_est = np.zeros((m, 1))
    angles_ann = np.zeros((m, 1))
    for i in range(m):
        # estimated angle 
        c1, c2, c3 = df.iloc[i, 2:5]
        a1, a2, a3 = df.iloc[i, 10:13]
        t1, t2, t3 = df.iloc[i, 34:37]
        v1 = np.array([c1 - a1, c2 - a2, c3 - a3])
        v2 = np.array([t1 - a1, t2 - a2, t3 - a3])
        print(v1)
        print(v2)
        angle = np.arccos(np.dot(v1, v2)/(np.linalg.norm(v1) * np.linalg.norm(v2)))
        if angle >= np.pi:   
            angle = np.pi - angle 
        angles_est[i] = angle 

        # angle found from annotation 
        c1, c2, c3 = df.iloc[i, 5:8]
        a1, a2, a3 = df.iloc[i, 13:16]
        t1, t2, t3 = df.iloc[i, 37:40]
        v1 = np.array([c1 - a1, c2 - a2, c3 - a3])
        v2 = np.array([t1 - a1, t2 - a2, t3 - a3])
        angle = np.arccos(np.dot(v1, v2)/(np.linalg.norm(v1) * np.linalg.norm(v2)))
        if angle >= np.pi:   
            angle = np.pi - angle 
        angles_ann[i] = angle 
    
    if is_degrees:
        angles_est *= 180 / np.pi 
        angles_ann *= 180 / np.pi 
    return angles_ann, angles_est

    
    