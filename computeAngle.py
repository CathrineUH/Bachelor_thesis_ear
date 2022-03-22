import numpy as np
import pandas as pd 
import json 
import SimpleITK as sitk 
# def compute_angle_pca(results, is_degrees=True):
#     """
#     Function that computes the angle between the facial nerve and chorda tympani
#     The calculations are based on the landmarks: C, A, and R, M, T  
#     Input:
#         results: the path to the results.csv file from evaluation
#         is_degrees: no input for result in degrees. Input for radians 
#     Output:
#         The angle in radians/degrees 
#     """
#     if is_degrees != True:
#         is_degrees = is_degrees
    
#     df = pd.read_csv(results, delimiter=',')
#     m, _ = df.shape 
#     angles_est = np.zeros((m, 1))
#     angles_ann = np.zeros((m, 1))
#     for i in range(m):
#         ########################## estimated angle ##########################
#         angles_est[i] = get_angle_pca(i, df, "Agent")

#         ########################## angle found from annotation ##########################
#         angles_ann[i] = get_angle_pca(i, df, "Landmark")

    
#     if is_degrees:
#         angles_est *= 180 / np.pi 
#         angles_ann *= 180 / np.pi 
#     return angles_ann, angles_est

# def get_angle_pca(i, df, which):
#     c1, c2, c3 = df.loc[i, which + " 0 pos x"], df.loc[i, which + " 0 pos y"], df.loc[i, which + " 0 pos z"]
#     a1, a2, a3 = df.loc[i, which + " 1 pos x"], df.loc[i, which + " 1 pos y"], df.loc[i, which + " 1 pos z"]
#     r1, r2, r3 = df.loc[i, which + " 2 pos x"], df.loc[i, which + " 2 pos y"], df.loc[i, which + " 2 pos z"]
#     m1, m2, m3 = df.loc[i, which + " 3 pos x"], df.loc[i, which + " 3 pos y"], df.loc[i, which + " 3 pos z"]
#     t1, t2, t3 = df.loc[i, which + " 4 pos x"], df.loc[i, which + " 4 pos y"], df.loc[i, which + " 4 pos z"]

#     # chorda vector 
#     v1 = np.array([c1 - a1, c2 - a2, c3 - a3])

#     # fit vector for mastoid segment using SVD 
#     data = np.array([[r1, r2, r3], [m1, m2, m3], [t1, t2, t3]])
#     mu = np.mean(data, axis = 0)
#     _, _, V = np.linalg.svd(data - mu)
#     V = V.T 
#     v2 = V[:, 0]

#     # check for opposite signs 
#     if np.sign(v1[2]) != np.sign(v2[2]):
#         v1 = -v1 

#     # compute angle 
#     angle = np.arccos(np.dot(v1, v2)/(np.linalg.norm(v1) * np.linalg.norm(v2)))
#     if angle >= np.pi:   
#         angle = np.pi - angle 
#     return angle 
    

# def compute_angle_naiv(results, is_degrees = True):
#     """
#     Function that computes the angle between the facial nerve and chorda tympani
#     The calculations are based on the landmarks: C, A, T 
#     Input:
#         results: the path to the results.csv file from evaluation
#         is_degrees: no input for result in degrees. Input for radians 
#     Output:
#         The angle in radians/degrees 
#     """
#     if is_degrees != True:
#         is_degrees = is_degrees
    
#     df = pd.read_csv(results, delimiter=',')
#     m, _ = df.shape 
#     angles_est = np.zeros((m, 1))
#     angles_ann = np.zeros((m, 1))

#     for i in range(m):
#         ########################## estimated angle ##########################
#         angles_est[i] = get_angle_naiv(i, df, "Agent")

#         ########################## angle found from annotation ##########################
#         angles_ann[i] = get_angle_naiv(i, df, "Landmark")
    
#     if is_degrees:
#         angles_est *= 180 / np.pi 
#         angles_ann *= 180 / np.pi 
#     return angles_ann, angles_est
        



# def get_angle_naiv(i, df, which):
#     c1, c2, c3 = df.loc[i, which + " 0 pos x"], df.loc[i, which + " 0 pos y"], df.loc[i, which + " 0 pos z"]
#     a1, a2, a3 = df.loc[i, which + " 1 pos x"], df.loc[i, which + " 1 pos y"], df.loc[i, which + " 1 pos z"]
#     t1, t2, t3 = df.loc[i, which + " 4 pos x"], df.loc[i, which + " 4 pos y"], df.loc[i, which + " 4 pos z"]

#     # find vectors 
#     v1 = np.array([c1 - a1, c2 - a2, c3 - a3])
#     v2 = np.array([t1 - a1, t2 - a2, t3 - a3])

#     # compute angle 
#     angle = np.arccos(np.dot(v1, v2)/(np.linalg.norm(v1) * np.linalg.norm(v2)))
#     if angle >= np.pi:   
#         angle = np.pi - angle 
#     return angle 

def read_json_files():
    """
    A function that reads the json files of the testing images
    and returns the physical coordinates in a (6, 3, #test) size array 
    """

    # get filenames for testing files 
    path_test = "Cmarl/src/data/filenames/filenames_map/testing.txt"
    file = open(path_test)
    paths = file.read()
    paths = paths.split("\n")
    file.close()
    m = np.size(paths)

    # get jason files 
    path_json = "json_Files/json_100"
    physical_points = np.zeros((6, 3, m))

    for i in range(m):
        name = paths[i][34:46] 
        json_file = open(path_json + "/" + name + ".json")
        data = json.load(json_file)
        t = data['markups'][0]['controlPoints']
        Dict = {'C': 0, 'A': 1, 'R': 2, 'M': 3, 'T': 4, 'B': 5}
        Coordinat = np.zeros((6,3))
        for lm in range(6):
            Coordinat[Dict[t[lm]['label']]] = t[lm]['position']
        json_file.close()

        physical_points[:, :, i] = Coordinat
    
    return physical_points





def get_physical_point_from_result(df, number_of_agents):
    """
    A function that computes the physical coordinates from agents index position
    Input:
        results: Path to csv file with results
        number_of_agents: the number of agents used to train (6 or 12)
    Output:
        A (6, 3, #test_im) array with physical coordinates 
    """
 
    m, _ = df.shape
    
    path_image = "Cmarl/src/data/images/image_map"
    physical_points = np.zeros((6, 3, m))

    for i in range(m):
        name = df.loc[i, "Filename 0"]
        im = sitk.ReadImage(path_image + "/" + name + ".nii.gz")

        if number_of_agents == 12:
            C = np.min([df.loc[i, "Distance 0"], df.loc[i, "Distance 1"]])
            if [df.loc[i, "Distance 0"] == df.loc[i, "Distance 1"]]:
                Cidx = 0
            else: 
                Cidx = int(np.where(C == [df.loc[i, "Distance 0"], df.loc[i, "Distance 1"]])[0])

            A = np.min([df.loc[i, "Distance 2"], df.loc[i, "Distance 3"]])
            if df.loc[i, "Distance 2"] == df.loc[i, "Distance 3"]
                Aidx = 0
            else:
                Aidx = int(np.where(A == [df.loc[i, "Distance 2"], df.loc[i, "Distance 3"]])[0]) + 2

            R = np.min([df.loc[i, "Distance 4"], df.loc[i, "Distance 5"]])
            if df.loc[i, "Distance 4"] == df.loc[i, "Distance 5"]:
                Ridx = 0
            else: 
                Ridx = int(np.where(R == [df.loc[i, "Distance 4"], df.loc[i, "Distance 5"]])[0]) + 4

            M = np.min([df.loc[i, "Distance 6"], df.loc[i, "Distance 7"]])
            if df.loc[i, "Distance 6"] == df.loc[i, "Distance 7"]:
                Midx = 0
            else: 
                Midx = int(np.where(M == [df.loc[i, "Distance 6"], df.loc[i, "Distance 7"]])[0]) + 6

            T = np.min([df.loc[i, "Distance 8"], df.loc[i, "Distance 9"]])
            if df.loc[i, "Distance 8"] ==  df.loc[i, "Distance 9"]:
                Tidx = 0
            else: 
                Tidx = int(np.where(T == [df.loc[i, "Distance 8"], df.loc[i, "Distance 9"]])[0]) + 8 

            B = np.min([df.loc[i, "Distance 10"], df.loc[i, "Distance 11"]])
            if df.loc[i, "Distance 10"] ==  df.loc[i, "Distance 11"]
            else: 
                Bidx = int(np.where(B == [df.loc[i, "Distance 10"], df.loc[i, "Distance 11"]])[0]) + 10 

            agent_nr = [Cidx, Aidx, Ridx, Midx, Tidx, Bidx]
        else: 
            agent_nr = [0, 1, 2, 3, 4, 5]
        
        for j in range(6):
            x = df.loc[i, "Agent " + str(agent_nr[j]) + " pos x"]
            y = df.loc[i, "Agent " + str(agent_nr[j]) + " pos y"]
            z = df.loc[i, "Agent " + str(agent_nr[j]) + " pos z"]
            physical_points[j, :, i] = im.TransformIndexToPhysicalPoint([int(x), int(y), int(z)])

    return  physical_points




def compute_angle_naiv(df, number_of_agents, is_degrees=True):

    if is_degrees != True:
        is_degrees = is_degrees
    
    coor_anno = read_json_files()
    coor_esto = get_physical_point_from_result(df, number_of_agents)
    
    if coor_anno.shape[2] != coor_esto.shape[2]:
        print("Number of test files do not match")
        return 
    else: 
        m = coor_anno.shape[2]

    angles_ann = np.zeros((m, 1))
    angles_est = np.zeros((m, 1))

    for i in range(m):
        chorda_anno = coor_anno[0, :, i] - coor_anno[1, :, i]
        chorda_esto = coor_esto[0, :, i] - coor_esto[1, :, i]
        
        facial_anno = coor_anno[4, :, i] - coor_anno[1, :, i]
        facial_esto = coor_esto[4, :, i] - coor_esto[1, :, i]

        angle_anno = np.arccos(np.dot(chorda_anno, facial_anno)/(np.linalg.norm(chorda_anno) * np.linalg.norm(facial_anno)))
        angle_esto = np.arccos(np.dot(chorda_esto, facial_esto)/(np.linalg.norm(chorda_esto) * np.linalg.norm(facial_esto)))

        if angle_anno > np.pi:
            angle_anno = np.pi - angle_anno
        if angle_esto > np.pi:
            angle_esto = np.pi - angle_esto
        
        angles_ann[i] = angle_anno
        angles_est[i] = angle_esto 
    
    if is_degrees:
        angles_est *= 180 / np.pi 
        angles_ann *= 180 / np.pi 

    return angles_ann, angles_est

def get_pca_direction(R, M, T):
    """
    Computes the first principal component from the three landmarks 
    """
    data = np.array([R, M, T])
    mu = np.mean(data, axis = 0)
    _, _, V = np.linalg.svd(data - mu)
    V = V.T 
    direction = V[:, 0]
    return direction 


def compute_angle_pca(df, number_of_agents, is_degrees=True):
    if is_degrees != True:
        is_degrees = is_degrees
    
    coor_anno = read_json_files()
    coor_esto = get_physical_point_from_result(df, number_of_agents)
    
    if coor_anno.shape[2] != coor_esto.shape[2]:
        print("Number of test files do not match")
        return 
    else: 
        m = coor_anno.shape[2]

    angles_ann = np.zeros((m, 1))
    angles_est = np.zeros((m, 1))

    for i in range(m):
        # C A R M T B 
        chorda_anno = coor_anno[0, :, i] - coor_anno[1, :, i]
        chorda_esto = coor_esto[0, :, i] - coor_esto[1, :, i]
        
        facial_anno = get_pca_direction(coor_anno[2, :, i], coor_anno[3, :, i], coor_anno[4, :, i])
        facial_esto = get_pca_direction(coor_esto[2, :, i], coor_esto[3, :, i], coor_esto[4, :, i])

        if np.sign(facial_anno[2]) != np.sign(chorda_anno[2]):
            facial_anno = -facial_anno

        if np.sign(facial_esto[2]) != np.sign(chorda_esto[2]):
            facial_esto = -facial_esto

        angle_anno = np.arccos(np.dot(chorda_anno, facial_anno)/(np.linalg.norm(chorda_anno) * np.linalg.norm(facial_anno)))
        angle_esto = np.arccos(np.dot(chorda_esto, facial_esto)/(np.linalg.norm(chorda_esto) * np.linalg.norm(facial_esto)))

        if angle_anno > np.pi:
            angle_anno = np.pi - angle_anno
        if angle_esto > np.pi:
            angle_esto = np.pi - angle_esto
        
        angles_ann[i] = angle_anno
        angles_est[i] = angle_esto 
    
    if is_degrees:
        angles_est *= 180 / np.pi 
        angles_ann *= 180 / np.pi 

    return angles_ann, angles_est


