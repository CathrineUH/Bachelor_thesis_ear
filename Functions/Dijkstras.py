import dijkstra3d
import numpy as np
import os
import pandas as pd
import SimpleITK as sitk
from skimage import data
import Functions as F 
import matplotlib.pyplot as plt 


def read_from_df(df, idx, nr, which): 
    position = np.array([df.loc[nr, which + " " + str(idx) + " pos x"], df.loc[nr, which + " " + str(idx) + " pos y"], df.loc[nr, which + " " + str(idx) + " pos z"],])
    return position


def run_dijkstra(results, nr_agents, which, con_chorda, con_facial):
    ind = 0
    data = pd.read_csv(results)
    weights = np.zeros((data.shape[0], 2))
    idx = F.get_best_agents(data) if nr_agents == 12 else [0, 1, 2, 3, 4, 5]
    with open(r"Cmarl\src\data\filenames\testing.txt", 'r') as file: 
        filenames = file.read().split("\n")


    for nr in range(data.shape[0]):
        if nr >= 9: 
            id = 41
        else: 
            id = 36
        print(f"nr = {nr}")
        C = read_from_df(data, idx[0], nr, which)
        A = read_from_df(data, idx[1], nr, which)
        R = read_from_df(data, idx[2], nr, which)
        M = read_from_df(data, idx[3], nr, which)
        T = read_from_df(data, idx[4], nr, which)

        #load image
        mr_scan = sitk.GetArrayFromImage(sitk.ReadImage(filenames[nr]))

        #positive (but mr is always positive)
        field = mr_scan
                
        # path is an [N,3] numpy array i.e. a list of x,y,z coordinates
        # terminates early, default is 26 connected
        # conectivity 18 2 landmarks
        path_chorda = dijkstra3d.dijkstra(field, A, C, connectivity=con_chorda, bidirectional=True)
        
        label = mr_scan.copy()*0
        for (x,y,z) in path_chorda:
            label[x,y,z] = 1
        sitk.WriteImage(sitk.GetImageFromArray(label), os.path.join('paths/paths_chorda',filenames[nr][id:])  , useCompression=True)
        np.savetxt("paths/chordatxt/" + filenames[nr][id:-6] + "txt", path_chorda, fmt='%s')
        
        
        #here we use 3 landmarks so there are two paths
        path_facialRM = dijkstra3d.dijkstra(field, R, M, connectivity = con_facial, bidirectional=True)

        label_26 = mr_scan.copy()*0
        for (x,y,z) in path_facialRM:
            label_26[x,y,z] = 1

        np.savetxt("paths/facialtxtRM/" + filenames[nr][id:-6] + "txt", path_facialRM,fmt='%s', delimiter = ",")

        path_facialMT = dijkstra3d.dijkstra(field, M, T, connectivity = con_facial, bidirectional=True)
        for (x,y,z) in path_facialMT:
            label_26[x,y,z] = 1

        sitk.WriteImage(sitk.GetImageFromArray(label_26), os.path.join('paths/paths_facial',filenames[nr][id:])  , useCompression=True)
        np.savetxt("paths/facialtxtMT/" + filenames[nr][id:-6] + "txt", path_facialMT,fmt='%s', delimiter = ",")



        # Compute vector with intensisties for each candidate path

        val_chorda=[]
        for (x,y,z) in path_chorda:
            val_chorda.append(field[x,y,z])       
        
        val_facial = []
        for (x,y,z) in path_facialRM:
            val_facial.append(field[x,y,z])
        
        for (x, y, z) in path_facialMT:
            val_facial.append(field[x,y,z])


            
        val_chorda = np.array(val_chorda)
        val_facial = np.array(val_facial)
        
        # Compute comulative squared derivative
        rat_chorda = (np.sum(np.square(np.array(val_chorda[1:]-val_chorda[:-1], dtype='int64')))+np.sum(np.square(np.array(val_chorda[1:2:]-val_chorda[0:2:-1], dtype='int64'))))#/(len(val_26))
        rat_facial = (np.sum(np.square(np.array(val_facial[1:]-val_facial[:-1], dtype='int64')))+np.sum(np.square(np.array(val_facial[1:2:]-val_facial[0:2:-1], dtype='int64'))))#/(len(val_18))
        

        weights[ind,:]= [rat_chorda,rat_facial]
        ind += 1
    
    return weights 
    
    