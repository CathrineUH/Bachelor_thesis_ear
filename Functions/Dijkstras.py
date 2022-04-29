import dijkstra3d
import numpy as np
import os
import pandas as pd
import SimpleITK as sitk
from skimage import data
from .computeAngle import *
import matplotlib.pyplot as plt 
from .Dataloader import *
import matplotlib as mat


def read_from_df(df, idx, nr, which): 
    position = np.array([df.loc[nr, which + " " + str(idx) + " pos x"], df.loc[nr, which + " " + str(idx) + " pos y"], df.loc[nr, which + " " + str(idx) + " pos z"],])
    return position


def run_dijkstra(results, nr_agents, which, con_chorda, con_facial):
    ind = 0
    data = pd.read_csv(results)
    weights = np.zeros((data.shape[0], 2))
    idx = get_best_agents(data) if nr_agents == 12 else [0, 1, 2, 3, 4, 5]
    with open(r"Cmarl\src\data\filenames\testing.txt", 'r') as file: 
        filenames = file.read().split("\n")
    file.close()

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



def visualize(nr,rotation):

    paths_chorda = "paths\\chordatxt"
    paths_facialRM = "paths\\facialtxtRM"
    paths_facialMT = "paths\\facialtxtMT"
    
    Filenames_chorda = getFiles(paths_chorda)
    Filenames_facialRM = getFiles(paths_facialRM)
    Filenames_facialMT = getFiles(paths_facialMT)

    chorda_path = paths_chorda+"\\"+Filenames_chorda[nr]
    FacialRM_path = paths_facialRM+"\\"+Filenames_facialRM[nr]
    FacialMT_path = paths_facialMT+"\\"+Filenames_facialMT[nr]

    chorda = np.loadtxt(chorda_path, dtype=int)
    facialRM = np.loadtxt(FacialRM_path, dtype=int)
    facialMT = np.loadtxt(FacialMT_path, dtype=int)

    path_chorda = (rotation @ chorda.T).T 
    facial = np.concatenate([facialRM,facialMT], axis = 0)

    path_facial = (rotation @ facial.T).T 
    xn = np.linspace(-100, 100)

    slope_chorda, intersection_chorda = np.polyfit(path_chorda[:, 0], path_chorda[:, 1], 1)
    y_chorda = np.polyval([slope_chorda, intersection_chorda], xn)

    slope_facial, intersection_facial, = np.polyfit(path_facial[:, 0], path_facial[:, 1], 1)
    y_facial = np.polyval([slope_facial, intersection_facial], xn)  

    p1 = np.array([xn[0], y_chorda[0]])
    p2 = np.array([xn[-1], y_chorda[-1]])
    chorda_direction = p2 - p1 

    p3 = np.array([xn[0], y_facial[0]])
    p4 = np.array([xn[-1], y_facial[-1]])
    facial_direction = p4 - p3

    angle = np.arccos(chorda_direction @ facial_direction /(np.linalg.norm(chorda_direction) * np.linalg.norm(facial_direction))) * 180 / np.pi 
    return path_chorda, path_facial,angle, xn, y_chorda, y_facial


def plotDijkstras(path_chorda, path_facial,angle, xn, y_chorda, y_facial):
    mat.rcParams.update({'font.size': 18})
    plt.scatter(path_chorda[:, 0], path_chorda[:, 1], color = "b", label = "CTY") # A
    plt.scatter(path_facial[:, 0], path_facial[:, 1], color = "r", label = "FN")
    plt.axis("square")
    plt.plot(xn, y_chorda)
    plt.plot(xn, y_facial)
    plt.legend()
    plt.title("Angle = " + str(angle))
    plt.xticks([])
    plt.yticks([])
    plt.show()