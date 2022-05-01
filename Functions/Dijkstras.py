import imp
import dijkstra3d
import numpy as np
import pandas as pd
import SimpleITK as sitk
from sympy import nroots
from .computeAngle import *
import matplotlib.pyplot as plt 
from .Dataloader import *
import matplotlib as mat
from .color import *

class Dijkstras: 
    def __init__(self, df_path, nr_agents,nr,rotation): 
        self.df_path = df_path
        self.df = pd.read_csv(self.df_path)
        self.nr_image = self.df.shape[0]
        self.weights_model = np.zeros((self.nr_image, 2))
        self.weights_ann = np.zeros((self.nr_image, 2))
        self.nr_agents = nr_agents 
        self.idx = get_best_agents(self.df) if nr_agents == 12 else [0, 1, 2, 3, 4, 5]
        self.nr_model = nr
        self.nr_ann = nr + 22
        self.rotation = rotation
        


    def read_from_df(self,idx): 
        position_agent = np.array([self.df.loc[self.nr_model, "Agent " + str(idx) + " pos x"], self.df.loc[self.nr_model, "Agent " + str(idx) + " pos y"], self.df.loc[self.nr_model, "Agent " + str(idx) + " pos z"],])
        position_ann = np.array([self.df.loc[self.nr_model, "Landmark " + str(idx) + " pos x"], self.df.loc[self.nr_model, "Landmark " + str(idx) + " pos y"], self.df.loc[self.nr_model, "Landmark " + str(idx) + " pos z"],])
        return position_agent, position_ann


    def run_dijkstra(self, con_chorda, con_facial):
        ind = 0

        with open(r"Cmarl\src\data\filenames\testing.txt", 'r') as file: 
            filenames = file.read().split("\n")
        file.close()

        for nr in range(self.nr_image):
            if nr >= 9: 
                id = 41
            else: 
                id = 36
            
            print(f"nr = {nr}")
            C_agent,C_ann = self.read_from_df(self.idx[0])
            A_agent,A_ann = self.read_from_df(self.idx[1])
            R_agent,R_ann = self.read_from_df(self.idx[2])
            M_agent,M_ann = self.read_from_df(self.idx[3])
            T_agent,T_ann = self.read_from_df(self.idx[4])

            #load image
            mr_scan = sitk.GetArrayFromImage(sitk.ReadImage(filenames[nr]))

            #positive (but mr is always positive)
            field = mr_scan
                    
            # path is an [N,3] numpy array i.e. a list of x,y,z coordinates
            # terminates early, default is 26 connected
            # conectivity 18 2 landmarks
            path_chorda_model = dijkstra3d.dijkstra(field, A_agent, C_agent, connectivity = con_chorda, bidirectional = True)
            path_chorda_ann = dijkstra3d.dijkstra(field, A_ann, C_ann, connectivity = con_chorda, bidirectional = True)
            
            label_model = mr_scan.copy()*0
            label_ann = mr_scan.copy()*0

            for (x,y,z) in path_chorda_model:
                label_model[x,y,z] = 1
            
            for (x,y,z) in path_chorda_ann:
                label_ann[x,y,z] = 1

            sitk.WriteImage(sitk.GetImageFromArray(label_model), 'paths/paths_chorda/Agent'+ "_"+ filenames[nr][id:]  , useCompression = True)
            sitk.WriteImage(sitk.GetImageFromArray(label_ann), 'paths/paths_chorda/Landmark'+ "_"+ filenames[nr][id:]  , useCompression = True)

            np.savetxt("paths/chordatxt/Agent" + "_" + filenames[nr][id:-6] + "txt", path_chorda_model, fmt='%s')
            np.savetxt("paths/chordatxt/Landmark" + "_" + filenames[nr][id:-6] + "txt", path_chorda_ann, fmt='%s')
            
            
            #here we use 3 landmarks so there are two paths
            path_facialRM_model = dijkstra3d.dijkstra(field, R_agent, M_agent, connectivity = con_facial, bidirectional = True)
            path_facialRM_ann = dijkstra3d.dijkstra(field, R_ann, M_ann, connectivity = con_facial, bidirectional = True)


            label_26_model = mr_scan.copy()*0
            for (x,y,z) in path_facialRM_model:
                label_26_model[x,y,z] = 1

            label_26_ann = mr_scan.copy()*0
            for (x,y,z) in path_facialRM_ann:
                label_26_ann[x,y,z] = 1

            np.savetxt("paths/facialtxtRM/Agent" + "_" +  filenames[nr][id:-6] + "txt", path_facialRM_model,fmt='%s')
            np.savetxt("paths/facialtxtRM/Landmark" + "_" +  filenames[nr][id:-6] + "txt", path_facialRM_ann,fmt='%s')

            path_facialMT_model = dijkstra3d.dijkstra(field, M_agent, T_agent, connectivity = con_facial, bidirectional = True)
            path_facialMT_ann = dijkstra3d.dijkstra(field, M_ann, T_ann, connectivity = con_facial, bidirectional = True)

            for (x,y,z) in path_facialMT_model:
                label_26_model[x,y,z] = 1
            
            for (x,y,z) in path_facialMT_ann:
                label_26_ann[x,y,z] = 1

            sitk.WriteImage(sitk.GetImageFromArray(label_26_model), 'paths/paths_facial/Agent' + "_" + filenames[nr][id:]  , useCompression = True)
            sitk.WriteImage(sitk.GetImageFromArray(label_26_ann), 'paths/paths_facial/Landmark' + "_" + filenames[nr][id:]  , useCompression = True)
            
            np.savetxt("paths/facialtxtMT/Agent"  + "_" + filenames[nr][id:-6] + "txt", path_facialMT_model,fmt='%s')
            np.savetxt("paths/facialtxtMT/Landmark"  + "_" + filenames[nr][id:-6] + "txt", path_facialMT_ann,fmt='%s')


            # Compute vector with intensisties for each candidate path

            val_chorda_model = []
            val_chorda_ann = []
            for (x,y,z) in path_chorda_model:
                val_chorda_model.append(field[x,y,z])       

            for (x,y,z) in path_chorda_ann:
                val_chorda_ann.append(field[x,y,z])   
            
            val_facial_model = []
            val_facial_ann = []
            for (x,y,z) in path_facialRM_model:
                val_facial_model.append(field[x,y,z])
            
            for (x,y,z) in path_facialRM_ann:
                val_facial_ann.append(field[x,y,z])

            for (x, y, z) in path_facialMT_model:
                val_facial_model.append(field[x,y,z])
            
            for (x, y, z) in path_facialMT_ann:
                val_facial_ann.append(field[x,y,z])

                
            val_chorda_model = np.array(val_chorda_model)
            val_chorda_ann = np.array(val_chorda_ann)
            val_facial_model = np.array(val_facial_model)
            val_facial_ann = np.array(val_facial_model)
            
            # Compute comulative squared derivative
            rat_chorda_model = (np.sum(np.square(np.array(val_chorda_model[1:]-val_chorda_model[:-1], dtype='int64')))+np.sum(np.square(np.array(val_chorda_model[1:2:]-val_chorda_model[0:2:-1], dtype='int64'))))#/(len(val_26))
            rat_facial_model = (np.sum(np.square(np.array(val_facial_model[1:]-val_facial_model[:-1], dtype='int64')))+np.sum(np.square(np.array(val_facial_model[1:2:]-val_facial_model[0:2:-1], dtype='int64'))))#/(len(val_18))
            rat_chorda_ann = (np.sum(np.square(np.array(val_chorda_ann[1:]-val_chorda_ann[:-1], dtype='int64')))+np.sum(np.square(np.array(val_chorda_ann[1:2:]-val_chorda_ann[0:2:-1], dtype='int64'))))#/(len(val_26))
            rat_facial_ann = (np.sum(np.square(np.array(val_facial_ann[1:]-val_facial_ann[:-1], dtype='int64')))+np.sum(np.square(np.array(val_facial_ann[1:2:]-val_facial_ann[0:2:-1], dtype='int64'))))#/(len(val_18))
            

            self.weights_model[ind,:]= [rat_chorda_model,rat_facial_model]
            self.weights_ann[ind,:]= [rat_chorda_ann,rat_facial_ann]
            ind += 1
    
    def pcaDijkstras(self,chorda, facial):
        mu_chorda = np.mean(chorda, axis = 0)
        _, _, V_chorda = np.linalg.svd(chorda - mu_chorda)
        V_chorda = V_chorda.T 
        v2_chorda = V_chorda[:, 0]

        mu_facial = np.mean(facial, axis = 0)
        _, _, V_facial = np.linalg.svd(facial - mu_facial)
        V_facial = V_facial.T 
        v2_facial = V_facial[:, 0]

        if np.sign(v2_chorda[2]) != np.sign(v2_facial[2]):
            v2_facial = -v2_facial

        v2_chorda =  v2_chorda 
        v2_facial =  v2_facial

        return v2_chorda, v2_facial

    def compute_angle(self, chorda, facial): 
        """
        Computes the angles between the two nerves after projection
        """
        chorda = np.array([chorda[0], chorda[1]])
        facial = -1 *  np.array([facial[0], facial[1]])

        angle = np.arccos((chorda @ facial ) /(np.linalg.norm(chorda) * np.linalg.norm(facial))) 
        if angle >= np.pi: 
            angle = np.pi - angle 

        return angle * 180 / np.pi 

    def get_translation_chorda(self, points_rot, chorda): 
        """
        Compute translatio of chorda 
        """
        length = np.linalg.norm(points_rot[0][0:2] - points_rot[1][0:2])
        l = np.reshape(np.array(np.linspace(0, length, 200)), (200, 1))
        mu = np.mean(points_rot[2:], axis = 0)
        chorda_plot = chorda * l +mu
        # p1 = chorda_plot[-1]
        # p2 = points_rot[0][0:2]
        # translation = p2 - p1   
        return chorda_plot,l

    def get_translation_facial(self, points_rot, facial): 
        """
        Compute translatio of facial 
        """

        length = np.linalg.norm(points_rot[2][0:2] - points_rot[4][0:2])
        l = np.reshape(np.array(np.linspace(-length/2 - 5, length/2 + 5, 200)), (200, 1))
        mu = np.mean(points_rot[2:], axis = 0)
        facial_plot = facial * l + mu 
        return facial_plot, l


    def visualize(self):

        paths_chorda = "paths\\chordatxt"
        paths_facialRM = "paths\\facialtxtRM"
        paths_facialMT = "paths\\facialtxtMT"
        
        Filenames = getFiles(paths_chorda)

        chorda_path_model = paths_chorda+"\\"+Filenames[self.nr_model]
        FacialRM_path_model = paths_facialRM+"\\"+Filenames[self.nr_model]
        FacialMT_path_model = paths_facialMT+"\\"+Filenames[self.nr_model]

        chorda_path_ann = paths_chorda+"\\"+Filenames[self.nr_ann]
        FacialRM_path_ann = paths_facialRM+"\\"+Filenames[self.nr_ann]
        FacialMT_path_ann = paths_facialMT+"\\"+Filenames[self.nr_ann]

        chorda_model = np.loadtxt(chorda_path_model, dtype=int)
        facialRM_model = np.loadtxt(FacialRM_path_model, dtype=int)
        facialMT_model = np.loadtxt(FacialMT_path_model, dtype=int)

        chorda_ann = np.loadtxt(chorda_path_ann, dtype=int)
        facialRM_ann = np.loadtxt(FacialRM_path_ann, dtype=int)
        facialMT_ann = np.loadtxt(FacialMT_path_ann, dtype=int)


        facial_model = np.concatenate([facialRM_model,facialMT_model], axis = 0)
        facial_ann = np.concatenate([facialRM_ann,facialMT_ann], axis = 0)


        chorda_direction_model, facial_direction_model = self.pcaDijkstras(chorda_model, facial_model)
        chorda_direction_ann, facial_direction_ann = self.pcaDijkstras(chorda_ann, facial_ann)
        
        facial_rot_model = (self.rotation @ (np.reshape(facial_direction_model,(3,1)))).T 
        chorda_rot_model = (self.rotation @ (np.reshape(chorda_direction_model,(3,1)))).T 

        facial_rot_ann = (self.rotation @ (np.reshape(facial_direction_ann,(3,1)))).T 
        chorda_rot_ann = (self.rotation @ (np.reshape(chorda_direction_ann,(3,1)))).T 

        chorda_point_model = (self.rotation @ chorda_model.T).T 
        chorda_point_ann = (self.rotation @ chorda_ann.T).T 

        facial_point_model = (self.rotation @ facial_model.T).T 
        facial_point_ann = (self.rotation @ facial_ann.T).T 

        facial_model,xn_facial_model = self.get_translation_facial(facial_point_model,facial_rot_model)
        facial_ann,xn_facial_ann = self.get_translation_facial(facial_point_ann,facial_rot_ann)
        chorda_model,xn_chorda_model = self.get_translation_chorda(chorda_point_model,chorda_rot_model)
        chorda_ann,xn_chorda_ann = self.get_translation_chorda(chorda_point_ann,chorda_rot_ann)

        # angle_ann = self.compute_angle(chorda_rot_ann, facial_rot_ann)
        # angle_model = self.compute_angle(chorda_rot_model, facial_rot_model)
       
        angle_ann = 0
        angle_model = 0
       
        return chorda_point_model, chorda_point_ann,facial_point_model, facial_point_ann, angle_ann, angle_model, facial_model,facial_ann, chorda_model,chorda_ann ,xn_chorda_model,xn_chorda_ann,xn_facial_model,xn_facial_ann


    def plotDijkstras(self,chorda_point, facial_point, chorda, facial,  xn_chorda,xn_facial,angle,title):
        mat.rcParams.update({'font.size': 18})
        plt.figure()
        plt.scatter(chorda_point[:, 0], chorda_point[:, 1], color = "b", label = "CTY") # A
        plt.scatter(facial_point[:, 0], facial_point[:, 1], color = "r", label = "FN")
        plt.axis("square")
        plt.plot(xn_chorda, chorda, linestyle = '-', color = col(8).color, label = "CTY")
        plt.plot(xn_facial,facial, linestyle = '-', color = col(1).color, label = "FN")
        plt.legend()
        plt.title(title + ": " + "Angle = " + str(angle))
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def plot_both(self):
        chorda_point_model, chorda_point_ann,facial_point_model, facial_point_ann, angle_ann, angle_model, facial_model,facial_ann, chorda_model,chorda_ann, xn_chorda_model,xn_chorda_ann,xn_facial_model,xn_facial_ann = self.visualize()

        self.plotDijkstras(chorda_point_model, facial_point_model, chorda_model, facial_model, xn_chorda_model, xn_facial_model,angle_model,"Model")
        self.plotDijkstras(chorda_point_ann, facial_point_ann, chorda_ann, facial_ann, xn_chorda_ann,xn_facial_ann ,angle_ann,"Landmark")
