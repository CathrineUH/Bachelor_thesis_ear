import dijkstra3d
import numpy as np
import pandas as pd
import SimpleITK as sitk
from .computeAngle import *
import matplotlib.pyplot as plt 
from .Dataloader import *
import matplotlib as mat
from .color import *
from .angleVis import *

class Dijkstras: 
    def __init__(self, df_path, nr_agents, nr, rotation): 
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
        
        


    def read_from_df(self,nr,idx): 
        position_agent = np.array([self.df.loc[nr, "Agent " + str(idx) + " pos x"], self.df.loc[nr, "Agent " + str(idx) + " pos y"], self.df.loc[nr, "Agent " + str(idx) + " pos z"],])
        position_ann = np.array([self.df.loc[nr, "Landmark " + str(idx) + " pos x"], self.df.loc[nr, "Landmark " + str(idx) + " pos y"], self.df.loc[nr, "Landmark " + str(idx) + " pos z"],])
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
        
            idx = np.where(self.df.loc[:, 'Filename 0'] == filenames[nr][id:-7])[0][0]
            
            C_agent,C_ann = self.read_from_df(idx, self.idx[0])
            A_agent,A_ann = self.read_from_df(idx, self.idx[1])
            R_agent,R_ann = self.read_from_df(idx, self.idx[2])
            M_agent,M_ann = self.read_from_df(idx, self.idx[3])
            T_agent,T_ann = self.read_from_df(idx, self.idx[4])
            
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
                label_model[z, y, x] = 1
            
            for (x,y,z) in path_chorda_ann:
                label_ann[z, y, x] = 1

            sitk.WriteImage(sitk.GetImageFromArray(label_model), 'paths/paths_chorda/Agent'+ "_"+ filenames[nr][id:]  , useCompression = True)
            sitk.WriteImage(sitk.GetImageFromArray(label_ann), 'paths/paths_chorda/Landmark'+ "_"+ filenames[nr][id:]  , useCompression = True)

            np.savetxt("paths/chordatxt/Agent" + "_" + filenames[nr][id:-6] + "txt", path_chorda_model, fmt='%s')
            np.savetxt("paths/chordatxt/Landmark" + "_" + filenames[nr][id:-6] + "txt", path_chorda_ann, fmt='%s')
            
            
            #here we use 3 landmarks so there are two paths
            path_facialRM_model = dijkstra3d.dijkstra(field, R_agent, M_agent, connectivity = con_facial, bidirectional = True)
            path_facialRM_ann = dijkstra3d.dijkstra(field, R_ann, M_ann, connectivity = con_facial, bidirectional = True)

            label_26_model = mr_scan.copy()*0
            for (x,y,z) in path_facialRM_model:
                label_26_model[z, y, x] = 1

            label_26_ann = mr_scan.copy()*0
            for (x,y,z) in path_facialRM_ann:
                label_26_ann[z, y, x] = 1

            np.savetxt("paths/facialtxtRM/Agent" + "_" +  filenames[nr][id:-6] + "txt", path_facialRM_model,fmt='%s')
            np.savetxt("paths/facialtxtRM/Landmark" + "_" +  filenames[nr][id:-6] + "txt", path_facialRM_ann,fmt='%s')

            path_facialMT_model = dijkstra3d.dijkstra(field, M_agent, T_agent, connectivity = con_facial, bidirectional = True)
            path_facialMT_ann = dijkstra3d.dijkstra(field, M_ann, T_ann, connectivity = con_facial, bidirectional = True)

            for (x,y,z) in path_facialMT_model:
                label_26_model[z, y, x] = 1
            
            for (x,y,z) in path_facialMT_ann:
                label_26_ann[z, y, x] = 1

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
        v_chorda = V_chorda[:, 0]

        mu_facial = np.mean(facial, axis = 0)
        _, _, V_facial = np.linalg.svd(facial - mu_facial)
        V_facial = V_facial.T 
        v_facial = V_facial[:, 0]

        if np.sign(v_chorda[2]) != np.sign(v_facial[2]):
            v_facial = -v_facial

        return v_chorda, v_facial

    def compute_angle(self, chorda, facial): 
        """
        Computes the angles between the two nerves after projection
        """
        chorda = np.array([chorda[0][0], chorda[0][1]])
        facial = np.array([facial[0][0], facial[0][1]])

        angle = np.arccos((chorda @ facial ) /(np.linalg.norm(chorda) * np.linalg.norm(facial))) 
        if angle >= np.pi: 
            angle = np.pi - angle 

        return angle * 180 / np.pi 

    def get_translation_chorda(self, points_rot, chorda): 
        """
        Compute translatio of chorda 
        """
        length = np.linalg.norm(points_rot[np.argmin(points_rot[:, 0])] - points_rot[np.argmax(points_rot[:, 0])])
        l = np.reshape(np.array(np.linspace(-length/2.0, length/2.0, 200)), (200, 1))
        mu = np.mean(points_rot, axis = 0)
        chorda_plot = chorda * l + mu 
        return chorda_plot

    def get_translation_facial(self, points_rot, facial): 
        """
        Compute translation of facial 
        """

        length = np.linalg.norm(points_rot[np.argmin(points_rot[:, 1])] - points_rot[np.argmax(points_rot[:, 1])])
        l = np.reshape(np.array(np.linspace(-length/2.0, length/2.0,  200)), (200, 1))
        mu = np.mean(points_rot, axis = 0)
        facial_plot = facial * l + mu 
        return facial_plot

    def intersection(self, chorda, facial,angle,number):
        # defining points of lines 
        x1, y1, _ = chorda[0]
        x2, y2, _= chorda[-1]
        x3, y3, _ = facial[0]
        x4, y4, _ = facial[-1]

        D = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        first = (x1 * y2 - y1 * x2) 
        last = (x3 * y4 - y3 * x4)

        # points of intersection 
        Px = (first * (x3 - x4) - (x1 - x2) * last)/ D
        Py = (first * (y3 - y4) - (y1 - y2) * last) / D
        r = 2 

        x_list = []
        y_list = []
        slope_facial = (y4 - y3) / (x4 - x3)
        slope_chorda = (y2 - y1) / (x2 - x1)
        b_facial = y4 - slope_facial * x4 
        b_chorda = y1 - slope_chorda * x1 
        def line_circle_intersect(m, b, x0, y0, r): 
            c1 = 1 + m ** 2
            c2 = - 2.0 * x0 + 2 * m * ( b - y0 )
            c3 = x0 * 2 + ( b - y0 ) * 2 - r ** 2

            # solve the quadratic equation:
            delta = c2 ** 2 - 4.0 * c1 * c3

            x1 = ( - c2 + np.sqrt(delta) ) / ( 2.0 * c1 )
            x2 = ( - c2 - np.sqrt(delta) ) / ( 2.0 * c1 )

            x_list.append(x1)
            x_list.append(x2)

            y1 = m * x1 + b
            y2 = m * x2 + b

            y_list.append(y1)
            y_list.append(y2)
                
        line_circle_intersect(slope_facial, b_facial, Px, Py, r)
        line_circle_intersect(slope_chorda, b_chorda, Px, Py, r)

        def get_point_angle(x,y,x0,y0):

            num = x - x0
            den = np.sqrt( ( x - x0 )**2 + ( y - y0 )**2 )

            theta = np.arccos( num / den )

            if not y - y0 >= 0: theta = 2 * np.pi - theta

            #print(theta, np.rad2deg(theta), y - y0 )

            return theta

        theta_list = []

        for i in range(len(x_list)):

            x = x_list[i]
            y = y_list[i]

            theta_list.append(get_point_angle(x,y,Px,Py) )
        
        P0 = theta_list[0]
        P1 = theta_list[1]
        P2 = theta_list[2]
        P3 = theta_list[3]
       
        if(abs(P1-P3)*180/np.pi > angle-0.1 and abs(P1-P3)*180/np.pi<angle+0.1):
            theta1 = P1 
            theta2 = P3

        elif((abs(P0-P2)*180/np.pi > angle-0.1 and abs(P0-P2)*180/np.pi<angle+0.1)):
            theta1 = P0
            theta2 = P2

        elif((abs(P0-P3)*180/np.pi > angle-0.1 and abs(P0-P3)*180/np.pi < angle+0.1)):
            theta1 = P0
            theta2 = P3
        elif((abs(P2-P3)*180/np.pi > angle-0.1 and abs(P2-P3)*180/np.pi<angle+0.1)):
            theta1 = P2
            theta2 = P3
        elif((abs(P0-P1)*180/np.pi > angle-0.1 and abs(P0-P1)*180/np.pi<angle+0.1)):
            theta1 = P0
            theta2 = P1
        else:
            theta1 = P1
            theta2 = P2

            
        circ = np.linspace(theta1, theta2, 100)
        circ_x = r * np.cos(circ) + Px 
        circ_y = r * np.sin(circ) + Py 

        mid_angle = (theta1 + theta2) / 2.0 
        mid_angle_x = (r + number) * np.cos(mid_angle) + Px -1
        mid_angle_y = (r + number) * np.sin(mid_angle) + Py
        return circ_x, circ_y, mid_angle_x, mid_angle_y, Px,Py

    def drilling_point(self, angle, c, px, py):
        Hoz_point = np.array([[1],[0]])
        C_point = c[-1][0:2]-c[0][0:2]
        angle_chorda = np.arccos((C_point@ Hoz_point) /(np.linalg.norm(C_point) * np.linalg.norm(Hoz_point))) 
                
        if angle_chorda >= np.pi: 
            angle_chorda = np.pi - angle_chorda

        angle = angle/2
        angle = angle*np.pi/180
        angle_mid = (angle_chorda-angle)

        lenght_mid = 1.5 / np.sin(angle)
        
        Dril_x = px + (lenght_mid) * np.cos(angle_mid)
        Dril_y = py + (lenght_mid) * np.sin(angle_mid)
        return Dril_x, Dril_y

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

        facial_model = self.get_translation_facial(facial_point_model,facial_rot_model)
        facial_ann   = self.get_translation_facial(facial_point_ann,facial_rot_ann)
        chorda_model = self.get_translation_chorda(chorda_point_model,chorda_rot_model)
        chorda_ann   = self.get_translation_chorda(chorda_point_ann,chorda_rot_ann)

        angle_ann = self.compute_angle(chorda_rot_ann, facial_rot_ann)
        angle_model = self.compute_angle(chorda_rot_model, facial_rot_model)

       
        return chorda_point_model, chorda_point_ann,facial_point_model, facial_point_ann, angle_ann, angle_model, facial_model,facial_ann, chorda_model,chorda_ann


    def plotDijkstras(self,chorda_point, facial_point, chorda, facial, Dril_x, Dril_y, angle,title):
        mat.rcParams.update({'font.size': 18})
        plt.figure(figsize=(8,6))
        plt.scatter(chorda_point[:, 0], chorda_point[:, 1], marker = "*", color = [0.63215686, 0.39607843, 0.68980392], s = 50) 
        plt.scatter(facial_point[:, 0], facial_point[:, 1],marker = "o", color =[0.3 , 0.67333333, 0.4454902],s = 50)
        plt.axis("square")
        plt.plot(chorda[:, 0], chorda[:, 1], linestyle = '-', color = [0.63215686, 0.39607843, 0.68980392], label = "CTY",linewidth = 5*0.4)
        plt.plot(facial[:, 0], facial[:, 1], linestyle = '-', color = [0.3 , 0.67333333, 0.4454902], label = "FN",linewidth = 5*1.2)
        plt.scatter(Dril_x, Dril_y, marker = "x", color = col(0).color, label = "DP")
        plt.legend()
        plt.title(title + ": " + "Angle = " + str(angle))
        # plt.title("Shortest Path")
        plt.xticks([])
        plt.yticks([])
        min_x = np.min(np.concatenate([facial[:, 0],chorda[:, 0]], axis = 0))
        min_y = np.min(np.concatenate([facial[:, 1],chorda[:, 1]], axis = 0))
        max_x = np.max(np.concatenate([facial[:, 0],chorda[:, 0]], axis = 0))
        max_y = np.max(np.concatenate([facial[:, 1],chorda[:, 1]], axis = 0))

        x_lim = max_x-min_x
        y_lim = max_y-min_y
        if(x_lim> y_lim):
            val = (x_lim-y_lim)/2
            max_y = max_y +val
            min_y = min_y -val
        else:
            val = (y_lim-x_lim)/2
            max_x = max_x + val
            min_x = min_x - val

        plt.xlim([min_x - 2, max_x + 2])
        plt.ylim([min_y - 2, max_y + 2])

        plt.show()
        return chorda, facial

    def plot_both(self):
        Viz = VisualizeAngle(self.df_path, self.nr_agents)
        Viz.get_coor()
        chorda_point_model, chorda_point_ann,facial_point_model, facial_point_ann, angle_ann, angle_model, facial_model,facial_ann, chorda_model,chorda_ann = self.visualize()
        Px_model, Py_model = self.intersection(chorda_model, facial_model)
        Px_ann, Py_ann = self.intersection(chorda_ann, facial_ann)
        Dril_x_model, Dril_Y_model = self.drilling_point(angle_model,chorda_model,Px_model,Py_model)
        Dril_x_ann, Dril_Y_ann = self.drilling_point(angle_ann,chorda_ann,Px_ann,Py_ann)
        # Viz.plot_angle_ann(self.nr_model,Dril_x_ann = Dril_x_model, Dril_y_ann = Dril_Y_model)
        # chorda_model, facial_model = self.plotDijkstras(chorda_point_model, facial_point_model, chorda_model, facial_model, Dril_x_ann, Dril_Y_ann ,np.round(angle_model, 2),"Model")
        chorda_ann, facial_ann = self.plotDijkstras(chorda_point_ann, facial_point_ann, chorda_ann, facial_ann, Dril_x_ann, Dril_Y_ann,np.round(angle_ann, 2),"Landmark")
    
    def compute_all_angles(self): 
        angles = np.zeros((self.nr_image, 2))

        for nr in range(self.nr_image): 
            self.nr_model = nr
            self.nr_ann = nr + 22 
            Viz = VisualizeAngle(self.df_path, self.nr_agents)
            Viz.get_coor()
            Viz.plot_angle_in_plane(self.nr_model, True)
            self.rotation = Viz.R
            _, _,_, _, angle_ann, angle_model, _,_, _,_ = self.visualize()
            angles[nr, 0], angles[nr, 1] = angle_ann, angle_model 
        
        return angles 
    def plot_together(self):
        chorda_point_model, chorda_point_ann,facial_point_model, facial_point_ann, angle_ann, angle_model, facial_model,facial_ann, chorda_model,chorda_ann = self.visualize()
        circ_x_model, circ_y_model, mid_angle_x_model, mid_angle_y_model,Px_model, Py_model = self.intersection(chorda_model, facial_model,angle_model,2.8)
        circ_x_ann, circ_y_ann, mid_angle_x_ann, mid_angle_y_ann,Px_ann, Py_ann = self.intersection(chorda_ann, facial_ann,angle_ann, 2.8)
        Dril_x_model, Dril_Y_model = self.drilling_point(angle_model,chorda_model,Px_model,Py_model)
        Dril_x_ann, Dril_Y_ann = self.drilling_point(angle_ann,chorda_ann,Px_ann,Py_ann)
        mat.rcParams.update({'font.size': 18})
        plt.figure(figsize=(8,6))
        plt.plot(chorda_model[:, 0], chorda_model[:, 1], linestyle = '-', color = [0.3 , 0.67333333, 0.4454902], label = "Model", alpha = 0.7, linewidth = 5*0.4, zorder=0)
        plt.plot(facial_model[:, 0], facial_model[:, 1], linestyle = '-', color = [0.3 , 0.67333333, 0.4454902], alpha = 0.7,linewidth = 5*1.2,zorder=0)

        plt.plot(chorda_ann[:, 0], chorda_ann[:, 1], linestyle = '-', color = [0.63215686, 0.39607843, 0.68980392], label = "Ann", alpha = 0.7,linewidth = 5*0.4,zorder=0)
        plt.plot(facial_ann[:, 0], facial_ann[:, 1], linestyle = '-', color = [0.63215686, 0.39607843, 0.68980392], alpha = 0.7,linewidth = 5*1.2,zorder=0)

        
        # plt.scatter(chorda_point_model[:, 0],  chorda_point_model[:, 1], marker = "o", color =[0.3 , 0.67333333, 0.4454902], s=50,zorder=5)
        # plt.scatter(facial_point_model[:, 0],  facial_point_model[:, 1], marker = "o", color =[0.3 , 0.67333333, 0.4454902], s=50,zorder=5)
        # plt.scatter(chorda_point_ann[:, 0], chorda_point_ann[:, 1], marker = "*", color = [0.63215686, 0.39607843, 0.68980392], s=50,zorder=5 )
        # plt.scatter(facial_point_ann[:, 0], facial_point_ann[:, 1], marker = "*", color = [0.63215686, 0.39607843, 0.68980392], s=50,zorder=5 )

        plt.plot(circ_x_model, circ_y_model, color = [0.3 , 0.67333333, 0.4454902])
        plt.text(51,52, np.round(angle_model,2), fontsize = 16)
        plt.plot(circ_x_ann, circ_y_ann, color = [0.63215686, 0.39607843, 0.68980392])
        plt.text(mid_angle_x_ann,mid_angle_y_ann, np.round(angle_ann,2), fontsize = 16)
        plt.scatter(Dril_x_model, Dril_Y_model, marker = "x", color = col(0).color, label = "Model", s = 50, zorder=10)
        plt.scatter(Dril_x_ann, Dril_Y_ann, marker = "x", color = col(3).color, label = "Ann",s = 50, zorder=10)

        plt.legend(loc = 'lower left')
        plt.axis("square")
        plt.gca().set_aspect("equal")
        # plt.title(title)
        min_x = np.min(np.concatenate([chorda_model[:, 0],facial_model[:, 0],facial_ann[:, 0],chorda_ann[:, 0]], axis = 0))
        min_x = np.min([min_x,Dril_x_model,Dril_x_ann])
        min_y = np.min(np.concatenate([chorda_model[:, 1],facial_model[:, 1],facial_ann[:, 1],chorda_ann[:, 1]], axis = 0))
        min_y = np.min([min_y,Dril_Y_model,Dril_Y_ann])
        max_x = np.max(np.concatenate([chorda_model[:, 0],facial_model[:, 0],facial_ann[:, 0],chorda_ann[:, 0]], axis = 0))
        max_x = np.max([max_x,Dril_x_model,Dril_x_ann])
        max_y = np.max(np.concatenate([chorda_model[:, 1],facial_model[:, 1],facial_ann[:, 1],chorda_ann[:, 1]], axis = 0))
        max_y = np.max([max_y,Dril_Y_model,Dril_Y_ann])

        x_lim = max_x-min_x
        y_lim = max_y-min_y
        if(x_lim> y_lim):
            val = (x_lim-y_lim)/2
            max_y = max_y +val
            min_y = min_y -val
        else:
            val = (y_lim-x_lim)/2
            max_x = max_x +val
            min_x = min_x -val
        plt.title("On Nerve")
        plt.xlim([min_x - 2, max_x + 2])
        plt.ylim([min_y - 2, max_y + 2])
        plt.xticks([])
        plt.yticks([])
        plt.show()
    
      
