# link: https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
from itertools import cycle
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib
from sqlalchemy import true
from .computeAngle import * 
from .color import * 


class VisualizeAngle: 
    def __init__(self, df_path, nr_agents): 
        self.df_path = df_path
        self.df = pd.read_csv(self.df_path)
        self.nr_image = self.df.shape[0]
        self.nr_agents = nr_agents 
        self.idx = get_best_agents(self.df) if nr_agents == 12 else [0, 1, 2, 3, 4, 5]
        self.coor_model = None
        self.coor_ann = None 
        self.plane_model = None
        self.plane_ann = None 
        self.R = None
    
    def get_coor(self): 
        """
        finds coordinates for all landmarks for both annotations and the model prediction
        """
        self.coor_model = np.zeros((self.nr_image, 5, 3))
        self.coor_ann = np.zeros((self.nr_image, 5,3))
        for nr in range(self.nr_image): 
            for i in range(5): 
                x = self.df.loc[nr, "Agent " + str(self.idx[i]) + " pos x"]
                y = self.df.loc[nr, "Agent " + str(self.idx[i]) + " pos y"]
                z = self.df.loc[nr, "Agent " + str(self.idx[i]) + " pos z"]
                self.coor_model[nr, i] = [x, y, z]

                x = self.df.loc[nr, "Landmark " + str(self.idx[i]) + " pos x"]
                y = self.df.loc[nr, "Landmark " + str(self.idx[i]) + " pos y"]
                z = self.df.loc[nr, "Landmark " + str(self.idx[i]) + " pos z"]
                self.coor_ann[nr, i] = [x, y, z]
        
    def get_points_for_plane(self, nr): 
        """
        find the points for the plane that the two vectors span 
        """
        chorda_model = self.coor_model[nr][0] - self.coor_model[nr][1]
        facial_model =  - get_pca_direction(self.coor_model[nr][2], self.coor_model[nr][3], self.coor_model[nr][4])    
        
        chorda_ann = self.coor_ann[nr][0] - self.coor_ann[nr][1]
        facial_ann =  - get_pca_direction(self.coor_ann[nr][2], self.coor_ann[nr][3], self.coor_ann[nr][4])

        proj_model = np.cross(chorda_model, facial_model)
        proj_ann = np.cross(chorda_ann, facial_ann)

        self.plane_model = np.reshape(np.array([[chorda_model], [facial_model], [proj_model]]), (3, 3))
        self.plane_ann = np.reshape(np.array([[chorda_ann], [facial_ann], [proj_ann]]), (3, 3))
        
    def get_translation_chorda(self, points_rot, chorda): 
        """
        Compute translatio of chorda 
        """
        length = np.linalg.norm(points_rot[0][0:2] - points_rot[1][0:2])
        l = np.reshape(np.array(np.linspace(0, length, 200)), (200, 1))
        chorda_plot = chorda * l 
        p1 = chorda_plot[-1]
        p2 = points_rot[0][0:2]
        translation = p2 - p1
    
        return chorda_plot, translation 

    def get_translation_facial(self, points_rot, facial): 
        """
        Compute translation of facial 
        """

        length = np.linalg.norm(points_rot[2][0:2] - points_rot[4][0:2])
        l = np.reshape(np.array(np.linspace(-length/2 - 5, length/2 + 5, 200)), (200, 1))
        mu = np.mean(points_rot[2:], axis = 0)
        facial_plot = facial * l + mu 
        return facial_plot

    def rotate(self, points_plane_ann, points_plane_model, points_ann, points_model, nr):
        """
        Rotates the normal vector of the plane to be the [0, 0, 1] such that 
        the plane is the xy-plane 
        """
        # Helper function # 
        chorda_ann = points_plane_ann[0]
        facial_nerve_ann = points_plane_ann[1]
        chorda_model = points_plane_model[0]
        facial_nerve_model = points_plane_model[1]
        n = points_plane_ann[2]
        n /= np.linalg.norm(n)

        # rotate normal vector to z 
        v = np.cross(n, np.array([0, 0, 1]))
        s = np.linalg.norm(v)
        c = np.array([0, 0, 1]) @ n 
        skew_v = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        self.R = np.identity(3) + skew_v + skew_v @ skew_v * (1 - c) / (s ** 2)

        chorda_rot_ann = self.R @ chorda_ann
        facial_rot_ann = self.R @ facial_nerve_ann
        p_rot_ann = (self.R @ points_ann[nr].T).T 

        chorda_rot_model = self.R @ chorda_model
        facial_rot_model = self.R @ facial_nerve_model
        p_rot_model = (self.R @ points_model[nr].T).T 

        chorda_rot_ann = chorda_rot_ann / np.linalg.norm(chorda_rot_ann) 
        facial_rot_ann = facial_rot_ann / np.linalg.norm(facial_rot_ann) 
        chorda_rot_model = chorda_rot_model / np.linalg.norm(chorda_rot_model) 
        facial_rot_model = facial_rot_model / np.linalg.norm(facial_rot_model) 

        chorda_plot_ann, translation_ann = self.get_translation_chorda(p_rot_ann, np.array([chorda_rot_ann[0], chorda_rot_ann[1]]))
        chorda_plot_model, translation_model = self.get_translation_chorda(p_rot_model, np.array([chorda_rot_model[0], chorda_rot_model[1]]))
        p_rot_ann[:, 0:2] -= translation_ann 
        p_rot_model[:, 0:2] -= translation_model 
        facial_plot_ann = self.get_translation_facial(p_rot_ann, facial_rot_ann)
        facial_plot_model = self.get_translation_facial(p_rot_model, facial_rot_model)

        return chorda_plot_ann, chorda_plot_model, chorda_rot_ann, chorda_rot_model, facial_plot_ann,facial_plot_model, facial_rot_ann, facial_rot_model,p_rot_ann, p_rot_model

    def project_onto_plane(self, nr): 
        chorda_plot_ann, chorda_plot_model,chorda_ann, chorda_model,facial_plot_ann, facial_plot_model,facial_ann ,facial_model,points_ann, points_model = self.rotate(self.plane_ann, self.plane_model, self.coor_ann, self.coor_model, nr)
        
        return chorda_plot_ann, chorda_ann, facial_plot_ann, facial_ann, points_ann, chorda_plot_model, chorda_model, facial_plot_model, facial_model, points_model 

    def compute_angle(self, chorda, facial): 
        """
        Computes the angles between the two nerves after projection
        """
        chorda = np.array([chorda[0], chorda[1]])
        facial = np.array([facial[0], facial[1]])
        if (facial[1]<=0.2):
            facial = -1 * facial
        angle = np.arccos((chorda @ facial ) /(np.linalg.norm(chorda) * np.linalg.norm(facial))) 
        if angle >= np.pi: 
            angle = np.pi - angle 

        return angle * 180 / np.pi 


    def plot_angle_slice(self, chorda, facial,angle, nr):
        # defining points of lines 
        x1, y1 = chorda[0]
        x2, y2 = chorda[-1]
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

            
        print("theta = " + str(theta1) + " " + str(theta2))
        circ = np.linspace(theta1, theta2, 100)
        circ_x = r * np.cos(circ) + Px 
        circ_y = r * np.sin(circ) + Py 

        mid_angle = (theta1 + theta2) / 2.0 
        mid_angle_x = (r + 2.8) * np.cos(mid_angle) + Px
        mid_angle_y = (r + 2.8) * np.sin(mid_angle) + Py
        return circ_x, circ_y, mid_angle_x, mid_angle_y, Px,Py

    def drilling_point(self,angle,c,px,py):
        Hoz_point = np.array([[1],[0]])
        C_point = c[-1]-c[0]
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

    def plot_angle_in_plane(self, nr, getRotation = False): 
        """
        This is the function to call. It plots the angle 
        """
        
        if getRotation: 
            self.get_points_for_plane(nr)
            chorda_plot_ann, chorda_ann, facial_plot_ann, facial_ann, points_ann, chorda_plot_model, chorda_model, facial_plot_model, facial_model, points_model   = self.project_onto_plane(nr)
        else: 
            self.get_points_for_plane(nr)
            chorda_plot_ann, chorda_ann, facial_plot_ann, facial_ann, points_ann, chorda_plot_model, chorda_model, facial_plot_model, facial_model, points_model   = self.project_onto_plane(nr)
            
            angle_ann = self.compute_angle(chorda_ann, facial_ann)
            angle_model = self.compute_angle(chorda_model, facial_model)
            matplotlib.rcParams.update({'font.size': 18})

            circ_x_model, circ_y_model, mid_angle_x_model, mid_angle_y_model, px_model, py_model = self.plot_angle_slice(chorda_plot_model, facial_plot_model, angle_model, nr)
            circ_x_ann, circ_y_ann, mid_angle_x_ann, mid_angle_y_ann, px_ann, py_ann = self.plot_angle_slice(chorda_plot_ann, facial_plot_ann, angle_ann,nr)
            Dril_x_model, Dril_y_model= np.array(self.drilling_point(angle_model,chorda_plot_model,px_model,py_model))
            Dril_x_ann, Dril_y_ann= np.array(self.drilling_point(angle_ann,chorda_plot_ann,px_ann,py_ann))
           
            c_model = chorda_plot_model
            f_model = facial_plot_model
            c_ann = chorda_plot_ann
            f_ann = facial_plot_ann
            

            plt.figure()
            plt.plot(c_model[:, 0], c_model[:, 1], linestyle = '-', color = col(9).color, label = "Model", alpha = 0.7)
            plt.plot(f_model[:, 0], f_model[:, 1], linestyle = '-', color = col(9).color, alpha = 0.7)

            plt.plot(c_ann[:, 0], c_ann[:, 1], linestyle = '-', color = col(10).color, label = "Ann", alpha = 0.7)
            plt.plot(f_ann[:, 0], f_ann[:, 1], linestyle = '-', color = col(10).color, alpha = 0.7)
            plt.scatter(Dril_x_model, Dril_y_model, marker = "x", color = col(0).color, label = "Model")
            plt.scatter(Dril_x_ann, Dril_y_ann, marker = "x", color = col(3).color, label = "Ann")

            plt.scatter(points_model[:, 0], points_model[:, 1], marker = "o", color = col(9).color, s=50)
            plt.scatter(points_ann[:, 0], points_ann[:, 1], marker = "*", color = col(10).color, s=50 )

            plt.plot(circ_x_model, circ_y_model, color = col(9).color)
            plt.text(mid_angle_x_model, mid_angle_y_model, np.round(angle_model,2), fontsize = 12)
            plt.plot(circ_x_ann, circ_y_ann, color = col(10).color)
            plt.text(mid_angle_x_ann, mid_angle_y_ann, np.round(angle_ann,2), fontsize = 12)
            plt.legend()
            plt.axis("square")
            plt.gca().set_aspect("equal")
            # plt.title(title)
            min_x = np.min(np.concatenate([f_model[:, 0],c_model[:, 0],f_ann[:, 0],c_ann[:, 0]], axis = 0))
            min_x = np.min([min_x,Dril_x_model,Dril_x_ann])
            min_y = np.min(np.concatenate([f_model[:, 1],c_model[:, 1],f_ann[:, 1],c_ann[:, 1]], axis = 0))
            min_y = np.min([min_y,Dril_y_model,Dril_y_ann])
            max_x = np.max(np.concatenate([f_model[:, 0],c_model[:, 0],f_ann[:, 0],c_ann[:, 0]], axis = 0))
            max_x = np.max([max_x,Dril_x_model,Dril_x_ann])
            max_y = np.max(np.concatenate([f_model[:, 1],c_model[:, 1],f_ann[:, 1],c_ann[:, 1]], axis = 0))
            max_y = np.max([max_y,Dril_y_model,Dril_y_ann])

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

            plt.xlim([min_x - 2, max_x + 2])
            plt.ylim([min_y - 2, max_y + 2])
            plt.xticks([])
            plt.yticks([])
            plt.show()

    
    def plot_angle_ann(self, nr, getRotation = False, Dril_x_ann = 0, Dril_y_ann = 0): 
        """
        This is the function to call. It plots the angle 
        """
        
        if getRotation: 
            self.get_points_for_plane(nr)
            chorda_plot_ann, chorda_ann, facial_plot_ann, facial_ann, points_ann, _, _,_, _, _ = self.project_onto_plane(nr)
        else: 
            self.get_points_for_plane(nr)
            chorda_plot_ann, chorda_ann, facial_plot_ann, facial_ann, points_ann, _, _, _, _, _  = self.project_onto_plane(nr)
            
            angle_ann = self.compute_angle(chorda_ann, facial_ann)
        
            matplotlib.rcParams.update({'font.size': 18})

            
            circ_x_ann, circ_y_ann, mid_angle_x_ann, mid_angle_y_ann, px_ann, py_ann = self.plot_angle_slice(chorda_plot_ann, facial_plot_ann, angle_ann,nr)
            
            if(Dril_x_ann == 0 and Dril_y_ann == 0):
                Dril_x_ann, Dril_y_ann = np.array(self.drilling_point(angle_ann,chorda_plot_ann,px_ann,py_ann))
           
            
            c_ann = chorda_plot_ann
            f_ann = facial_plot_ann
            

            plt.figure()

            plt.plot(c_ann[:, 0], c_ann[:, 1], linestyle = '-', color = col(10).color, label = "Ann", alpha = 0.7)
            plt.plot(f_ann[:, 0], f_ann[:, 1], linestyle = '-', color = col(10).color, alpha = 0.7)
            plt.scatter(Dril_x_ann, Dril_y_ann, marker = "x", color = col(3).color, label = "Ann")

            plt.scatter(points_ann[:, 0], points_ann[:, 1], marker = "*", color = col(10).color, s=50 )

            plt.plot(circ_x_ann, circ_y_ann, color = col(10).color)
            plt.text(mid_angle_x_ann, mid_angle_y_ann, np.round(angle_ann,2), fontsize = 12)
            plt.legend()
            plt.axis("square")
            plt.gca().set_aspect("equal")
            # plt.title(title)
            min_x = np.min(np.concatenate([f_ann[:, 0],c_ann[:, 0]], axis = 0))
            min_x = np.min([min_x,Dril_x_ann])
            min_y = np.min(np.concatenate([f_ann[:, 1],c_ann[:, 1]], axis = 0))
            min_y = np.min([min_y,Dril_y_ann])
            max_x = np.max(np.concatenate([f_ann[:, 0],c_ann[:, 0]], axis = 0))
            max_x = np.max([max_x,Dril_x_ann])
            max_y = np.max(np.concatenate([f_ann[:, 1],c_ann[:, 1]], axis = 0))
            max_y = np.max([max_y,Dril_y_ann])

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

            plt.xlim([min_x - 2, max_x + 2])
            plt.ylim([min_y - 2, max_y + 2])
            plt.xticks([])
            plt.yticks([])
            plt.show()

        
    def compute_all_angles(self): 
        angles = np.zeros((self.nr_image, 2))

        for nr in range(self.nr_image): 
            self.get_points_for_plane(nr)
            _, chorda_ann, _, facial_ann, _, _, chorda_model, _, facial_model, _ = self.project_onto_plane(nr) 
            angles[nr, 0], angles[nr, 1] = self.compute_angle(chorda_ann, facial_ann), self.compute_angle(chorda_model, facial_model)
        
        return angles 
