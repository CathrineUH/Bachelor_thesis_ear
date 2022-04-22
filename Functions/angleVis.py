# link: https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib
from .computeAngle import * 


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
        Compute translatio of facial 
        """

        length = np.linalg.norm(points_rot[2][0:2] - points_rot[4][0:2])
        l = np.reshape(np.array(np.linspace(-length/2 - 5, length/2 + 5, 200)), (200, 1))
        mu = np.mean(points_rot[2:], axis = 0)
        facial_plot = facial * l + mu 
        return facial_plot

        
    def rotate(self, points_plane, points, nr):
        """
        Rotates the normal vector of the plane to be the [0, 0, 1] such that 
        the plane is the xy-plane 
        """
        # Helper function # 
        chorda = points_plane[0]
        facial_nerve = points_plane[1]
        n = points_plane[2]
        n /= np.linalg.norm(n)

        # rotate normal vector to z 
        v = np.cross(n, np.array([0, 0, 1]))
        s = np.linalg.norm(v)
        c = np.array([0, 0, 1]) @ n 
        skew_v = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.identity(3) + skew_v + skew_v @ skew_v * (1 - c) / (s ** 2)

        chorda_rot = R @ chorda
        facial_rot = R @ facial_nerve
        p_rot = (R @ points[nr].T).T 

        chorda_rot = chorda_rot / np.linalg.norm(chorda_rot) 
        facial_rot = facial_rot / np.linalg.norm(facial_rot) 

        chorda_plot, translation = self.get_translation_chorda(p_rot, np.array([chorda_rot[0], chorda_rot[1]]))
        p_rot[:, 0:2] -= translation 
        facial_plot = self.get_translation_facial(p_rot, facial_rot)

        return chorda_plot, chorda_rot, facial_plot, facial_rot, p_rot 

    def project_onto_plane(self, nr): 
        chorda_plot_ann, chorda_ann, facial_plot_ann,  facial_ann, points_ann = self.rotate(self.plane_ann, self.coor_ann, nr)
        chorda_plot_model, chorda_model, facial_plot_model, facial_model, points_model = self.rotate(self.plane_model, self.coor_model, nr)

        return chorda_plot_ann, chorda_ann, facial_plot_ann, facial_ann, points_ann, chorda_plot_model, chorda_model, facial_plot_model, facial_model, points_model 

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

    def plot_angle_in_plane(self, nr): 
        """
        This is the function to call. It plots the angle 
        """

        self.get_points_for_plane(nr)
        chorda_plot_ann, chorda_ann, facial_plot_ann, facial_ann, points_ann, chorda_plot_model, chorda_model, facial_plot_model, facial_model, points_model   = self.project_onto_plane(nr)
        
        angle_ann = self.compute_angle(chorda_ann, facial_ann)
        angle_model = self.compute_angle(chorda_model, facial_model)
        matplotlib.rcParams.update({'font.size': 18})
        def plot(chorda, facial, p_rot, title, angle): 
            c = chorda 
            f = facial 
            plt.figure()
            plt.plot(c[:, 0], c[:, 1], linestyle = '-', color = "b", label = "chorda nerve")
            plt.plot(f[:, 0], f[:, 1], linestyle = '-', color = "r", label = "facial nerve")
            marker = ["o", "o", "v", "v", "v"]
            color = ["g", "g", "k", "k", "k"]
            label = ["C", "A", "R", "M", "T"]
            for i in range(5): 
                plt.scatter(p_rot[i, 0], p_rot[i, 1], marker = marker[i], color = color[i], label = label[i])
            plt.legend()
            plt.axis("square")
            plt.gca().set_aspect("equal")
            plt.title("From " + title + ". Angle = " + str(angle))
            # plt.xlim([np.min(p_rot[:, 0]) + 2, np.max(p_rot[:, 0] - 2)])
            plt.ylim([np.min(p_rot[:, 1]) - 2, np.max(p_rot[:, 1] + 2)])
            plt.grid()
            plt.show()

        # plot time
        plot(chorda_plot_model, facial_plot_model, points_model,  "model", np.round(angle_model, 2))
        plot(chorda_plot_ann, facial_plot_ann, points_ann, "landmarks", np.round(angle_ann, 2))

    def compute_all_angles(self): 
        angles = np.zeros((self.nr_image, 2))

        for nr in range(self.nr_image): 
            self.get_points_for_plane(nr)
            chorda_ann, facial_ann, _,  chorda_model, facial_model, _ = self.project_onto_plane()
            angles[nr, 0], angles[nr, 1] = self.compute_angle(chorda_ann, facial_ann), self.compute_angle(chorda_model, facial_model)
        
        return angles 
