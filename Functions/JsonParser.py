import SimpleITK as sitk
import json
import numpy as np
from .Dataloader import*
# Tips:
# https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks/blob/master/Utilities/intro_animation.py
# https://stackoverflow.com/questions/30237024/operate-on-slices-of-3d-image-in-simpleitk-and-create-new-3d-image

# https://slicer.readthedocs.io/en/latest/developer_guide/script_repository.html
# returns JSON object as a dictionary

def read_ear_landmarks(path_json,path_data,full_name):
    f = open(path_json+"\\"+full_name)
    # file_name = path_data +"\\"+ full_name[0:12]
    file_name = path_data +"\\"+ full_name[0:7]+".nii.gz"
    data = json.load(f)
    t = data['markups'][0]['controlPoints']
    Dict = {'C': 0, 'A': 1, 'R': 2, 'M': 3, 'T': 4, 'B': 5}
    Coordinat = np.zeros((6,3))
    for lm in range(6):
        Coordinat[Dict[t[lm]['label']]] = t[lm]['position']
    f.close()
    Coordinat_change = change_coordinate_system(Coordinat, file_name)
    return Coordinat_change

def change_coordinate_system(positions, file_name):
    img = sitk.ReadImage(file_name)
    coordinates = np.zeros((6, 3))
    for i in range(6):
        coordinates[i] = img.TransformPhysicalPointToIndex(positions[i])
    return coordinates 


def write_all_landmarks_to_txt(files, path_data, path_out, path_json):
   for i in files:
       coordinates = read_ear_landmarks(path_json,path_data,i)
       file_name = path_out +"\\"+ i[0:7]+".txt"
       np.savetxt(file_name,coordinates,fmt='%.f')
