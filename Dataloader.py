import numpy as np 
import SimpleITK as sitk
import os
def reName(path):
    """
    The function reName the data
    Input: 
        path: path to data 
    """
    os.chdir(path)
    files = [f for f in os.listdir('.') if os.path.isfile(f)]
    count = 0
    for f in files:
        newName = "P"+ str(count)+"_T1w.nii.gz"
        os.rename(f, newName)
        count+=1


def loadTraining(path): 
    """ 
    The function loads all training scans 
    
    Input: 
        path: path to data 

    Output: 
        imTest: 3D array with all traing scans 
    """
    trainingIdx = np.arange(2, 7)   # Just examples for now  
    dim = [512, 512, 512]       # or whatever dimensions we end with 
    imTraining = np.zeros([dim[0], dim[1], dim[2] * len(trainingIdx)])

    count = 0 
    for i in trainingIdx: 
        imTraining[:,:,count * dim[2]:(count + 1) * dim[2]] = loadScan(path, i)
        count += 1 
    
    return imTraining 


def loadTest(path): 
    """ 
    The function loads all test scans 
    
    Input: 
        path: path to data 

    Output: 
        imTest: 3D array with all test scans 
    """
    testIdx = np.arange(2, 7)   # Just examples for now  
    dim = [512, 512, 512]       # or whatever dimensions we end with 
    imTest = np.zeros([dim[0], dim[1], dim[2] * len(testIdx)])

    count = 0 
    for i in testIdx: 
        imTest[:,:,count * dim[2]:(count + 1) * dim[2]] = loadScan(path, i)
        count += 1 
    
    return imTest 


def loadScan(path, i): 
    """
    The function loads a single 3D image
    
    Input: 
        path: path to file location 
        i: index of image (cannot be 16 or 78)

    Output: 
        im: 3D image
    """
    
    i = str(i)
        
    im = sitk.ReadImage(path +"\\P"+ i+"_T1w.nii.gz")
    
    return sitk.GetArrayFromImage(im)

def loadCropImage(ear,i):
    """
    The function loads a single 3D cropped image of the ear
    
    Input: 
        ear: either 1 or 2 depends on which ear to load
        i: index of image (cannot be 16 or 78)
        
    Output: 
        im: 3D cropped image of the ear

    """
    i = str(i)
    if len(i) == 1:
        im = sitk.ReadImage("Data_cropped\\ear" + str(ear) +"-00" + i + ".nii.gz")
    elif len(i) == 2:
        im = sitk.ReadImage("Data_cropped\\ear" + str(ear) +"-0" + i + ".nii.gz")
    elif len(i) == 3:
        im = sitk.ReadImage("Data_cropped\\ear" + str(ear) + "-" + i + ".nii.gz")
    return sitk.GetArrayFromImage(im)