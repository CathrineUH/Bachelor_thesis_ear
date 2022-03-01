from Dataloader import loadScan as ls
from Dataloader import loadImageFromFile as lf
from Dataloader import getFiles as gf
import SimpleITK as sitk
import numpy as np
import os 

def resampleImage(path,image):
    """
    The function resamples a 3D image to the dimension of image 1
    
    Input: 
        path: path to file location 
        i: index of image (cannot be 16 or 78)

    Output: 
        im: 3D resampled image 
    """
    im =  ls(path,1)
    if(image.GetSize() == im.GetSize()):
        return image
    else:
        im_resam = sitk.Resample(image,im)
    return im_resam

def getEars(image):
    """
    The function crops the image to get the ears(left and rigth)
    Input: 
        image: The image to crop 
    Output: 
        im: 3D cropped image 
    """
    return image[50:130, 80:160, 10:90],image[50:130, 80:160, 90:170]

def saveCropImage(path):
    """
    The function saves all the cropped image to the folder Data_cropped 
    
    Input: 
        path: path to file location 

    Output: 
        Files to the folder Data_cropped of the cropped images
    """
    os.chdir(path)
    files = [f for f in os.listdir('.') if os.path.isfile(f)]
    for j in files:
        i = int(j[1:5])
        scan = int(j[6])
        im = ls(path, i,scan)
        im_1,im_2 = getEars(im)
        im_1 = sitk.GetImageFromArray(im_1)
        im_2 = sitk.GetImageFromArray(im_2)
        i = str(i)
        scan = str(scan)
        if(len(i)==1):
            sitk.WriteImage(im_1, "../Data_cropped\\ear1-P000"+ i +"_"+ scan +".nii.gz")
            sitk.WriteImage(im_2, "../Data_cropped\\ear2-P000"+ i +"_"+ scan +".nii.gz")
        elif len(i)==2:
            sitk.WriteImage(im_1, "../Data_cropped\\ear1-P00"+ i +"_"+ scan +".nii.gz")
            sitk.WriteImage(im_2, "../Data_cropped\\ear2-P00"+ i +"_"+ scan +".nii.gz")
        elif len(i)==3:
            sitk.WriteImage(im_1, "../Data_cropped\\ear1-P0"+ i +"_"+ scan +".nii.gz")
            sitk.WriteImage(im_2, "../Data_cropped\\ear2-P0"+ i +"_"+ scan +".nii.gz")
        else:
            sitk.WriteImage(im_1, "../Data_cropped\\ear1-P"+ i +"_"+ scan +".nii.gz")
            sitk.WriteImage(im_2, "../Data_cropped\\ear2-P"+ i +"_"+ scan +".nii.gz")
       
def flipImage(path):
    files = gf(path)
    nr = 0
    for j in files:
        if(j[3]==str(1)):
            im = lf(path,nr)
            flipped_img = np.flip(im,axis=2)
            flipped_img  = sitk.GetImageFromArray(flipped_img)
            sitk.WriteImage(flipped_img, j)
        nr+=1    
   

        

        




