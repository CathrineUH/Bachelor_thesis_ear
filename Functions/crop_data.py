from .Dataloader import *
import SimpleITK as sitk
import numpy as np
import shutil as sh


def getEars(image,nr=80):
    """
    The function crops the image to get the ears(left and rigth)
    Input: 
        image: The image to crop 
    Output: 
        im: 3D cropped image 
    """
    if nr==110:
        return image[50:160, 40:150, 40:150], image[0:110, 40:150, 40:150]
    else:
        return image[80:160, 70:150, 40:120], image[10:90, 70:150, 40:120]

def saveCropImage(path,name_of_folder,nr = 80):
    """
    The function saves all the cropped image to the folder Data_cropped 
    
    Input: 
        path: path to file location 

    Output: 
        Files to the folder of the cropped images
    """
    files = getFiles(path)
    for j in files:
        i = int(j[1:5])
        scan = int(j[6])
        im = loadImageFromFile(path,j)
        im_1,im_2 = getEars(im,nr)
        i = str(i)
        scan = str(scan)
        if(len(i)==1):
            sitk.WriteImage(im_1, name_of_folder+"\\ear1-P000"+ i +"_"+ scan +".nii.gz")
            sitk.WriteImage(im_2, name_of_folder+"\\ear2-P000"+ i +"_"+ scan +".nii.gz")
        elif len(i)==2:
            sitk.WriteImage(im_1, name_of_folder+"\\ear1-P00"+ i +"_"+ scan +".nii.gz")
            sitk.WriteImage(im_2, name_of_folder+"\\ear2-P00"+ i +"_"+ scan +".nii.gz")
        elif len(i)==3:
            sitk.WriteImage(im_1, name_of_folder+"\\ear1-P0"+ i +"_"+ scan +".nii.gz")
            sitk.WriteImage(im_2, name_of_folder+"\\ear2-P0"+ i +"_"+ scan +".nii.gz")
        else:
            sitk.WriteImage(im_1, name_of_folder+"\\ear1-P"+ i +"_"+ scan +".nii.gz")
            sitk.WriteImage(im_2, name_of_folder+"\\ear2-P"+ i +"_"+ scan +".nii.gz")
    
def moveImage(data_path,fileName,path_new,newFileName):
    sh.move(data_path+"\\"+fileName,path_new+"\\"+newFileName)

def copyImage(data_path,FileName,newPath):
    sh.copy(data_path+"\\"+FileName,newPath,follow_symlinks=True)


def flipImage(path):
    """
    The function flips and saves all the images in that path 
    
    Input: 
        path: path to file locations

    Output: 
        Saves the flip image in the same path
    """
    files = getFiles(path)
    for j in files:
        if(j[3]==str(1)):
            im = loadImageFromFile(path,j)
            flipped_img = sitk.Flip(im, [True, False, False])
            sitk.WriteImage(flipped_img, path +"\\"+ j)
   

        

        




