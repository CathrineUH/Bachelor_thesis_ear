from curses import keyname
from Dataloader import loadScan as ls
import SimpleITK as sitk
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
    return image[:, 80:160, 10:90],image[:, 80:160, 90:170]

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
        i = int(j[1])
        im = ls(path, i)
        im_1,im_2 = getEars(im)
        i = str(i)
        if len(i) == 1:
            sitk.WriteImage(im_1, "Data_cropped\\ear1-00"+ i +".nii.gz")
            sitk.WriteImage(im_2, "Data_cropped\\ear2-00"+ i +".nii.gz")
        elif len(i) == 2:
            sitk.WriteImage(im_1, "Data_cropped\\ear1-0"+ i +".nii.gz")
            sitk.WriteImage(im_2, "Data_cropped\\ear2-0"+ i +".nii.gz")
        elif len(i) == 3:
            sitk.WriteImage(im_1, "Data_cropped\\ear1-"+ i +".nii.gz")
            sitk.WriteImage(im_2, "Data_cropped\\ear2-"+ i +".nii.gz")


        

        




