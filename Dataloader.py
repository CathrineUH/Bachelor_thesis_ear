import SimpleITK as sitk
import os

def getFiles(path):
    """
    The function takes a path and return the names of all files in that path.
    
    Input: 
        path: path to files location.

    Output: 
        files: All file names in the path.
    """
    files = os.listdir(path)
    return files

def loadImageFromFile(path,file_name):
    """
    The function takes a filename of an image and returns the image.
    
    Input: 
        file_name: filename of the image.

    Output: 
        im: Images as a numpy array.
    """
    im = sitk.ReadImage(path +"\\"+ file_name)
    return im


def loadTraining(path):
    la = 100
    return la 

def changetxtfile(path,filename,txtfile,nr):
    f = open(path +"\\"+filename, "r+")
    l = f.readlines()
    with open(txtfile,'a') as file:
        for i in l:
            first = i[0:24]
            last = i[24:]
            temp = first + nr +"/"+last
            file.write(temp)