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
    return sitk.GetArrayFromImage(im)

