from Dataloader import loadScan as ls
import SimpleITK as sitk
def resampleImage(path,image):
    '''

    '''
    im =  ls(path,1)
    if(image.GetSize() == im.GetSize()):
        return image
    else:
        im_temp = sitk.Resample(image,im)
    return im_temp

def getEars(image):
    return image[225:325,70:170,:],image[230:330,220:320,:]

def saveCropImage(path):
    for i in range(1,(109+1)):
        if i == 16 or i == 78:
            continue
        im = resampleImage(path,ls(path,i))
        im_1,im_2 = getEars(im)
        i = str(i)
        if len(i) == 1:
            sitk.WriteImage(im_1, "Data_cropped\\ear1-00"+ i +".mha")
            sitk.WriteImage(im_2, "Data_cropped\\ear1-00"+ i +".mha")
        elif len(i) == 2:
            sitk.WriteImage(im_1, "Data_cropped\\ear1-0"+ i +".mha")
            sitk.WriteImage(im_2, "Data_cropped\\ear1-0"+ i +".mha")
        elif len(i) == 3:
            sitk.WriteImage(im_1, "Data_cropped\\ear1-"+ i +".mha")
            sitk.WriteImage(im_2, "Data_cropped\\ear1-"+ i +".mha")

        

        




