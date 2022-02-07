from Dataloader import loadScan
import itk
import numpy as np
path = "C:\\Users\\There\\OneDrive\\Uddannelse\\Universitet\\6 semester\\Bachelorprojekt\\Data"
savePath = "C:\\Users\\There\\OneDrive\\Dokumenter\\GitHub\\Bachelor-thesis-ear\\Data_cropped"

for i in range(1, 109+1): 
    if i == 16 or i == 78: 
        continue 
    im = loadScan(path, i)
    if im.shape == (160, 512, 392): 
        im_save = im[:, 230:330, 220:320]
    elif im.shape == (128, 256, 192): 
        im_save = im[:, 90:190, 80:180]
    
    if len(str(i)) == 1: 


