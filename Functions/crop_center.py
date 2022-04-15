import SimpleITK as sitk
import json
import numpy as np
from Dataloader import getFiles 

def bound_landmarks(full_name):
    f = open(full_name)

    data = json.load(f)
    t = data['markups'][0]['controlPoints']
    Dict = {'C': 0, 'A': 1, 'R': 2, 'M': 3, 'T': 4, 'B': 5}
    Coordinat = np.zeros((6,3))
    for lm in range(6):
        Coordinat[Dict[t[lm]['label']]] = t[lm]['position']

    first = True
    xmin = xmax = ymax = ymin = zmin = zmax = 0
    for lm in Coordinat:
        pos = lm
        x = pos[0]
        y = -pos[1]
        z =  pos[2]
        if first:
            first = False
            xmin = xmax = x
            ymin = ymax = y
            zmin = zmax = z
        else:
            xmin = min(xmin, x)
            xmax = max(xmax, x)
            ymin = min(ymin, y)
            ymax = max(ymax, y)
            zmin = min(zmin, z)
            zmax = max(zmax, z)

    f.close()
    bounds = [xmin, xmax, ymin, ymax, zmin, zmax]
    return bounds


def crop_ear_roi(full_name,bounds, resampled_name):
    image = sitk.ReadImage(full_name)
    ear = full_name[6]
    # Create the sampled image with same direction
    direction = image.GetDirection()

    # Desired voxel spacing for new image
    new_spacing = [1, 1, 1]
    # new_spacing = [0.5, 0.5, 0.5]

    # adjust bounds
    # Add some millimeters on each side
    padding = 40 

    # in slice size (max of x length and y length plus padding in both sides)
    max_l = max(bounds[1]-bounds[0], bounds[3]-bounds[2]) + 2 * padding
    nvox_xy = int(max_l / new_spacing[0] + 1)
    new_l_xy = nvox_xy * new_spacing[0] 
    nvox_z = int((bounds[5] - bounds[4] + 2*padding)/ new_spacing[2])
    print('Size of new volume: ', nvox_xy, nvox_xy, nvox_z, ' voxels')

    # Compute new origin from center of old bounds
    if (ear =="R"):
        new_origin_x = (bounds[1] + bounds[0]) / 2 - 6 / 10 * new_l_xy 
        new_origin_y = (bounds[3] + bounds[2]) / 2 - 1 / 6 * new_l_xy 
        new_origin_z = (bounds[5] + bounds[4]) / 2 - nvox_z * new_spacing[2] / 2
    else:
        new_origin_x = (bounds[1] + bounds[0]) / 2 + 4 / 10 * new_l_xy
        new_origin_y = (bounds[3] + bounds[2]) / 2 + 1 / 12 * new_l_xy
        new_origin_z = (bounds[5] + bounds[4]) / 2 - nvox_z * new_spacing[2] / 2



    # Size in number of voxels per side
    # new_size = [100, 100, 100]
    new_size = [nvox_xy, nvox_xy, nvox_z]
    new_image = sitk.Image(new_size, image.GetPixelIDValue())
    new_image.SetOrigin([new_origin_x, new_origin_y, new_origin_z])
    new_image.SetSpacing(new_spacing)
    new_image.SetDirection(direction)

    # Make translation with no offset, since sitk.Resample needs this arg.
    translation = sitk.TranslationTransform(3)
    translation.SetOffset((0, 0, 0))

    interpolator = sitk.sitkLinear
    # Create final reasampled image
    resampled_image = sitk.Resample(image, new_image, translation, interpolator)

    sitk.WriteImage(resampled_image, resampled_name)


if __name__ == '__main__':
    full_name = getFiles("Annotations_good")
    # for i in range(1, len(full_name)):
    #     full_name_image = "Data_good/" + full_name[i][0:5] + ".nii.gz"
    #     resampled_name = "Data_crop/" + full_name[i][0:7]+".nii.gz"
    #     bds = bound_landmarks(full_name = "Annotations_good/"+full_name[i])
    #     crop_ear_roi(full_name_image,bds, resampled_name)
    full_name = full_name[0]
    full_name_image = "Data_good/" + full_name[0:5] + ".nii.gz"
    resampled_name = full_name[0:7]+".nii.gz"
    bds = bound_landmarks(full_name = "Annotations_good/"+full_name)
    print(bds)
    crop_ear_roi(full_name_image,bds, resampled_name)

