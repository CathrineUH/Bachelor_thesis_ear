import SimpleITK as sitk
import json
import numpy as np
from sqlalchemy import true

def compute_center_of_mass(path_json,full_name):
    f = open(path_json+"\\"+full_name)

    data = json.load(f)
    t = data['markups'][0]['controlPoints']
    Dict = {'C': 0, 'A': 1, 'R': 2, 'M': 3, 'T': 4, 'B': 5}
    Coordinat = np.zeros((6,3))
    for lm in range(6):
        Coordinat[Dict[t[lm]['label']]] = t[lm]['position']

    xmean = ymean = zmean = 0
    for lm in Coordinat:
        pos = lm
        x = pos[0]
        y = pos[1]
        z = pos[2]
        xmean += x
        ymean += y
        zmean += z
    
    xmean /= 6
    ymean /= 6
    zmean /= 6
    f.close()
    bounds = [xmean, ymean, zmean]
    return bounds


def crop_roi(full_name,resampled_name,bounds,radius):
    image = sitk.ReadImage(full_name)

    # Create the sampled image with same direction
    direction = image.GetDirection()

    # Desired voxel spacing for new image
    new_spacing = [1,1,1]

    # in slice size (max of x length and y length plus padding in both sides)
    max_l = max(bounds[0],bounds[1]) + 2 * radius
    nvox_xy = int(max_l / new_spacing[0] + 1)
    new_l_xy = nvox_xy * new_spacing[0]
    nvox_z = int((bounds[2] + 2 * radius) / new_spacing[2])

    # Compute new origin from center of old bounds
    new_origin_x = (bounds[0])- new_l_xy 
    new_origin_y = (bounds[1]) - new_l_xy 
    new_origin_z = (bounds[2]) - nvox_z * new_spacing[2]

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

