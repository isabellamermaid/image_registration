
import os
from os import listdir
from skimage import io
from skimage.color import rgb2gray
from skimage.data import stereo_motorcycle
from skimage.registration import phase_cross_correlation
from skimage.transform import AffineTransform, warp
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from skimage.util import img_as_ubyte
from skimage.util import img_as_uint
from skimage import exposure

    
def parseArgs():
    parser = argparse.ArgumentParser(description="Image registration parameters")
    parser.add_argument('-path_to_round1', default= "/Users/isabelmo/Downloads/registration/testdata/round1")
    parser.add_argument('-path_to_round2', default= "/Users/isabelmo/Downloads/registration/testdata/round2")
    parser.add_argument('-outputs', default= "/Users/isabelmo/Downloads/registration/testdata/outputs")
    args = parser.parse_args()
    return args

def cal_phase_correlate(fixed, moving):
    # calculate phase correlations
    shift, error, phasediff = phase_cross_correlation(fixed, moving)
    shift = [shift[1], shift[0]]
    shift = np.array(shift)
    shift = shift*-1
    print(shift, error, phasediff)
    return shift

def transform_phase_correlate(moving, shift):
    # calculate correction transform
    transform = AffineTransform(translation=shift)
    # apply it to round 2 image
    return warp(moving, transform)

def main():
    # load all image names into lists
    args = parseArgs()
    path_to_round1 = args.path_to_round1
    path_to_round2 = args.path_to_round2
    ls_imgs1_names = os.listdir(path_to_round1)
    ls_imgs2_names = os.listdir(path_to_round2)
    path_to_outputs = args.outputs 
    Path(path_to_outputs + "/regs").mkdir(parents=True, exist_ok=True)

    image_r1_names = []

    # Create tables for each experiment
    for image_r1 in ls_imgs1_names:
        # For each DAPI image in round 1
        if '_DAPI_ORG' in image_r1:
            # add round 1 path list
            image_r1_names.append(image_r1)
            # store sample id
            spot_str = image_r1[-7:-4]

            image_r2_names = []
            # Find corresponding channel images from round 2
            for image_r2 in ls_imgs2_names:
                if f'roi{spot_str}' in image_r2:
                    image_r2_names.append(image_r2)

            # for dapi from 2nd round calculate transformation
            for image_r2 in image_r2_names:
                if f'_DAPI_ORG_roi{spot_str}' in image_r2:
                    # open round 1 & round 2 DAPIs
                    fixed = io.imread(f"{path_to_round1}/{image_r1}")
                    moving = io.imread(f"{path_to_round2}/{image_r2}")
                    
                    # calculate transformation
                    shift = cal_phase_correlate(fixed, moving)

                    # transform dapi
                    moving = transform_phase_correlate(moving.astype(np.uint8), shift)
                    #moving = moving.astype(np.uint8)
                    #fixed = fixed.astype(np.uint8)
                    

                    # create tumbnails
                    r1image = np.expand_dims(fixed, axis=-1)
                    r2image = np.expand_dims(moving, axis=-1)
                    thumbnail = np.concatenate([r1image, r2image, r2image], axis=-1)
                    # print(thumbnail.shape)
                    # exit()
                    io.imsave(f'{path_to_outputs}/DAPIaftertransform_roi{spot_str}.jpg', thumbnail)
                    break

            transformed_images = []
            # transform all round 2 images
            for image_r2 in image_r2_names:
                # open image
                moving = io.imread(f"{path_to_round2}/{image_r2}")
                #moving = moving.astype(np.uint8)
                #print(moving.dtype)

                # before

                # create tumbnails
                r1image = np.expand_dims(fixed, axis=-1)
                r2image = np.expand_dims(moving, axis=-1)
                r3image = np.zeros_like(r2image)
                thumbnail = np.concatenate([r1image, r2image, r3image], axis=-1)
                # print(thumbnail.shape)
                # exit()
                io.imsave(f'{path_to_outputs}/before_{image_r2}.jpg', thumbnail)

                # transform image
                #img_as_ubyte(moving)
                moving = transform_phase_correlate(moving, shift) #astype(np.uint8)
                moving = img_as_uint(moving)
                #image = exposure.rescale_intensity(moving, in_range='uint8')
                transformed_images.append(moving)
                #moving = moving.astype(np.uint8)
                print(moving.dtype)

                # after

                # create tumbnails
                r1image = np.expand_dims(fixed, axis=-1)
                r2image = np.expand_dims(moving, axis=-1)
                thumbnail = np.concatenate([r1image, r2image, r3image], axis=-1)
                # print(thumbnail.shape)
                # exit()
                io.imsave(f'{path_to_outputs}/after_{image_r2}.jpg', thumbnail)
                
                
                io.imsave(f'{path_to_outputs}/regs/{image_r2}', moving) #astype(np.uint16))
                #io.imsave(f'{path_to_outputs}/regs/{image_r2}', moving.astype(np.uint8))
                
                print(fixed.dtype)
                

if __name__ == "__main__":
    main()
    