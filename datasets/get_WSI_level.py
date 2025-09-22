import openslide
import numpy as np
import cv2

'''
You can obtain the corresponding magnification from the slide scanning metadata properties using the code below. 
The slide contain the magnification and other related metadata properties, which can be accessed through slide.properties using the openslide library. 
This includes:
Magnification (e.g., aperio.AppMag or openslide.mpp-x)
Dimensions of the slide (e.g., width and height)
Size information for different levels (e.g., slide.level_dimensions)

'''


def display_wsi_info_and_level(wsi_path, level):
    # Open the WSI file
    slide = openslide.open_slide(wsi_path)
    # Get the dimensions of the WSI
    width, height = slide.dimensions
    print("WSI Dimensions: {}x{}".format(width, height))

    # Get the number of levels in the WSI
    num_levels = slide.level_count
    dimensions = slide.level_dimensions
    print("WSI Level Count:", num_levels)
    print("WSI Level Details:", dimensions)
    
    # Get WSI metadata and read related information
    if 'aperio.AppMag' in slide.properties.keys():
        level_0_magnification = int(slide.properties['aperio.AppMag'])
    elif 'openslide.mpp-x' in slide.properties.keys():
        level_0_magnification = 40 if int(np.floor(float(slide.properties['openslide.mpp-x']) * 10)) == 2 else 20
    else:
        level_0_magnification = 40
        
    # Output the magnification corresponding to level 0
    print("Magnification for level_0:", level_0_magnification)
    
    magnification = 10
    # Calculate the downsampling rate based on the level 0 magnification and the desired magnification, 
    # e.g., if level 0 has a magnification of 40, and we want 5x, then the downsampling rate is 40/5=8
    downsample = level_0_magnification / 5
    # Get the level corresponding to the closest magnification for this downsampling rate
    level = slide.get_best_level_for_downsample(downsample)
    print("{}x magnification corresponds to Level_{}".format(magnification, level))

# Specify the WSI file path and selected level
wsi_path = r'./dataset/BRCA/BRCA-slides/TCGA-5L-AAT1-01Z-00-DX1.F3449A5B-2AC4-4ED7-BF44-4C8946CDB47D.svs'
selected_level = 2

# Call the function to display
display_wsi_info_and_level(wsi_path, selected_level)
