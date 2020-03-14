#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/get_pet_labels.py
#                                                                             
# PROGRAMMER: Dina TAKLIT
# DATE CREATED:  26/12/2019                                
# REVISED DATE: 14/03/2020
# PURPOSE: Create the function get_pet_labels that creates the pet labels from 
#          the image's filename. This function inputs: 
#           - The Image Folder as image_dir within get_pet_labels function and 
#             as in_arg.dir for the function call within the main function. 
#          This function creates and returns the results dictionary as results_dic
#          within get_pet_labels function and as results within main. 
#          The results_dic dictionary has a 'key' that's the image filename and
#          a 'value' that's a list. This list will contain the following item
#          at index 0 : pet image label (string).
#
##
# Imports python modules
from os import listdir
from os import path
# Done 2: Define get_pet_labels function below please be certain to replace None
#       in the return statement with results_dic dictionary that you create 
#       with this function
# 
def get_pet_labels(image_dir):
    """
    Creates a dictionary of pet labels (results_dic) based upon the filenames 
    of the image files. These pet image labels are used to check the accuracy 
    of the labels that are returned by the classifier function, since the 
    filenames of the images contain the true identity of the pet in the image.
    Be sure to format the pet labels so that they are in all lower case letters
    and with leading and trailing whitespace characters stripped from them.
    (ex. filename = 'Boston_terrier_02259.jpg' Pet label = 'boston terrier')
    Parameters:
     image_dir - The (full) path to the folder of images that are to be
                 classified by the classifier function (string)
    Returns:
      results_dic - Dictionary with 'key' as image filename and 'value' as a 
      List. The list contains for following item:
         index 0 = pet image label (string)
    """
    # Replace None with the results_dic dictionary that you created with this
    # function
    
    # Retrieve the filenames from folder pet_images/
    in_files = listdir(image_dir)
    # Creates empty dictionary for the results (pet labels, etc.)
    results_dic = dict()
    # Processes through each file in the directory, extracting only the words
    # of the file that contain the pet image label
    for idx in range(0, len(in_files), 1):
        # Skips file if starts with . (like .DS_Store of Mac OSX) because it 
        # isn't an pet image file
        if in_files[idx][0] != ".":
            if in_files[idx] not in results_dic:
                # Creates temporary label variable to hold pet label name extracted 
                pet_label = ""         
                # Sets pet_image variable to a filename after removing the file extension 
                pet_image = path.splitext(in_files[idx])[0]
                # Sets string to lower case letters
                low_pet_image = pet_image.lower()             
                word_list_pet_image = low_pet_image.split("_")
                # Create pet_name starting as empty string
                pet_name = ""
                # Loops to check if word in pet name is only
                # alphabetic characters - if true append word
                # to pet_name separated by trailing space 
                for word in word_list_pet_image:                  
                    if word.isalpha():                        
                        pet_name += word + " "

                # Strip off starting/trailing whitespace characters 
                pet_label = pet_name.strip()
                
                 # Add pet_label as value of the file name key
                results_dic[in_files[idx]] = [pet_label]
            else:
                print("** Warning: Duplicate files exist in directory:",in_files[idx])
                
    return results_dic
