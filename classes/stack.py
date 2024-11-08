# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 14:42:39 2022

@author: florent.brondolo
"""
import os
import numpy as np
import random
from code.Backup.ProgressBar import ProgressBar
import tifffile as tiff
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Stack:

    def __repr__(self):
        return 'Stack'

    # def __init__(self, name, isSegmented, maskPreprocessing, isMulticlass,
    #              width=None, height=None, channel=1, path=None):
    def __init__(self, **kwargs):
        """
            Initializes the Stack instance with optional parameters.

            Args:
                name (str): Name of the image stack.
                isSegmented (bool): Indicates if the stack is segmented. Default is False.
                imageType (int): Image type (0 for grayscale, 1 for RGB). Default is 0.
                stackImage (array): Array of stack images.
                stackLabel (array): Array of stack labels.
                path (str): Path to the image directory.
                channel (int): Number of image channels.
                width (int): Image width.
                height (int): Image height.
                numSlice (int): Number of slices to select.
                selectedFiles (list): List of selected files.

            Sets attributes, validates inputs, and initializes the slice list.
        """
        self.numClass = None
        self.listSlice = []
        self.listSlicePath = []

        if "name" in kwargs:
            self.name = kwargs['name']

        if "isSegmented" in kwargs:
            self.isSegmented = kwargs['isSegmented']
        else:
            self.isSegmented = False

        if "imageType" in kwargs:
            self.imageType = kwargs['imageType']
        else:
            self.imageType = 0  # We assume that it's a grayscale scanner img if not specified
        '''
        if "isMulticlass" in kwargs:
            self.isMulticlass = kwargs['isMulticlass']
        else:
            self.isMulticlass = False
        '''

        if "stackImage" in kwargs:
            self.stackImage = kwargs['stackImage']
        else:
            self.stackImage = None

        if "stackLabel" in kwargs:
            self.stackLabel = kwargs['stackLabel']

        if "path" in kwargs:
            self.path = kwargs['path']
        else:
            self.path = None

        if "channel" in kwargs:
            self.channel = kwargs['channel']
        '''
        else:
            self.channel = 1
        '''

        if ("width" in kwargs) and ("height" in kwargs):
            self.width = kwargs['width']
            self.height = kwargs['height']
        else:
            # Check which length is the shortest, then applies it to both height and width
            # This is done in order to keep a square volume
            width_tmp = cv2.imread(self.path + '//' + next(os.walk(self.path))[2][0]).shape[1]
            height_tmp = cv2.imread(self.path + '//' + next(os.walk(self.path))[2][0]).shape[0]
            if width_tmp < height_tmp:
                self.width = width_tmp
                self.height = width_tmp
            elif width_tmp > height_tmp:
                self.width = height_tmp
                self.height = height_tmp
            else:
                self.width = width_tmp
                self.height = height_tmp

        # path setting
        # if a path is provided: next
        # if no path is provided: either a list of slice is provided (used in the Util class), or it'll raise an error
        # if a path and a list of slice are provided : an error will be raised
        if self.path is None:
            if "isSliceListSupplied" not in kwargs:
                raise Exception(repr(self) + 'class - a folder path or a list of slices must be provided')
            else:  # isSliceListSupplied is in kwargs
                self.isSliceListSupplied = kwargs['isSliceListSupplied']
        else:  # path is in kwargs (path is provided)
            if "isSliceListSupplied" not in kwargs:
                self.isSliceListSupplied = False
                # Get the total amount of slices present in the CT scan
                # num_total_slice = len([name for name in os.listdir(self.path)])
            else:  # isSliceListSupplied is in kwargs
                raise Exception(repr(self) + 'class - path and isSliceListSupplied cannot be set at the same time')

        '''
        if ("numSlice" in kwargs) and ("selectedFiles" not in kwargs): 
        #If a number of slices is specified, but no list of preselected files is
            try:
              #Check if a number of slice to pick has been specified
              #If yes : n slices are randomly selected (no duplicate)
              #If not : all slices are selected
              if kwargs['numSlice'] == None:
                self.numSlice = numTotalSlice
                self.selectedFiles = range(numTotalSlice) 
              else:
                if self.numSlice <= numTotalSlice: # Check that numSlice is < to the total number of available slices  
                    self.numSlice = kwargs['numSlice']
                    self.selectedFiles = random.sample(range(0,numTotalSlice-1), self.numSlice) 
                else:
                    raise Exception(repr(self) + ' class - numSlice should be less or equal to the number 
                    of slice in the folder provided')    
            except:
                raise Exception(repr(self) + ' class - if numSlice is provided and selectedFiles is not : 
                the path must me provided')       
        elif ("numSlice" not in kwargs) and ("selectedFiles" not in kwargs): #If all slices are selected, 
        and not preselected files are passed on
            if self.isSegmented: #if the stack processed is a mask, then selectedFiles should have been provided
                raise Exception(repr(self) + ' class - if the stack is a mask stack then the associated selectedFiles 
                from the corresponding image stack must be provided')    
            else:   
              try:
                  self.numSlice = numTotalSlice
                  self.selectedFiles = range(numTotalSlice)
              except:
                  raise Exception(repr(self) + ' class - if numSlice is not provided and selectedFiles is not: 
                  the path must me provided')       
        elif ("numSlice" in kwargs) and ("selectedFiles" in kwargs): #If a number of slice and the list 
        of preselected files are passed on
            #Check if a number of slice to pick has been specified
            #If yes : n slices are randomly selected (no duplicate)
            #If not : all slices are selected
            if (type(kwargs['numSlice']) == int) or (type(kwargs['numSlice']) == float):
              if int(kwargs['numSlice']) != (len(kwargs['selectedFiles'])):
                raise Exception(repr(self) + ' class - numSlice and selectedFiles must display the same length')
              else:
                self.numSlice = int(kwargs['numSlice'])
                self.selectedFiles = kwargs['selectedFiles']
            elif kwargs['numSlice'] == None:
              self.numSlice = numTotalSlice
              self.selectedFiles = kwargs['selectedFiles']
            else:
              raise Exception(repr(self) + ' class - numSlice must either be a number or None')
        elif ("numSlice" not in kwargs) and ("selectedFiles" in kwargs): #If only the list 
        of preselected files is passed on
            self.selectedFiles = kwargs['selectedFiles']
            self.numSlice = (len(self.selectedFiles))
        '''
        '''
        #Cleaning is needed if masks are pixelated
        #By default set at False if the stack is not the set of segmented slices
        if self.isSegmented:
          self.maskPreprocessing = kwargs['maskPreprocessing']
        else:
          self.maskPreprocessing = False
        '''

        if "numSlice" in kwargs and "selectedFiles" not in kwargs:
            # If a number of slices is specified, but no list of preselected files is:
            # - If numSlice is None, all slices are selected.
            # - If numSlice is a valid number, n slices are randomly selected (no duplicate).
            # - If numSlice is greater than the total number of slices, an exception is raised.
            if kwargs["numSlice"] is None:
                self.numSlice = self.numTotalSlice
                self.selectedFiles = range(self.numTotalSlice)
            elif kwargs["numSlice"] <= self.numTotalSlice:
                self.numSlice = kwargs["numSlice"]
                self.selectedFiles = random.sample(range(0, self.numTotalSlice - 1), self.numSlice)
            else:
                raise Exception(
                    repr(self)
                    + " class - numSlice should be less or equal to the number of slice in the folder provided"
                )
        elif "numSlice" not in kwargs and "selectedFiles" not in kwargs:
            # If all slices are selected, and not preselected files are passed on:
            # - If the stack is a mask stack, an exception is raised.
            # - If the stack is an image stack, all slices are selected.
            if self.isSegmented:
                raise Exception(
                    repr(self)
                    + " class - if the stack is a mask stack then the associated selectedFiles from the corresponding "
                      "image stack must be provided"
                )
            else:
                self.numSlice = self.numTotalSlice
                self.selectedFiles = range(self.numTotalSlice)
        elif "numSlice" in kwargs and "selectedFiles" in kwargs:
            # If a number of slices and the list of preselected files are passed on:
            # - If numSlice is a valid number and the length of the list of preselected files is different,
            # an exception is raised.
            # - If numSlice is None, the length of the list of preselected files is used.
            # - If numSlice is not a valid number, an exception is raised.
            if (
                    type(kwargs["numSlice"]) == int
                    or type(kwargs["numSlice"]) == float
            ) and int(kwargs["numSlice"]) != len(kwargs["selectedFiles"]):
                raise Exception(
                    repr(self)
                    + " class - numSlice and selectedFiles must display the same length"
                )
            elif kwargs["numSlice"] is None:
                self.numSlice = len(kwargs["selectedFiles"])
                self.selectedFiles = kwargs["selectedFiles"]
            else:
                raise Exception(repr(self) + " class - numSlice must either be a number or None")
        elif "numSlice" not in kwargs and "selectedFiles" in kwargs:
            # If only the list of preselected files is passed on:
            # - The length of the list of preselected files is used.
            self.selectedFiles = kwargs["selectedFiles"]
            self.numSlice = len(self.selectedFiles)

        self.setListSlice()

    # GETs
    def get_path(self):
        """
            Returns the path to the image directory.

            Returns:
                str: The path to the image directory.
        """
        return self.path

    def get_is_segmented(self):
        """
            Returns the segmentation status of the stack.

            Returns:
                bool: True if the stack is segmented, False otherwise.
        """
        return self.isSegmented

    def get_image_type(self):
        """
            Returns the image type of the stack.

            Returns:
                int: 0 for grayscale, 1 for RGB.
        """
        return self.imageType

    '''
    def getIsMulticlass(self):
        return self.isMulticlass
    '''

    def get_name(self):
        """
           Returns the name of the image stack.

           Returns:
               str: The name of the stack.
        """
        return self.name

    '''
    def getMaskPreprocessing(self):
        return self.maskPreprocessing
    '''

    def get_slice_from_position(self, position):
        """
            Retrieves slice(s) from the specified position(s).

            Args:
                position (int or list): Index or list of indices of the slice(s) to retrieve.

            Returns:
                object or list: The slice or list of slices at the specified position(s).

            Raises:
                Exception: If the position is not an integer or a list of integers.
        """
        # If the user wants a single slice to be returned
        if isinstance(position, int):
            return self.listSlice[position]
        # If multiple slices are selected
        elif isinstance(position, list):
            return [self.listSlice[i] for i in position]
        else:
            raise Exception(repr(self) + ' class - getSliceFromPosition can only take integer '
                                         'or list of integers as input')

    def get_slice_from_name(self, name):
        """
            Retrieves a slice by its name.

            Args:
                name (str): The name of the slice to retrieve.

            Returns:
                object: The slice with the specified name, or None if not found.
        """
        slice_tmp = next((x for x in self.listSlice if x.name == name), None)
        return slice_tmp

    def get_channel(self):
        """
            Returns the number of image channels.

            Returns:
                int: The number of channels in the images.
        """
        return self.channel

    def get_stack_size(self):
        """
            Returns the number of slices in the stack.

            Returns:
                int: The number of slices.
        """
        return self.numSlice

    def get_list_slice(self):
        """
            Returns the list of image slices.

            Returns:
                list: The list of slices.
        """
        return self.listSlice

    def get_width(self):
        """
            Returns the width of the images.

            Returns:
                int: The width of the images.
        """
        return self.width

    def get_height(self):
        """
            Returns the height of the images.

            Returns:
                int: The height of the images.
        """
        return self.height

    """
    def getListSlicePath(self):
        return self.listSlicePath
    """

    def get_stack_image(self):
        """
            Returns the stack of images.

            Returns:
                array: The stack of images.
        """
        return self.stackImage

    def get_stack_label(self):
        """
            Returns the stack of labels.

            Returns:
                array: The stack of labels.
        """
        return self.stackLabel

    def get_selected_files(self):
        """
            Returns the list of selected files.

            Returns:
                list: The list of selected files.
        """
        return self.selectedFiles

    def get_num_class(self):
        """
            Returns the number of classes.

            Returns:
                int: The number of classes.
        """
        return self.numClass

    """ 
    def getstackTransformed(self):
        return self.stackTransformed
    """

    def get_is_slice_list_supplied(self):
        """
            Returns whether a slice list is supplied.

            Returns:
                bool: True if a slice list is supplied, False otherwise.
        """
        return self.isSliceListSupplied

    def crop_around_center(self, img):
        """
            Crops the image to the specified width and height around its center.

            Args:
                img (array): The NumPy/OpenCV image to be cropped.

            Returns:
                array: The cropped image.
        """
        image_size = (img.shape[1], img.shape[0])
        image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

        '''
        if (width > image_size[0]):
          width = image_size[0]

        if (height > image_size[1]):
          height = image_size[1]
        '''
        x1 = int(image_center[0] - self.width * 0.5)
        x2 = int(image_center[0] + self.width * 0.5)
        y1 = int(image_center[1] - self.height * 0.5)
        y2 = int(image_center[1] + self.height * 0.5)

        return img[y1:y2, x1:x2]

    def set_path(self, path):
        """
           Sets the path to the image directory.

           Args:
               path (str): The path to be set.
        """
        self.path = path

    def set_list_slice(self):
        """
            Initializes the list of image slices.

            If a list of slices is supplied, it uses that list. Otherwise, it generates
            the slices from the images in the specified directory, processing each image
            based on its type and dimensions.
        """
        # If a list of slices is supplied already
        if self.isSliceListSupplied:
            self.listSlice = self.selectedFiles
        else:
            # If no list of slices is supplied, the slices must be generated
            image_files = sorted(os.listdir(self.path))

            # Define self.numTotalSlice
            numTotalSlice = len(image_files)

            # Iterate through all image files in the input folder
            def image_reading(picture_path):
                if picture_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif', '.tiff')):
                    # Use tifffile to read TIFF files
                    if picture_path.lower().endswith(('.tif', '.tiff')):
                        image = tiff.imread(picture_path)
                        # Transpose TIFF images to (width, height) format if necessary
                        if image.shape[0] < image.shape[1]:
                            image = np.transpose(image, (1, 0))
                    else:
                        image = cv2.imread(picture_path, cv2.IMREAD_UNCHANGED)

                    # Check if the image is grayscale (1 channel)
                    if image.ndim == 2:
                        image = np.expand_dims(image, axis=-1)
                        self.channel = 1
                    else:
                        # if RGB as input and the user want grayscale scanner img as output
                        if self.imageType == 0:
                            # Convert multi-channel image to grayscale
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                            image = np.expand_dims(image, axis=-1)
                            self.channel = 1
                        # if RGB as input and the user want RGB img type as output
                        elif self.imageType == 1:
                            self.channel = 3

                    image = self.cropAroundCenter(image)
                    image_sequence.append(image)
                else:
                    raise Exception(repr(self) + ' class - Files can be png, jpg, jpeg, bmp, gif, tif or tiff')

            # If it's a scanner/image
            if not self.isSegmented:
                pb = ProgressBar(self.numSlice, txt=repr(self) + '- Loading ' + self.name)
                image_sequence = []
                for index in self.selectedFiles:
                    image_path = os.path.join(self.path, image_files[index])
                    image_reading(image_path)
                    self.listSlicePath.append(image_path)
                    self.numClass = 0
                    pb += 1
                self.listSlice = image_sequence
            # If it's a label/mask
            else:
                pb = ProgressBar(self.numSlice, txt=repr(self) + '- Loading ' + self.name)
                image_sequence = []
                for index in self.selectedFiles:
                    image_path = os.path.join(self.path, image_files[index])
                    image_reading(image_path)
                    self.listSlicePath.append(image_path)
                    self.numClass = 0
                    pb += 1
                self.listSlice = image_sequence

    def set_num_class(self, num_class):
        """
            Sets the number of classes.

            Args:
                num_class (int): The number of classes to set.
        """
        self.numClass = num_class

    def set_stack_image(self, stack_image):
        """
            Sets the stack of images.

            Args:
                stack_image (array): The stack of images to set.
        """
        self.stackImage = stack_image

    def set_stack_label(self, stack_label):
        """
            Sets the stack of labels.

            Args:
                stack_label (array): The stack of labels to set.
        """
        self.stackLabel = stack_label
