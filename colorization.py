import numpy as np
import cv2

# Protoxt: https://github.com/richzhang/colorization/blob/caffe/colorization/models/colorization_deploy_v2.prototxt
# CaffeModel: http://eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/demo_v2/colorization_release_v2.caffemodel
# AB Points: https://github.com/richzhang/colorization/blob/caffe/colorization/resources/pts_in_hull.npy

cf_model_path = 'models/colorization_release_v2.caffemodel'
prototxt_path = 'models/colorization_deploy_v2.prototxt'
points_path = 'models/pts_in_hull.npy'
img_path = 'input_imgs/nyc.jpg'

#TO DO: Take in filename from command line, append name to "input_imgs/", then use this as path

    input_img = cv2.imread(img_path)

    #Check that input image file exists and was read correctly, else exit
    if input_img is None:
        exit('Error: Input file does not exist, exiting')

    #pts_in_hull.npy contains 313 cluster kernels computed from stacking a and b values in 2D
    pts_file = np.load(points_path)

    #Load pre-trained caffemodel into net_cf variable
    net_cf = cv2.dnn.readNetFromCaffe(prototxt_path, cf_model_path)

    #Define and load CaffeModel blobs - from opencv documentation
    pts_file = pts_file.transpose().reshape(2, 313, 1, 1)
    net_cf.getLayer(net_cf.getLayerId('class8_ab')).blobs = [pts_file.astype(np.float32)]
    net_cf.getLayer(net_cf.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, np.float32)]

    #Normalize values by scaling them down to fit between 0 and 1 (divide by 255)
    normalized_img = (input_img[:,:,[2, 1, 0]] * 1.0 / 255).astype(np.float32)
    #Convert RGB to LAB - LAB is a color space expressed as L - lightness, A - (green/red), B - (blue/yellow)
    lab_converted = cv2.cvtColor(normalized_img, cv2.COLOR_RGB2Lab)
    #Extract L channel (lightness)
    l_chnl = lab_converted[:,:,0] 

    #Model is trained to work with dimensions 224 x 224
    model_width, model_height = 224, 224

#Resize image to model dimensions (224 x 224)
    l_resized = cv2.resize(l_chnl, (model_width, model_height)) 
    #Subtract mean value (this value can be tweaked)
    l_resized -= 50

net_cf.setInput(cv2.dnn.blobFromImage(l_resized))
ab_chnl = net_cf.forward()[0,:,:,:].transpose((1,2,0)) 

#Now resize image back to original size
(orig_input_height, orig_input_width) = normalized_img.shape[:2] 
ab_us = cv2.resize(ab_chnl, (orig_input_width, orig_input_height))
colorized = np.concatenate((l_chnl[:,:,np.newaxis],ab_us), axis = 2)
#Convert from LAB back to BGR
colorized = np.clip(cv2.cvtColor(colorized, cv2.COLOR_Lab2BGR), 0, 1)
#Multiply by 255 to scale back up
img_result = (colorized*255).astype(np.uint8)

#Output image to destination file
if not cv2.imwrite('result.png', img_result):
    raise Exception("Error: could not write image")