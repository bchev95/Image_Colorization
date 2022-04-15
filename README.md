# Image Colorization

A containerized Python program for colorizing images using the Open Computer Vision (OpenCV) library.


### Dependencies
- **OpenCV**, **numpy** Python libraries
- **Protoxt**: https://github.com/richzhang/colorization/blob/caffe/colorization/models/colorization_deploy_v2.prototxt
- **CaffeModel**: http://eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/demo_v2/colorization_release_v2.caffemodel
- **AB Points**: https://github.com/richzhang/colorization/blob/caffe/colorization/resources/pts_in_hull.npy

Prototxt, CaffeModel and AB Points files should be put in a directory named /models within the source folder for this project.
Black and white images to colorize should be added to /input_imgs directory


### Running the program
When running the program, provide the name of the file you would like to colorize with the '-p' flag
 - **Example**: "python3 colorization.py -p nyc.jpg"
 This will run the program and output the colorized image into the /results directory as (inputname)_colorized.png