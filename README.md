# Behavioral-Cloning-Project

## Overview
This project aims at cloning the driving behavior of user in the simulator. The captured data is then used to train a model which would drive the vehicle in autonomous mode in the same simulator, such that the vehicle completes at least one lap on the track for which the data was collected.

The steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Important Files and Directories
* model.py    : Python code for the model, its training and saving the model
* drive.py    : Python code to drive the car using the trained model
* video.py    : Python code to create video from captured images
* model.ipynb : Jupyter python notebook containing the code for the model
* model.h5    : Trained model
* writeup_report.md : Writeup report for this project in markdown format
* video.mp4   : Output video of car autonomously completing one lap of the track

## Running the Code
This model was trained using following versions of major libraries:

* Tensorflow-gpu 1.0.1
* Keras 1.2.4
* Cuda Toolkit 8.0
* Cudnn v5.1

To train the model, you can either use __model.ipynb__ or __model.py__. Modifyable parameters can be found at lines 13-15 in these files.

Model(`model.h5`) can be used to drive the car in Udacity simulator by using the following command:

    `python drive.py model.h5`

To record the run, following command can be used:

    `python drive.py model.h5 recorded_run`

This will create a directory with name _recorded\_run_ containing the captured images.

These captured imaged can then be used to create a video by using the following command:

    `python video.py recorded_run`

This model was trained on the PC using Integrated graphics for Display and external Nvidia graphics for training (only). So, it may not be able to use external GPU and throw errors if used to drive car in simulator in autonomous mode using `drive.py`. To avoid such errors and run the model on CPU, I've added following line in `drive.py` (drive.py line 24).

    `os.environ["CUDA_VISIBLE_DEVICES"] = "-1"`

If you are using same Nvidia GPU for display and training, you can uncomment this line and proceed.
