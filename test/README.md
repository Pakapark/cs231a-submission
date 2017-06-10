# Project Proposal: Vision-based Pedestrian Tracking
By Pakapark Bhumiwat, Franklin Jia

## Motivation

Object tracking is the process of locating and following a moving object over time through a camera. We will be tackling a more specific subset of object tracking: pedestrian tracking. As technology continues to take the world by storm, an increase in demand for automated video analysis has led to a great deal of research being conducted in tracking and on tracking algorithms. Because object tracking involves detecting interesting objects, tracking these objects over time, and analyzing these tracks to discern the object’s behavior, there are many applications for tracking research in the fields of surveillance, autonomous vehicles, military operations, and many others.  

## Outline

There are several levels that we will address in this project, in the following order:

### BASIC
* Level 0 : Given a stationary camera with a single pedestrian in a frame, the program can keep track of that pedestrian from the beginning to the end. 

### INTERMEDIATE
* Level 1: Given a stationary camera with multiple pedestrians in a frame, the program can keep track of a certain pedestrian by whole and/or by part through pre-labeling from the beginning to the end
* Level 2: Given a stationary camera with multiple pedestrians in a frame, the program can keep track of all moving pedestrians (objects) by assigning unique ID (without pre-labeling) with a limitation of the number of pedestrians per frame, e.g., ordinary street camera.

### ADVANCED
* Level 3: Given a moving camera with multiple pedestrians in a frame, the program can keep track of a certain pedestrian by whole and/or by part through pre-labeling from the beginning to the end
* Level 4: Given a moving camera with multiple pedestrians in a frame, the program can keep track of all moving pedestrians (objects) by assigning unique ID (without pre-labeling) with a limitation of the number of pedestrians per frame, e.g., ordinary street camera, and the speed of the camera.


Our primary goal is to finish at least level 2, that is, to come up with an algorithm by incorporating previous works and also evaluate the preciseness through the rubrics on human vision. These rubrics will cover on the duration of time that the program is able to identify the pedestrian(s) while appearing on the frame and the duration of time that the program correctly identifies the pedestrian(s). Level 3 and 4 are also within our scope and we will try our best to expand our project to these levels. However, if necessary, we will drop the harder problems and tackle the preliminary goal.

## Roadmap

Getting Started (Get Data and Set up)
	This project doesn’t require a large dataset. We will need videos with the specifications for each level of complexity annotated above. We will take the stationary videos with phones, or, in most cases, a GoPro situated on a tripod. For moving videos, we will start by taking videos with a GoPro attached to a GoPro Karma Grip, which allows the user to move while taking a video without any disruptions, stabilizing the video and protecting the feed from any extraneous movement on the part of the user. The next step is to take the moving videos without a stabilizer. 

Baseline Algorithm/Model
At the very baseline for stationary with a single pedestrian, we will try to identify pedestrian simply through frame differentiation. Then, we will attempt to use single object trackers by identifying a certain person as a square (through pre-labeling) and track the movement of square in each frame.  

Complex Algorithm
	In case that the inputs have multiple noises, we will attempt to apply Kalman Filter more precisely predict the location of each pedestrian and reduce some random error and variation. If time permits, we may push forward our theoretical bound by modeling an algorithm that could specifically tackle this problem.

## Resources

Bodor, R., Jackson, B., Papanikolopoulos, N. (2006). Vision-Based Human Tracking and 
Activity Recognition. Procedings of AIRVL, Dept. of Computer Science and Engineering, University of Minnesota. 
http://mha.cs.umn.edu/Papers/Vision_Tracking_Recognition.pdf

Davis, L., Philomin, V., Duraiswami, R. (2015). Tracking Humans from a moving platform. 
Institute for Advanced Computer Studies. Computer Vision Laboratory at University of Maryland. 
https://pdfs.semanticscholar.org/20a1/754a416fcf9f926ab4b8b407f627540d43cf.pdf
 
