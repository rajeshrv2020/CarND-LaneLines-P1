### Project 1 : Finding Lane Lines on the Road** 


**Introduction**

This project introduces few concepts like
  * Canny Edge detection
  * Masking unwanted info from the scene and foucssing only on the region of interest
  * Extrapolate the lanes with the few points


** Description **

My pipeline consists of 7 steps.

  1. Convert the image into the grayscale
  2. Perform smoothening of image using gaussian blur
  3. Do the Canny edge detection to detect the change in the intensity of the image.
  4. Filter out the canny edge points using a polyfit to focus on region of interest
  5. Perform Hough transform to find the lanes
  6. Draw the coloured lines on the detected lanes and extropolate the lines by fitting a line using the min and max value of the lane point
  7. Superimpose the coloured lines on the original image


### Program
I'm yet to get use to juypter notebook. So, the project is completed outside the juypter notebook. 

  To run the program , do the following.

  > python lane_detect.py

   The test image are written into test_images_output directory and the videos are saved into test_videos_output directory


[//]: # (Image References)

[solidWhiteCurve_original_image]: ./test_images/solidWhiteCurve.jpg "Original"
[solidWhiteCurve_Extrapolated_image]: ./test_images_output/solidWhiteCurve.png "Extrapolated"
[solidWhiteCurve_no_Extrapolated_image]: ./test_images_output/solidWhiteCurve_no_extrapolate.png "NonExtrapolated"

[solidWhiteRight_original_image]: ./test_images/solidWhiteRight.jpg "Original"
[solidWhiteRight_Extrapolated_image]: ./test_images_output/solidWhiteRight.png "Extrapolated"
[solidWhiteRight_no_Extrapolated_image]: ./test_images_output/solidWhiteRight_no_extrapolate.png "NonExtrapolated"

[solidYellowCurve2_original_image]: ./test_images/solidYellowCurve2.jpg "Original"
[solidYellowCurve2_Extrapolated_image]: ./test_images_output/solidYellowCurve2.png "Extrapolated"
[solidYellowCurve2_no_Extrapolated_image]: ./test_images_output/solidYellowCurve2_no_extrapolate.png "NonExtrapolated"

[solidYellowCurve_original_image]: ./test_images/solidYellowCurve.jpg "Original"
[solidYellowCurve_Extrapolated_image]: ./test_images_output/solidYellowCurve.png "Extrapolated"
[solidYellowCurve_no_Extrapolated_image]: ./test_images_output/solidYellowCurve_no_extrapolate.png "NonExtrapolated"

[solidYellowLeft_original_image]: ./test_images/solidYellowLeft.jpg "Original"
[solidYellowLeft_Extrapolated_image]: ./test_images_output/solidYellowLeft.png "Extrapolated"
[solidYellowLeft_no_Extrapolated_image]: ./test_images_output/solidYellowLeft_no_extrapolate.png "NonExtrapolated"

[whiteCarLaneSwitch_original_image]: ./test_images/whiteCarLaneSwitch.jpg "Original"
[whiteCarLaneSwitch_Extrapolated_image]: ./test_images_output/whiteCarLaneSwitch.png "Extrapolated"
[whiteCarLaneSwitch_no_Extrapolated_image]: ./test_images_output/whiteCarLaneSwitch_no_extrapolate.png "NonExtrapolated"


## 1.solidYellowLeft 

** Original **

![alt text][solidYellowLeft_original_image]

** Non Extrapolated ** 

![alt text][solidYellowLeft_no_Extrapolated_image]

** Extrapolated **

![alt text][solidYellowLeft_Extrapolated_image]


## 2.solidWhiteRight 


** Original **

![alt text][solidWhiteRight_original_image]

** Non Extrapolated ** 

![alt text][solidWhiteRight_no_Extrapolated_image]

** Extrapolated **

![alt text][solidWhiteRight_Extrapolated_image]

## 3.solidWhiteCurve


** Original **

![alt text][solidWhiteCurve_original_image]

** Extrapolated **

![alt text][solidWhiteCurve_no_Extrapolated_image]

** Non Extrapolated ** 

![alt text][solidWhiteCurve_Extrapolated_image]

## 4.whiteCarLaneSwitch


** Original **

![alt text][whiteCarLaneSwitch_original_image]

** Non Extrapolated ** 

![alt text][whiteCarLaneSwitch_no_Extrapolated_image]

** Extrapolated **

![alt text][whiteCarLaneSwitch_Extrapolated_image]


## 5.solidYellowCurve

** Original **

![alt text][solidYellowCurve_original_image]

** Non Extrapolated ** 

![alt text][solidYellowCurve_no_Extrapolated_image]

** Extrapolated **

![alt text][solidYellowCurve_Extrapolated_image]

## 6.solidYellowCurve2


** Original **

![alt text][solidYellowCurve2_original_image]

** Non Extrapolated ** 

![alt text][solidYellowCurve2_no_Extrapolated_image]

** Extrapolated **

![alt text][solidYellowCurve2_Extrapolated_image]


### Video Files

The Video files are located in the test_videos_output directory

   1. solidWhiteRight.mp4
   2. solidYellowLeft.mp4

### Possible Improvements

Right now, we are trying to fit a straight line to the lanes. This may not work well for curved roads. 

See test_videos_output/challenge.mp4. 

Instead of straight line, we should be using a function which can fit to the curved lanes too. 


