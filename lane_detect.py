import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    #return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=8):
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    return img


lines_left_x = []
lines_left_y = []
lines_right_x = []
lines_right_y = []
def draw_lines_extrapolate(img, lines, color=[255, 0, 0], thickness=8) :
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    global lines_left_x, lines_left_y, lines_right_x, lines_right_y
    for line in lines:
        for x1,y1,x2,y2 in line:
            if x1 < img.shape[1]/2 and x2 < img.shape[1]/2 and (-0.6)*(x2-x1) > (y2-y1):
                lines_left_x.extend([x1,x2])
                lines_left_y.extend([y1,y2])
            if x1 > img.shape[1]/2 and x2 > img.shape[1]/2 and (0.6)*(x2-x1) < (y2-y1):
                lines_right_x.extend([x1,x2])
                lines_right_y.extend([y1,y2])
  
    fit_line_left = np.polyfit( lines_left_x, lines_left_y, 1)
    f_l = lambda x: x * fit_line_left[0] + fit_line_left[1]
    cv2.line(img,(np.min(lines_left_x), int(f_l(np.min(lines_left_x)))), (int(img.shape[1]/2), int(f_l(img.shape[1]/2))), color , thickness)
    
    fit_line_right = np.polyfit( lines_right_x, lines_right_y, 1)
    f_r = lambda x: x * fit_line_right[0] + fit_line_right[1]
    cv2.line(img,(int(img.shape[1]/2), int(f_r(img.shape[1]/2))), (np.max(lines_right_x), int(f_r(np.max(lines_right_x)))), color , thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, extrapolate=1):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    if extrapolate : 
        draw_lines_extrapolate(line_img, lines)
    else : 
        draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def save_image( img, filename) :
    print ("Saving ", filename)
    mpimg.imsave(filename, img)

def display_image(image) : 
     plt.imshow(image)
     plt.show()

def process_image (image, extrapolate=1) :    
    #display_image(image)
    # Convert into grey scale
    gray = grayscale(image)
    #display_image(gray)

    #Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
    blur_gray = gaussian_blur( gray, kernel_size)
    #display_image(blur_gray)

    # Define our parameters for Canny and apply
    low_threshold = 100
    high_threshold = 150
    edges = canny( blur_gray, low_threshold, high_threshold)
    #display_image(edges)
    
    #Region of Interest 
    imshape = edges.shape
    vertices = np.array([[(0,imshape[0]),(450, 320), (550, 320),(imshape[1],imshape[0])]], dtype=np.int32)
    masked_edges = region_of_interest( edges, vertices)
    #display_image( masked_edges)

    # Hough Transform
    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 2               # distance resolution in pixels of the Hough grid
    theta = np.pi/180     # angular resolution in radians of the Hough grid
    threshold = 30        # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 100 # minimum number of pixels making up a line
    max_line_gap = 200    # maximum gap in pixels between connectable line segments

    lines = hough_lines( masked_edges, rho, theta, threshold, min_line_length, max_line_gap, extrapolate)

    final_image = weighted_img( lines, image )
    #display_image(final_image)
    return final_image

#reading in an image
image_arr = os.listdir("test_images/")
for img_file in image_arr:
    in_img_file = 'test_images/' + img_file 
    out_img_file = 'test_images_output/' + os.path.splitext(img_file)[0] + '.png'
    out_img_noexp_file = 'test_images_output/' + os.path.splitext(img_file)[0] + '_no_extrapolate.png'
    image = mpimg.imread(in_img_file)
    save_image( process_image(image), out_img_file)
    save_image( process_image(image,0), out_img_noexp_file)

video_arr = os.listdir("test_videos/")
for video_file in video_arr :
   in_video_file  = 'test_videos/' + video_file
   out_video_file = 'test_videos_output/' + video_file
   out_video_noexp_file = 'test_videos_output/no_extrapolate_' + video_file 
   clip1 = VideoFileClip(in_video_file)
   white_clip = clip1.fl_image(process_image)
   white_clip.write_videofile( out_video_file, audio=False)



