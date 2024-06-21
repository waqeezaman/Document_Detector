import cv2 
import numpy as np 
from math import cos,sin,floor
from hough_lines_corners import HoughLineCornerDetector

from Contrast import apply_brightness_contrast


def getBestContourShape(contours):
     
    best = None
    maxArea = 0

    for contour in contours:
          
          perimeter = cv2.arcLength(contour,True)
          area = cv2.contourArea(contour) 
          approx_vertices = cv2.approxPolyDP(contour, 0.02*perimeter, True)
          if area > maxArea and len(approx_vertices)==4 :
               maxArea = area
               best = contour
    return best

               


image_path = "My_Doc_Examples/my_doc_example7.jpg"








closing_kernel =  cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (3, 3)
        )



canny_lower_threshold = 100
canny_upper_threshold= 180




## Read in image
image = cv2.imread(image_path)

initial_height , initial_width , initial_channels = image.shape


image_ratio = initial_width/initial_height
image_height = 1000
image_width = floor(image_height*image_ratio)


## Resize Image 
image = cv2.resize(image,(image_width,image_height))
cv2.imshow("image",image)


## Greyscale

greyscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("greyscale_image",greyscale_image)


contrast_image = apply_brightness_contrast(greyscale_image, 0,64)
cv2.imshow("contrast_image",contrast_image)



## Blur
# ## using gaussian, but could use mean denoiser
blur_image = cv2.GaussianBlur(contrast_image, (21,21), 1)
cv2.imshow("blur_image",blur_image)


## Threshold Image
threshold_used, thresholded_image = cv2.threshold(blur_image, 160,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# thresholded_image = cv2.adaptiveThreshold(blur_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#  cv2.THRESH_BINARY,11,2)
cv2.imshow("thresholded_image",thresholded_image)

## Closing 
closed_image = cv2.morphologyEx(
        thresholded_image,
        cv2.MORPH_CLOSE,
        closing_kernel,
        iterations=0
    )
cv2.imshow("closed_image",closed_image)


## Detect Edges 
canny_image = cv2.Canny( closed_image,
                         canny_lower_threshold,
                         canny_upper_threshold
                        )
cv2.imshow("canny_image",canny_image)




#############################
## contouring 
#############################

contours_img = image.copy()
biggest_contours = image.copy()


## Find Contours
contours, hierarchy = cv2.findContours(canny_image,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(contours_img, contours, -1, (0,255,0), 10)

print("contours")

## Display Contour Points
for points in contours:

   
    for point in points:
    
        x = int(point[0][0])
        y = int(point[0][1])
        
    
        cv2.circle(contours_img, (x,y), radius=20, color=(0, 0, 255), thickness=-1)

cv2.imshow("contours image: ", contours_img)



best_contour = getBestContourShape(contours)

if best_contour is not  None:
        
    cv2.drawContours(biggest_contours, [best_contour], -1, (0,255,0), 10)

    for point in best_contour:
    
                
        x = int(point[0][0])
        y = int(point[0][1])
        

        cv2.circle(biggest_contours, (x,y), radius=20, color=(0, 0, 255), thickness=-1)


    cv2.imshow("best contours image: ", biggest_contours)




    ## Warp Image
    ordered_points = HoughLineCornerDetector._order_points(best_contour)
    print("ORDERED POINTS: ", ordered_points)


    dst = np.array([
            [0, 0],                         # Top left point
            [image_width - 1, 0],              # Top right point
            [image_width - 1, image_height - 1],  # Bottom right point
            [0, image_height - 1]],            # Bottom left point
            dtype = "float32"               # Date type
        )


    matrix = cv2.getPerspectiveTransform(ordered_points, dst)
    warped_image = cv2.warpPerspective(image, matrix, (image_width, image_height)) 

    cv2.imshow("warped image contours", warped_image)

else:
    print("BEST CONTOUR NOT FOUND")









# ## make greyscale


# ## blur, get rid of noise 

# ## apply thresholding

# ## do closing to get rid of details on page temporarily
# ## canny edge detection 

# ## use hough lines to get all the lines 

# ## find potential corners of image 

## use k means to find best match for 4 corners

# ## use detected corners on un-closed image 
# ## to crop out non-document areas of image 

# ## fix distortion 




cv2.waitKey(0) 

