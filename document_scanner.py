import cv2 
import numpy as np 
from math import cos,sin
from hough_lines_corners import HoughLineCornerDetector

from sklearn.cluster import KMeans


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

               


image_path = "my_doc_example4.jpg"


image_height = 1200
image_width = 600



closing_kernel =  cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (3, 3)
        )

canny_lower_threshold = 100
canny_upper_threshold= 180





image = cv2.imread(image_path)
image = cv2.resize(image,(image_width,image_height))

cv2.imshow("image",image)




greyscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


cv2.imshow("greyscale_image",greyscale_image)


# ## using gaussian, but could use mean denoiser
blur_image = cv2.GaussianBlur(greyscale_image, (5,5), 1)
cv2.imshow("blur_image",blur_image)


threshold_used, thresholded_image = cv2.threshold(blur_image, 128,255, cv2.THRESH_BINARY )


cv2.imshow("thresholded_image",thresholded_image)


closed_image = cv2.morphologyEx(
        thresholded_image,
        cv2.MORPH_CLOSE,
        closing_kernel,
        iterations=10
    )

cv2.imshow("closed_image",closed_image)


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


contours, hierarchy = cv2.findContours(canny_image,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(contours_img, contours, -1, (0,255,0), 10)

print("contours")

for points in contours:

        print("SINGLE CONTOUR")
    # if len(points) ==4:
        for point in points:
            print("POINT: ", point)
            print("---------------------")
            x = int(point[0][0])
            y = int(point[0][1])
            
        
            cv2.circle(contours_img, (x,y), radius=20, color=(0, 0, 255), thickness=-1)


cv2.imshow("contours image: ", contours_img)



best_contour = getBestContourShape(contours)

cv2.drawContours(biggest_contours, contours, -1, (0,255,0), 10)

for point in best_contour:
   
            
    x = int(point[0][0])
    y = int(point[0][1])
    

    cv2.circle(biggest_contours, (x,y), radius=20, color=(0, 0, 255), thickness=-1)


cv2.imshow("best contours image: ", biggest_contours)



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













lines = cv2.HoughLines(
        canny_image,
        0.1,
        0.1,
        10
    )


lines_image = image.copy()
lines_image[:,:,:]=0


for line in lines:
    rho = line[0][0]
    theta = line[0][1]
    

    a = cos(theta)
    b = sin(theta)
    x0 = a * rho
    y0 = b * rho
    pt1 = (int(x0 + image_width*(-b)), int(y0 + image_height*(a)))
    pt2 = (int(x0 - image_width*(-b)), int(y0 - image_height*(a)))
    cv2.line(lines_image, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
    



cv2.imshow("lines_image",lines_image)





corner_detector = HoughLineCornerDetector()
corner_points = corner_detector._get_intersections(lines,image_width,image_height)


corners_image = lines_image.copy()


for point in corner_points:
    
    x = point[0][0]
    y = point[0][1]
   
    if 0<=x<image_width and 0 <= y < image_height: 
        corners_image[y,x] = [255,0,0]
    

cv2.imshow("corners_image",corners_image)



quadrilaterals_points = corner_detector._find_quadrilaterals(corner_points)


print("QUADS")
print( quadrilaterals_points)

quadrilateral_image = lines_image.copy()




for point in quadrilaterals_points:
    # print(point)
    x = int(point[0][0])
    y = int(point[0][1])
    
    print("X: ",x)
    print("Y: ", y)
    cv2.circle(quadrilateral_image, (x,y), radius=20, color=(0, 255, 0), thickness=-1)





cv2.imshow("quadrilateral_image",quadrilateral_image)

print("QUAD POINTS LEN: ", len(quadrilaterals_points))
rect = HoughLineCornerDetector._order_points(np.asarray(quadrilaterals_points))


dst = np.array([
        [0, 0],                         # Top left point
        [image_width - 1, 0],              # Top right point
        [image_width - 1, image_height - 1],  # Bottom right point
        [0, image_height - 1]],            # Bottom left point
        dtype = "float32"               # Date type
    )


matrix = cv2.getPerspectiveTransform(rect, dst)
                         

warped_image = cv2.warpPerspective(image, matrix, (image_width, image_height)) 

cv2.imshow("warped image", warped_image)


# print(lines)



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

