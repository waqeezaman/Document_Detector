import cv2 
import numpy as np 
from math import cos,sin
from hough_lines_corners import HoughLineCornerDetector

image_path = "my_doc_example.jpg"


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


lines = cv2.HoughLines(
        canny_image,
        0.1,
        0.1,
        5
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
    


    print(rho,theta)

cv2.imshow("lines_image",lines_image)



# for line in lines:


# print(lines[0])


corner_detector = HoughLineCornerDetector()
corners = corner_detector._get_intersections(lines,image_width,image_height)


print(type(lines_image))



# print(lines)



# ## make greyscale
# ## blur, get rid of noise 

# ## apply thresholding

# ## do closing to get rid of details on page temporarily
# ## canny edge detection 

# ## use hough lines to get all the lines 

# ## find corners of image 

# ## use detected corners on un-closed image 
# ## to crop out non-document areas of image 

# ## fix distortion 




cv2.waitKey(0) 

