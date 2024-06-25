import cv2 
import numpy as np 
from math import cos,sin,floor
from hough_lines_corners import HoughLineCornerDetector

from Contrast import apply_brightness_contrast

from sys import argv

outputImage = False


def order_points(pts):
      """
      Function for getting the bounding box points in the correct
      order
      Params
      pts     The points in the bounding box. Usually (x, y) coordinates
      Returns
      rect    The ordered set of points
      """
      # initialzie a list of coordinates that will be ordered such that 
      # 1st point -> Top left
      # 2nd point -> Top right
      # 3rd point -> Bottom right
      # 4th point -> Bottom left
      rect = np.zeros((4, 2), dtype = "float32")

      # the top-left point will have the smallest sum, whereas
      # the bottom-right point will have the largest sum

      s = pts.sum(axis = 2)

    
      rect[0] = pts[np.argmin(s)]
      rect[2] = pts[np.argmax(s)]
      
      # now, compute the difference between the points, the
      # top-right point will have the smallest difference,
      # whereas the bottom-left will have the largest difference
      diff = np.diff(pts, axis = 2)

      
      rect[1] = pts[np.argmin(diff)]
      rect[3] = pts[np.argmax(diff)]
      
      

      # return the ordered coordinates
      return rect

def getBestContourShape(contours):
     
    best = None
    maxArea = 0

    for contour in contours:
          
          perimeter = cv2.arcLength(contour,True)
          area = cv2.contourArea(contour) 
          approx_vertices = cv2.approxPolyDP(contour, 0.02*perimeter, True)
          if area > maxArea:## and len(approx_vertices)==4 :
               maxArea = area
               best = contour
    return best

               
def preprocess(image):

    




    ## Greyscale
    greyscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    

    ## Increase Contrast 
    contrast_image = apply_brightness_contrast(greyscale_image, 0,128)
    



    ## Blur
    # ## using gaussian, but could use mean denoiser
    blur_image = cv2.GaussianBlur(contrast_image, (21,21), 1)
    


    ## Threshold Image
    threshold_used, thresholded_image = cv2.threshold(blur_image, 128,255, cv2.THRESH_BINARY)## + cv2.THRESH_OTSU)
    # thresholded_image = cv2.adaptiveThreshold(blur_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #  cv2.THRESH_BINARY,11,2)

    closing_kernel =  cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (3, 3)
        )
    

    ## Closing 
    closed_image = cv2.morphologyEx(
            thresholded_image,
            cv2.MORPH_CLOSE,
            closing_kernel,
            iterations=3
        )


    canny_lower_threshold = 100
    canny_upper_threshold= 180

    ## Detect Edges 
    canny_image = cv2.Canny( closed_image,
                            canny_lower_threshold,
                            canny_upper_threshold
                            )

    if outputImage:

        cv2.imshow("greyscale_image",greyscale_image)
        cv2.imshow("contrast_image",contrast_image)
        cv2.imshow("blur_image",blur_image)
        cv2.imshow("thresholded_image",thresholded_image)
        cv2.imshow("closed_image",closed_image)
        cv2.imshow("canny_image",canny_image)



    return canny_image


def contourProcessing(processed_image, original_image=None):

    

    ## Find Contours
    contours, hierarchy = cv2.findContours(processed_image,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if outputImage: 

        if original_image is None:
            original_image = processed_image
        
        contours_img = original_image.copy()
        cv2.drawContours(contours_img, contours, -1, (0,255,0), 10)

 

        ## Display Contour Points
        for points in contours:

        
            for point in points:
            
                x = int(point[0][0])
                y = int(point[0][1])
                
            
                cv2.circle(contours_img, (x,y), radius=20, color=(0, 0, 255), thickness=-1)

        cv2.imshow("contours image: ", contours_img)


    ## find best contour 
    best_contour = getBestContourShape(contours)

    if best_contour is not  None and outputImage:

        if original_image is None:
            original_image = processed_image
        
        best_contour_image = original_image.copy()
        cv2.drawContours(best_contour_image, [best_contour], -1, (0,255,0), 10)

        for point in best_contour:
        
                    
            x = int(point[0][0])
            y = int(point[0][1])
            

            cv2.circle(best_contour_image, (x,y), radius=20, color=(0, 0, 255), thickness=-1)


        cv2.imshow("best contours image: ", best_contour_image)
    
    return best_contour


def fitDocumentToImage(image, contour, width, height):

    ## Warp Image, and crop out background
    ordered_points = order_points(contour)
    

    dst = np.array([
            [0, 0],                         # Top left point
            [width - 1, 0],              # Top right point
            [width - 1, height - 1],  # Bottom right point
            [0, height - 1]],            # Bottom left point
            dtype = "float32"               # Date type
        )


    matrix = cv2.getPerspectiveTransform(ordered_points, dst)
    warped_image = cv2.warpPerspective(image, matrix, (width, height)) 

    if outputImage: cv2.imshow("warped image contours", warped_image)

    return warped_image
    
    

def extract_document(image, height=1000, width = None):

    if width == None :
        ## get dimensions of image
        initial_height , initial_width , initial_channels = image.shape

        ## get final image dimensions
        image_ratio = initial_width/initial_height

        height = 1000
        width = floor(height*image_ratio)


    ## Resize Image 
    image = cv2.resize(image,(width,height))
    if outputImage: cv2.imshow("image",image)

    ## Preprocessing
    preprocessed_image = preprocess(image)

    ## Get Best Contour 
    best_contour = contourProcessing(preprocessed_image,original_image=image)

    ## Fit Document To Whole Image
    if best_contour is not None:
        final_document = fitDocumentToImage(image, best_contour, width,height)

        if outputImage: cv2.imshow("final_document", final_document)

        return final_document
    
    else:

        return None





if __name__ == "__main__":
    outputImage = True

    if len(argv) < 2:
        image_path = "My_Doc_Examples/border3.jpg"
    else:
        image_path = argv[1]


    ## Read in image
    image = cv2.imread(image_path)

    extract_document(image)


































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

