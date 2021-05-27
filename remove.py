import os
import cv2
import numpy as np

def findLargestContour(edgeImg):
    _, contours, hierarchy = cv2.findContours(edgeImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # From among them, find the contours with large surface area.
    contoursWithArea = []
    for contour in contours:
        area = cv2.contourArea(contour)
        contoursWithArea.append([contour, area])
		
    contoursWithArea.sort(key=lambda tupl: tupl[1], reverse=True)
    largestContour = contoursWithArea[0][0]
    return largestContour

def remove_background(filename, background):
    image = cv2.imread(filename, 1)

    #make mask to crop out known background
    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.rectangle(mask, (40, 50), (600, 650), 255, -1)
    
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    masked_background = cv2.bitwise_and(background, background, mask=mask)

    blurred_image = cv2.GaussianBlur(masked_image, (21, 21), 0)
    blurred_background = cv2.GaussianBlur(masked_background, (21, 21), 0)

    #get the difference between the background and image with the object in it
    diff = cv2.absdiff(blurred_image, blurred_background)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    threshold = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)[1]

    #try to merge together nearby contours
    _, contours, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    thresh_gray = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (55,55)))
    # Find contours in thresh_gray after closing the gaps
    _, contours, hier = cv2.findContours(thresh_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    '''
    contourImg = np.copy(image)
    cv2.drawContours(contourImg, [contour], 0, (0, 255, 0), 2, cv2.LINE_AA, maxLevel=1)
    '''

    '''
    contour = findLargestContour(threshold)
    contour_mask = np.zeros_like(threshold)
    cv2.fillPoly(contour_mask, [contour], 255)
    res = cv2.bitwise_and(image, image, mask = contour_mask)
    '''

    #fill in the contours to make the mask
    contour_mask = np.zeros_like(threshold)
    cv2.fillPoly(contour_mask, contours, 255)
    res = cv2.bitwise_and(image, image, mask = contour_mask)

    hull = []
    # calculate points for each contour
    for i in range(len(contours)):
        # creating convex hull object for each contour
        hull.append(cv2.convexHull(contours[i], False))

    # create an empty black image
    drawing =  np.copy(image)
    # draw contours and hull points
    for i in range(len(contours)):
        color_contours = (0, 255, 0) # green - color for contours
        color = (0, 255, 0) # blue - color for convex hull
        # draw ith contour
        #cv2.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)
        # draw ith convex hull object
        cv2.drawContours(drawing, hull, i, color, 2, 8)

    hull_mask = np.zeros_like(threshold)
    cv2.fillPoly(hull_mask, hull, 255)

    cv2.imshow('drawing', drawing)
    cv2.imshow('hull_mask', hull_mask)
    res2 = cv2.bitwise_and(image, image, mask = hull_mask)
    cv2.imshow('res2', res2)


    cv2.imshow('image', image)
    #cv2.imshow('grayscale difference between background image and object image', gray)
    #cv2.imshow('mask', contour_mask)
    #cv2.imshow('cropped image2', res)
    keyboard = cv2.waitKey(25)
    
    
    fname = filename.split('/')[-1]
    name = fname.split('.')
    cv2.imwrite("./images/"+fname, image)
    cv2.imwrite("./images/"+name[0]+"_cropped."+name[1], res2)
    cv2.imwrite("./images/"+name[0]+"_overlay."+name[1], drawing)
    
    
background = cv2.imread("background.png", 1)

for root, dirs, files in os.walk("/home/phiggin1/gold/images/image_raw", topdown=False):
    for name in files:
        print("Processing "+os.path.join(root, name))
        remove_background(os.path.join(root, name), background)
        break
    