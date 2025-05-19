import cv2
import numpy as np
import math
IMG_NAME = 'ape.jpeg'
RES_IMG = (500,500) 

#This generate a padding of zeros that surround the image
def getPadding(img, img_axb, convo_axb, zeros):
    for i in range(img_axb[0]):
        for j in range(img_axb[1]):
            zeros[i+int((convo_axb[0]-1)/2), j+int((convo_axb[1]-1)/2)] = img[i, j]
    return zeros

#This multiply the matrix of the image with the kernel matrix      
def getNewImage(img, kernel, img_axb, convo_axb, zeros):
    for i in range(img_axb[0]):
        for j in range(img_axb[1]):
            pos = zeros[i: i + convo_axb[0], j: j + convo_axb[1]]
            new_value = np.sum(pos*kernel)
            img[i,j] = new_value
    return img

# Calcular el núcleo de convolución gaussiano
def gausskernel(size):
    sigma=10.00
    gausskernel=np.zeros((size,size),np.float32)
    for i in range (size):
        for j in range (size):
            norm=math.pow(i-1,2)+pow(j-1,2)
            gausskernel[i, j] = math.exp (-norm / (2 * math.pow (sigma, 2))) # Encuentra convolución gaussiana
            sum = np.sum(gausskernel) # sum
            kernel = gausskernel / sum # normalización
    return kernel

#Cross correlation function
def correlación_cruzada_2d(getPadding, getNewImage, img, kernel):
     
    #Convert the image to a matrix and kernel
    imagen = np.copy(img)
    img_matrix = np.array(imagen)
    
    #CONVERT TO AN n x m
    img_axb = img_matrix.shape
    convo_axb = kernel.shape  
    
    #ROWS and Columns
    rows = img_axb[0] + convo_axb[0] - 1
    columns = img_axb[1] + convo_axb[1] - 1
    zeros = np.zeros((rows, columns))

    zeros = getPadding(imagen, img_axb, convo_axb, zeros)
    return getNewImage(imagen, kernel, img_axb, convo_axb, zeros)        

#Gaussian Blur function
def gaussian_blur_kernel_2d(img, size):        
    kernel = gausskernel(size) # Calcular kernel de convolución gaussiana
    return correlación_cruzada_2d(getPadding, getNewImage, img, kernel)

#Sobel filter function
def sobelFilter(img, direction):
    if direction == "x":
        gx = np.array([[-1,0,+1], [-2,0,+2], [-1,0,+1]])
        return correlación_cruzada_2d(getPadding, getNewImage, img, gx)
    elif direction == "y":
        gy = np.array([[-1,-2,-1], [0,0,0], [+1,+2,+1]])
        return correlación_cruzada_2d(getPadding, getNewImage, img, gy)

#Normalizar la imagen
def normalize(img):
    img = img/np.max(img)
    return img

def nonMaxSup(sumSqr, grad):
    img = np.zeros(sumSqr.shape)
    for i in range(1, int(sumSqr.shape[0]) - 1):
        for j in range(1, int(sumSqr.shape[1]) - 1):
            if((grad[i,j] >= -22.5 and grad[i,j] <= 22.5) or (grad[i,j] <= -157.5 and grad[i,j] >= 157.5)):
                if((sumSqr[i,j] > sumSqr[i,j+1]) and (sumSqr[i,j] > sumSqr[i,j-1])):
                    img[i,j] = sumSqr[i,j]
                else:
                    img[i,j] = 0
            if((grad[i,j] >= 22.5 and grad[i,j] <= 67.5) or (grad[i,j] <= -112.5 and grad[i,j] >= -157.5)):
                if((sumSqr[i,j] > sumSqr[i+1,j+1]) and (sumSqr[i,j] > sumSqr[i-1,j-1])):
                    img[i,j] = sumSqr[i,j]
                else:
                    img[i,j] = 0
            if((grad[i,j] >= 67.5 and grad[i,j] <= 112.5) or (grad[i,j] <= -67.5 and grad[i,j] >= -112.5)):
                if((sumSqr[i,j] > sumSqr[i+1,j]) and (sumSqr[i,j] > sumSqr[i-1,j])):
                    img[i,j] = sumSqr[i,j]
                else:
                    img[i,j] = 0
            if((grad[i,j] >= 112.5 and grad[i,j] <= 157.5) or (grad[i,j] <= -22.5 and grad[i,j] >= -67.5)):
                if((sumSqr[i,j] > sumSqr[i+1,j-1]) and (sumSqr[i,j] > sumSqr[i-1,j+1])):
                    img[i,j] = sumSqr[i,j]
                else:
                    img[i,j] = 0

    return img

def doThreshHyst(img):
    highThresholdRatio =0.32
    lowThresholdRatio = 0.30
    GSup = np.copy(img)
    h = int(GSup.shape[0])
    w = int(GSup.shape[1])
    highThreshold = np.max(GSup) * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio            
    
    for i in range(1,h-1):
        for j in range(1,w-1):
            if(GSup[i,j] > highThreshold):
                GSup[i,j] = 1
            elif(GSup[i,j] < lowThreshold):
                GSup[i,j] = 0
            else:
                if((GSup[i-1,j-1] > highThreshold) or 
                    (GSup[i-1,j] > highThreshold) or
                    (GSup[i-1,j+1] > highThreshold) or
                    (GSup[i,j-1] > highThreshold) or
                    (GSup[i,j+1] > highThreshold) or
                    (GSup[i+1,j-1] > highThreshold) or
                    (GSup[i+1,j] > highThreshold) or
                    (GSup[i+1,j+1] > highThreshold)):
                    GSup[i,j] = 1
                        
    GSup = (GSup == 1) * GSup # This is done to remove/clean all the weak edges which are not connected to strong edges    
    return GSup

#How to load an image from a file
img = cv2.imread(IMG_NAME, 0)
img = cv2.resize( img, RES_IMG)

gauss_img = gaussian_blur_kernel_2d(img, 5)
gx = normalize(sobelFilter(gauss_img, 'x'))
gy = normalize(sobelFilter(gauss_img, 'y'))    
       
sumSqr = np.hypot(gx,gy) #Sum of squared
gradient = np.degrees(np.arctan2(gy, gx)) #find the gradient

img_NMS = normalize(nonMaxSup(sumSqr, gradient))

img_edge_detector = doThreshHyst(img_NMS)


cv2.imshow("original-image", img)
cv2.imshow("ape-edges", img_edge_detector)
cv2.imwrite("ape-edges.jpg", img_edge_detector*255)


cv2.waitKey(0)
cv2.destroyAllWindows()