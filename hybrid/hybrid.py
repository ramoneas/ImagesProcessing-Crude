import cv2
import numpy as np
import math
IMG_LEFT_NAME = 'boat.jpg' 
IMG_RIGHT_NAME = 'mar.jpg' 
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

#Convolve function
def convolve_2d(getPadding, getNewImage, img, kernel):
    kernel= np.flipud(kernel)
    return correlación_cruzada_2d(getPadding, getNewImage, img, kernel)
    
#Gaussian Blur function
def gaussian_blur_kernel_2d(img, size):        
    kernel = gausskernel(size) # Calcular kernel de convolución gaussiana
    return correlación_cruzada_2d(getPadding, getNewImage, img, kernel)

#Generate low filter img
def lowfilter(img, size):
    return gaussian_blur_kernel_2d(img, size)

#Generate high filter img
def highfilter(img, size):
    low_filter_img = lowfilter(img, size)   
    return img - low_filter_img

#Normalizar la imagen
def normalize(img):
    img = img/np.max(img)
    return img

#Hybrid Image function
def generate_hybrid_img(right_img, left_img):
    low_filter_img = normalize(lowfilter(right_img , 3)) #Lowfilter
    high_filter_img = normalize(highfilter(left_img, 11)) #highfilter
    return low_filter_img + high_filter_img
 
#How to load an image from a file
left_img = cv2.imread(IMG_LEFT_NAME, 0)
right_img = cv2.imread(IMG_RIGHT_NAME, 0)
left_img = cv2.resize(left_img, RES_IMG)
right_img = cv2.resize(right_img, RES_IMG)

hybrid = generate_hybrid_img(right_img, left_img)

cv2.imshow("boat-image", left_img )
cv2.imshow("mar-image", right_img )
cv2.imshow("hybrid-image", hybrid)
cv2.imwrite("hybrid_img.jpg", hybrid*255)
cv2.waitKey(0)
cv2.destroyAllWindows()