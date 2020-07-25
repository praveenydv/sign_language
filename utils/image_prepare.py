
import cv2
import tensorflow as tf
import imutils
import matplotlib.pyplot as plt
import numpy as np



def nothing(x): #needed for createTrackbar to work in python.
    pass

def getthresh():
    cv2.namedWindow('temp')
    cv2.createTrackbar('thresh', 'temp', 0, 255, nothing)
#     hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    thresh_temp=cv2.getTrackbarPos('thresh', 'temp')
    return thresh_temp



def change_RGBImage_toBinary(img):
#         threshold =getthresh();
        img = np.asarray(img)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#         plt.imshow(gray)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        # threshold the image, then perform a series of erosions +
        # dilations to remove any small regions of noise
        thresh1 = cv2.threshold(gray,130, 255, cv2.THRESH_BINARY_INV)[1]
        thresh = cv2.erode(thresh1, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        # find contours in thresholded image, then grab the largest
        # one
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
#         c = max(cnts, key=cv2.contourArea)
#         cv2.drawContours(img, [c], -1, (0, 255, 255), 2)
        image = cv2.cvtColor(thresh,cv2.COLOR_GRAY2RGB)

        return image




def reshape_and_resize_Image(image):
    img  = cv2.resize(image, (75, 75)) 
#     img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img=tf.reshape(img,[1,75, 75,3])
    return img




def load_model(path):
    model = tf.keras.models.load_model(path)
    return model

def load_saved_model(path):
    model = tf.compat.v2.saved_model.load(path, None)
    return model




def load_letter_labels():
    letter_labels= ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','del','space','nothing']
    return letter_labels



def predict(frame,model,letter_labels):
#     hand= cv2.imread('/home/praveen/ml/sign_language/asl_datbase/asl_alphabet_train/asl_alphabet_train/B/B5.jpg')
#     binary_hand= change_RGBImage_toBinary(frame)
#     bin_hand =changetoBinary(hand_skin)
#     cv2.imshow("Binary hand",binary_hand)
    final_hand= reshape_and_resize_Image(frame)
#     for keras model like h5
    pred= np.argmax(model.predict(final_hand,steps=1)) 
#     pred= np.argmax(model(final_hand))
    print(letter_labels[pred])
    return letter_labels[pred]

