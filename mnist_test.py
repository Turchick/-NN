# import required modules
import keras
import sys
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# import MNIST dataset
from keras.datasets import mnist


# load the weights already learned
model.load_weights('mnist_weights_epoch10.h5')


# load test input if provided
if(len(sys.argv) == 1):
    img = cv2.imread("0r.jpg")
    if(not img.data):
        print("Could not load image")
        exit

    # preprocessing
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (28, 28), interpolation = cv2.INTER_AREA)
    img = cv2.bitwise_not(img)
    img = img.reshape(1,28,28,1)
    img = img.astype('float32')
    img /= 255    

    # predict the handwritten digit in the input image 
    score = model.predict(img, batch_size=1, verbose=0)
    
    # display scores    
    print("\nPrediction score for test input: " + sys.argv[0])
    sort = sorted(range(len(score[0])), key=lambda k:score[0][k],reverse=True)
    for index in sort:
        print(str(index) + ": " + str(score[0][index]))  
    percent = format(score[0][sort[0]] * 100, '.2f')
    print("\nI am " + str(percent) + "% confident that it is " + str(sort[0]))
