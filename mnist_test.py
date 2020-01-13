import keras
import sys
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# импорт MNIST dataset
from keras.datasets import mnist


#Загрузка готовых весов
model.load_weights('mnist_weights_epoch100.h5')


# Проверка изображения
if(len(sys.argv) == 1):
    img = cv2.imread("6.jpg")
    if(not img.data):
        print("Could not load image")
        exit

    # предварительная обработка
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (28, 28), interpolation = cv2.INTER_AREA)
    img = cv2.bitwise_not(img)
    img = img.reshape(1,28,28,1)
    img = img.astype('float32')
    img /= 255    

    # Предсказание рукописной цифры во входном изображении
    score = model.predict(img, batch_size=1, verbose=0)
    
    # Показ результатов   
    print("\Шансы для всех возможных результатов: " + sys.argv[0])
    sort = sorted(range(len(score[0])), key=lambda k:score[0][k],reverse=True)
    for index in sort:
        print(str(index) + ": " + str(score[0][index]))  
    percent = format(score[0][sort[0]] * 100, '.2f')
    print("\nЯ думаю, что на " + str(percent) + "% это " + str(sort[0]))
