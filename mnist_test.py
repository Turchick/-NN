import keras
import sys
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# импорт MNIST dataset
from keras.datasets import mnist
from PIL import ImageTk, Image
from tkinter import *
from tkinter.filedialog import *
from tkinter import scrolledtext




#Загрузка готовых весов
model.load_weights('mnist_weights_epoch10.h5')
def clicked():
    op = askopenfilename()  
    # Проверка изображения
    if (len(sys.argv) == 1):
        img = cv2.imread(op)
        if (not img.data):
            print("Could not load image")
            exit

        # предварительная обработка
        # Меняем цветовое пространство на RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # меняем размер
        img = cv2.resize(img, (28, 28), interpolation = cv2.INTER_AREA)
        # конвертирование в числа
        img = cv2.bitwise_not(img)
        img = img.reshape(1,28,28,1)
        img = img.astype('float32')
        img /= 255    

        # Предсказание рукописной цифры во входном изображении
        score = model.predict(img, batch_size=1, verbose=0)

        
        # Показ результатов 
        
        txt.insert(INSERT, "\nШансы для всех возможных результатов: ")
        sort = sorted(range(len(score[0])), key=lambda k:score[0][k],reverse=True)
        for index in sort:
            txt.insert(INSERT, "\n" + str(index) + " : " + str(score[0][index]))
        percent = format(score[0][sort[0]] * 100, '.2f')
        txt.insert(INSERT, "\nЯ думаю, что на " + str(percent) + "% это " + str(sort[0]))
        
       
window = Tk()
window.geometry('470x600')
window.title("Распознавание рукописных цифр ")
txt = scrolledtext.ScrolledText(window, width=40, height=35)  
txt.grid(column=0, row=0) 
btn = Button(window, text="Открыть", command=clicked,background="#555",foreground="#ccc",padx="20",pady="8",font="16")   
btn.grid(column=1, row=0)
window.mainloop()   
