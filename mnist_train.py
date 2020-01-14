from __future__ import print_function
import keras
from keras.datasets import mnist  # подпрограммы для извлечения набора данных MNIST
from keras.models import Sequential  # последовательное описание модели
from keras.layers import Dense, Dropout, Flatten  # Dense - слой из 10 нейронов.
# Главная идея Dropout — вместо обучения одной DNN обучить ансамбль нескольких DNN, а затем усреднить полученные результаты.
# Flatten - слой выравнивания. Он служит соединительным узлом между слоями.
from keras.layers import Conv2D, MaxPooling2D
# Первые 2 слоя – Conv2D. Эти сверточные слои будут работать с входными изображениями, которые рассматриваются как двумерные матрицы.
# MaxPooling2D - слои подвыборки по максимальному значению.
from keras import backend as K
import pylab
import matplotlib.pyplot as plt

batch_size = 128  # batch_size — количество обучающих образцов, обрабатываемых одновременно за одну итерацию алгоритма градиентного спуска;
num_classes = 10  # число выходов
epochs = 3  # эпохи - количество итераций

# Размер изображения
img_rows, img_cols = 28, 28

# Данные, разделенные между train и test
# Загружаем данные x_train и x_test содержат двухмерный массив с изображение цифр
# x_test, y_test массив с проверочными данными сети.
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Трансформируем из двухмерного массива в трех мерный(28х28х1 канал)
# Функция reshape() изменяет форму массива без изменения его данных.
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# Изменение типов данных
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Преобразовать векторы классов в матрицы бинарных классов
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# создание модели
model = Sequential()
# Добавляем слой
# Первый слой будет сверточный двухмерный (Conv2D) .
# Эти сверточные слои будут работать с входными изображениями, которые рассматриваются как двумерные матрицы.
# kernel_size=3 — размер ядра 3х3.
# Функция активации 'relu' ( Rectified Linear Activation ) , 64 это число ядер свертки( сколько признаком будем искать)
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
# Второй сверточный слой
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# Создаем вектор для полносвязной сети. Flatten() – слой, преобразующий 2D-данные в 1D-данные.
model.add(Flatten())
model.add(Dense(128, activation='relu'))  # relu - функция активации для скрытых слоев.
model.add(Dropout(0.5))
# Создадим однослойный перцептрон
model.add(Dense(num_classes, activation='softmax'))  #  Функцию активации выходного слоя - softmax, которая применяется для задач классификации.
# Компиляция модели
# Оптимизатор весов optimizer='adam' (Адам: метод стохастической оптимизации).
# Функция потерь : loss='categorical_crossentropy' категориальная перекрестная энтропия (categorical crossentropy CCE).
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Nadam(),
              metrics=['accuracy'])
# Запуск обучения сети
# сохраняем историю обучения
# epochs - число эпох , validation_data=(x_test, y_test) — проверочные данные
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))

# 1 граф
pylab.figure(1)
x = range(3)

plt.grid(True)

# График обучения и достоверность значений точности
plt.plot(x,
         history.history['acc'],
         'bo-',
         label='Train accuracy')

plt.plot(x,
         history.history['val_acc'],
         'ro-',
         label='Validation accuracy')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.legend(loc='lower right')

# вывод итоговых данных
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# save learned weights
model.save_weights('mnist_weights_epoch10.h5')

# 2 граф
pylab.figure(2)
plt.grid(True)

plt.plot(x,
         history.history['loss'],
         'bo-',
         label='Train losses')

plt.plot(x,
         history.history['val_loss'],
         'ro-',
         label='Validation losses')

plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.legend(loc='upper right')
