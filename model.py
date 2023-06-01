import tensorflow as tf
from tensorflow import keras
from logconfig import get_logger
import pandas as pd
import matplotlib.pyplot as plt


def action():
    logging = get_logger('INFO')
    logging.info('This is an info message')
    logging.info(tf.__version__)
    logging.info(keras.__version__)

    fashion_mnist = keras.datasets.fashion_mnist
    (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
    X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
    y_valid, y_train = y_train_full[:5000] / 255.0, y_train_full[5000:] / 255.0

    class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag",
                   "Ankle boot"]
    # print(class_names[y_train[0]])
    # 顺序连接
    model = keras.models.Sequential()
    # 将图像转换为一维数组
    model.add(keras.layers.Flatten(input_shape=[28, 28]))
    # 添加300个神经元的Dense隐藏层
    model.add(keras.layers.Dense(300, activation='relu'))
    model.add(keras.layers.Dense(100, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))

    print(model.summary())
    print(model.layers)
    print(model.layers[1].name)
    print(model.layers[2].name)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))

    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()
    print('end--------')
