import tensorflow as tf
from tensorflow import keras
from logconfig import get_logger

if __name__ == '__main__':
    logging = get_logger('INFO')
    logging.info('This is an info message')

    logging.info(tf.__version__)
    logging.info(keras.__version__)

    fashion_mnist = keras.datasets.fashion_mnist
    (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

    # 顺序连接
    model = keras.models.Sequential()
    # 将图像转换为一维数组
    model.add(keras.layers.Flatten(input_shape=[28,28]))
    # 添加300个神经元的Dense隐藏层
    model.add(keras.layers.Dense(300, activation='relu'))
    model.add(keras.layers.Dense(100, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))

    print(model.summary())
    model.compile(loss='sparse_categotical_crossentropy',optimizer='sgd',metrics=['accuracy'])

    history=model.fit(X_test, y_test, epochs=30)
