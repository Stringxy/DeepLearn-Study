import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from tensorflow.examples.tutorials.mnist import input_data

batch_size = 128  # 每循环一次，使用的样本数量
num_classes = 10  # 数据类别数量
epochs = 20  # 训练次数

# 导入数据
# MNIST 数据集来自美国国家标准与技术研究所,
# National Institute of Standards and Technology (NIST).
# 训练集 (training set) 由来自 250 个不同人手写的数字构成,
# 其中 50% 是高中学生, 50% 来自人口普查局 (the Census Bureau) 的工作人员.
# 测试集(test set) 也是同样比例的手写数字数据.
# MNIST数据集的官网是Yann LeCun’s website
mnist = input_data.read_data_sets("MNIST_data/")

x_train, y_train = mnist.train.images, mnist.train.labels
x_test, y_test = mnist.test.images, mnist.test.labels
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# 数据的标准化
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
# 将y 标签转化为分类
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))  # Rectified Linear Unit(ReLU) - 用于隐层神经元输出
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))  # Softmax - 用于多分类神经网络输出

# 对模型进行总结 可以看到模型参数个数 每层的shape等信息
model.summary()

# 模型编译，loss是损失函数，optimizer是优化方法
# sigmoid和softmax是神经网络输出层使用的激活函数，分别用于两类判别和多类判别。
# binary cross-entropy和categorical cross-entropy是相对应的损失函数。
# RMSprop是Geoff Hinton提出的一种自适应学习率方法。Adagrad会累加之前所有的梯度平方，
# 而RMSprop仅仅是计算对应的平均值，因此可缓解Adagrad算法学习率下降较快的问题。
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

# 给模型喂数据
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))

# 对模型进行评估
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save("num_model.h5")