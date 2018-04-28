from keras.models import load_model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input

# 加载模型
model = load_model('num_model.h5')

# 本地图片路径
img_path = '4.jpeg'

# 加载本地图片， 模式为灰度，大小缩小为28*28
img = image.load_img(img_path, grayscale=True, target_size=(28, 28))

# 图片转为array
x = image.img_to_array(img)
x = x.reshape(1, 784)
x = preprocess_input(x)

print(model.predict(x))
