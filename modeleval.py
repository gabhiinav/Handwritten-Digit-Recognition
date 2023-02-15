from keras.models import load_model
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(train_images, train_labels),(test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape(60000, 28, 28, 1)
test_images = test_images.reshape(10000, 28, 28, 1)

train_images = train_images / 255.0
test_images = test_images / 255.0

model = load_model('/Users/abhinav/Desktop/cseProject/mnist.h5')

accuracy, loss = model.evaluate(test_images, test_labels)

print(accuracy)
print(loss)
