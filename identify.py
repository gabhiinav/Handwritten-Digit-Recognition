from keras.models import load_model
import numpy as np 
import matplotlib.pyplot as plt
import os 
import cv2

model = load_model('/Users/abhinav/Desktop/cseProject/mnist.h5')

image_number = 1 
while os.path.isfile(f"/Users/abhinav/Desktop/cseProject/Digits/{image_number}.png"):
	try:
		img = cv2.imread(f"/Users/abhinav/Desktop/cseProject/Digits/{image_number}.png")[:,:,0]
		img = np.invert(np.array([img]))
		prediction = model.predict(img)
		print(f"The result's probably a {np.argmax(prediction)}")
		plt.imshow(img[0], cmap=plt.cm.binary)
		plt.show()
	except:
		print("Error!")
	finally:
		image_number += 1