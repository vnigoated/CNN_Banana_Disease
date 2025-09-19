

IMG_SIZE = 150

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model('banana_disease_classifier.h5')

img = image.load_img('C:\\myprograms\\cnn-banana\\dataset\\validation\\banana_insect_pest_disease\\Augmented Banana Insect Pest Disease (33).jpg', target_size=(IMG_SIZE, IMG_SIZE))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

pred = model.predict(img_array)
class_idx = np.argmax(pred)


class_labels = [
	'banana_black_sigatoka_disease',
	'banana_bract_mosaic_virus_disease',
	'banana_healthy_leaf',
	'banana_insect_pest_disease',
	'banana_moko_disease',
	'banana_panama_disease',
	'banana_yellow_sigatoka_disease'
]
print(f'Predicted Class: {class_labels[class_idx]}')
