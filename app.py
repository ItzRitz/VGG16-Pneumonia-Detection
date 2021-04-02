from flask import Flask, render_template, request
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.models import load_model
from uuid import uuid4

app = Flask(__name__)
app.config['IMAGE_UPLOADS'] = 'static/'
cnn = load_model('saved_model/trained_model.h5')
image_name = ''

def predict_class():
	global image_name
	PATH = app.config['IMAGE_UPLOADS']
	upload_image = False
	for images in os.listdir(PATH):
		if images == image_name:
			upload_image = True
			img = load_img(f"{PATH}{images}", color_mode = "grayscale", target_size=(200,200))
			img_arr = img_to_array(img)
			img_arr = img_arr/255
			img_arr = np.expand_dims(img_arr, 0)
			prediction = cnn.predict_classes(img_arr)
	return prediction[0][0]


@app.route('/')
def home():
   return render_template('upload.html')
	
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_files():
	global image_name
	if request.method == 'POST':
		try:
			f = request.files['file']
			image_name = str(uuid4())+f.filename
			f.save(os.path.join(app.config['IMAGE_UPLOADS'], image_name))
			return render_template('predict.html', prediction=predict_class(), name=image_name)
		except:
			return render_template('error.html')
		
if __name__ == '__main__':
   app.run(debug = False, port=8000)