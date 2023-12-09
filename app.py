from flask import  Flask, render_template, request, jsonify
from PIL import Image 
import numpy as np
import tensorflow as tf

app = Flask(__name__)

model = tf.keras.models.load_model('model.h5')

target_size =(224, 224)

def preprocess_image(image):
    img = Image.open(image)
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)    
    return img_array

    
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])

def upload():
    
    if 'file' not in request.files:
        return jsonify({'error': 'no file'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': "file name is empty"})
    
    try:
        img_array = preprocess_image(file)
        
        predictions = model.predict(img_array)
        class_index = np.argmax(predictions[0])
        if class_index == 0:
            result = 'Cat'
        else:
            result = 'Dog'
            
        return jsonify({'class_name': result})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
    