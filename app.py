from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.image import load_img
from keras_preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
from PIL import Image
import imghdr
import os




app = Flask(__name__)
Model_Path = 'models/pneu_cnn_model.h5'
model = load_model(Model_Path)



@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'imagefile' not in request.files:
            return render_template('index.html', prediction="❌ No file part in the request.")
        imagefile = request.files['imagefile']
        if imagefile.filename == '':
            return render_template('index.html', prediction="❌ No file selected.")

        image_path = os.path.join('static', imagefile.filename)
        imagefile.save(image_path)

        # Validate image type using imghdr
        if imghdr.what(image_path) not in ['jpeg', 'png', 'bmp']:
            try:
                os.remove(image_path)
            except PermissionError:
                pass  # Ignore the error and move on
            return render_template('index.html', prediction="❌ Please upload a valid X-ray image (JPEG/PNG/BMP).")

        # Validate image mode
        try:
            with Image.open(image_path) as img_pil:
                if img_pil.mode != 'L':
                    try:
                        os.remove(image_path)
                    except PermissionError:
                        pass
                    return render_template('index.html', prediction="❌Please upload a valid images, Only grayscale X-ray images are allowed.")
        except Exception:
            try:
                os.remove(image_path)
            except PermissionError:
                pass
            return render_template('index.html', prediction="❌ Error reading the image. Please try again.")

        # Preprocess and predict
        img = load_img(image_path, target_size=(500, 500), color_mode='grayscale')
        x = img_to_array(img) / 255.0
        x = np.expand_dims(x, axis=0)

        classes = model.predict(x)
        result = classes[0][0]
        label = 'Positive' if result >= 0.5 else 'Negative'
        prediction = '%s (%.2f%%)' % (label, result * 100)

        return render_template(
     'index.html',
    prediction=prediction,
    imagePath=image_path,
    accuracy=round(result * 100, 2)
)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(port=5000, debug=True)