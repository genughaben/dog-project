import os
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from IPython.core.display import Image, display
from flask import send_from_directory
#from logging.config import dictConfig
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

SERVER = 'http://127.0.0.1:5000/'
UPLOAD_FOLDER = os.path.join('static', 'upload')
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def hello():
    return "Hello World!"

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('image_feedback', filename=filename))
            #return redirect(url_for('uploaded_file',
            #                        filename=filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

@app.route('/')
@app.route('/image_feedback/<filename>')
def image_feedback(filename):
    filepath = UPLOAD_FOLDER + "/" + filename
    greeting = "? - You are neither a human nor a dog!"
    nof_human_faces = face_detector(filepath)
    breed = "Could not determine your breed."
    if nof_human_faces:
        greeting = "human!"
        breed = dog_breed_detector(filepath)
    elif(dog_detector(filepath)):
        greeting = "dog!"
        breed = dog_breed_detector(filepath)
    image_display_path = SERVER + filepath
    return render_template("feedback.html", user_image = image_display_path, greeting=greeting, nof_human_faces=nof_human_faces, breed_1=breed[0])
    #dog_human_dog_breed_detector(filename)


# extract pre-trained face detector

def contains_face(img_path):
    return face_detector(img_path) > 0

def face_detector(img_path):
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
    img = cv2.imread(img_path)
    app.logger.info('image path in face detector: %s', img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces)

from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
from tqdm import tqdm

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

def dog_detector(img_path):
    ResNet50_model = ResNet50(weights='imagenet')
    img = preprocess_input(path_to_tensor(img_path))
    prediction = np.argmax(ResNet50_model.predict(img))
    return ((prediction <= 268) & (prediction >= 151))

from extract_bottleneck_features import *

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.models import Sequential

def create_dog_xception_model():
    input_shape = (7, 7, 2048)
    Xception_model = Sequential()
    Xception_model.add(GlobalAveragePooling2D(input_shape=input_shape))
    Xception_model.add(Dense(133, activation='softmax'))
    return Xception_model

def top_n_indices(array, top_n):
    return n.argsort()[-top_n:][::-1]

def dog_breed_detector(img_path):
    Xception_model = create_dog_xception_model()
    Xception_model.load_weights('saved_models/weights.best.Xception_model.hdf5')
    bottleneck_feature = extract_Xception(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = Xception_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]
    
    return dog_names[np.argmax(predicted_vector)]

def dog_human_dog_breed_detector(img_path):
    if face_detector(img_path):
        print("hello, human!")
        display(Image(img_path,width=200,height=200))
        print("The dogbreed you look most similiar to is a: ", Xception_predict_breed(img_path))
    elif dog_detector(img_path):
        print("hello, dog!")
        display(Image(img_path,width=200,height=200))
        print("You are most likely an:", Xception_predict_breed(img_path))
    else:
        print("Hi ???")
        display(Image(img_path,width=200,height=200))
        print("You look neither like a dog neither like a human... maybe next time!")
