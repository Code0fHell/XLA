import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from werkzeug.utils import secure_filename

# Khởi tạo Flask
app = Flask(__name__)

# Cấu hình đường dẫn lưu trữ ảnh upload
UPLOAD_FOLDER = './static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load mô hình đã huấn luyện
model = load_model('./model/model.h5')

# Các nhãn cảm xúc
emotion_labels = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']

# Khởi tạo bộ phát hiện khuôn mặt
face_classifier = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')

def detect_emotion_in_image(image_path):
    frame = cv2.imread(image_path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        roi_rgb = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2RGB)
        roi_rgb = cv2.resize(roi_rgb, (224, 224), interpolation=cv2.INTER_AREA)
        
        if np.sum([roi_rgb]) != 0:
            roi = tf.keras.applications.efficientnet.preprocess_input(roi_rgb)
            roi = np.expand_dims(roi, axis=0)
            prediction = model.predict(roi)[0]
            emotion = emotion_labels[prediction.argmax()]
            label_position = (x, y - 10)
            cv2.putText(frame, emotion, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'No Face Found', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result.jpg')
    cv2.imwrite(output_path, frame)
    return output_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Nhận diện cảm xúc
        result_image_path = detect_emotion_in_image(file_path)
        
        return render_template('result.html', result_image=result_image_path)

if __name__ == '__main__':
    app.run(debug=True)
