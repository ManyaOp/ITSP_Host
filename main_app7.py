from flask import Flask, request, jsonify, render_template, Response,session,redirect , url_for
from flask_mysqldb import MySQL
import sqlite3
import cv2
import numpy as np
import dlib
import os
import base64
import time
import threading
import queue
import logging
from logging.handlers import RotatingFileHandler
import MySQLdb.cursors, re, time, bcrypt, jwt, os, ffmpeg, subprocess
from werkzeug.utils import secure_filename
from PIL import Image
from moviepy.editor import VideoFileClip, AudioFileClip, ImageSequenceClip, concatenate_audioclips, ImageClip, concatenate_videoclips
from moviepy.video.fx import fadein
import psycopg2
import io
import tempfile
from functools import wraps
from flask import flash
app = Flask(__name__, static_folder='static', template_folder='templates')

app.secret_key = '880248'
app.config['MYSQL_HOST'] = 'sql6.freemysqlhosting.net' 
app.config['MYSQL_USER'] = 'sql6689061'       
app.config['MYSQL_PASSWORD'] = 'VdGbTa5KDU'  
app.config['MYSQL_DB'] = 'sql6689061'

UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 

mysql = MySQL(app)
db_url="postgresql://manya:JG_Plchon2-G_FKfN9_hDg@paper-singer-5092.7s5.aws-ap-south-1.cockroachlabs.cloud:26257/defaultdb?sslmode=verify-full"
conn = psycopg2.connect(db_url)
cursor = conn.cursor()

current = 1280

SECRET_KEY = "suspense"


def hash_password(password):
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed_password

def verify_password(hashed_password, input_password):
    if isinstance(hashed_password, str):
        hashed_password = hashed_password.encode('utf-8')
    return bcrypt.checkpw(input_password.encode('utf-8'), hashed_password)

# def verify_password(hashed_password, input_password):
#     return bcrypt.checkpw(input_password.encode('utf-8'), hashed_password)
# def verify_password(hashed_password, input_password):
#     hashed_password_bytes = bytes(hashed_password)
#     return bcrypt.checkpw(input_password.encode('utf-8'), hashed_password_bytes)
# def verify_password(hashed_password, input_password):
#     if isinstance(hashed_password, str):
#         hashed_password = hashed_password.encode('utf-8')
#     return bcrypt.checkpw(input_password.encode('utf-8'), hashed_password)


# def create_jwt_token(name, username, email, id):
#     payload = {"name": name, "username": username, "email": email, "id" : id}
#     token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
#     return token
def create_jwt_token(name, username, email, id):
    payload = {
        "name": str(name),
        "username": str(username),
        "email": str(email),
        "id": str(id)
    }
    try:
        token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
        if isinstance(token, bytes):
            return token.decode('utf-8')
        return token
    except Exception as e:
        print(f"Error creating JWT token: {str(e)}")
        return None

def decode_jwt_token(token):
    try:
        decoded_token = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return decoded_token
    except jwt.ExpiredSignatureError:
        return "Signature expired. Please log in again."
    except jwt.InvalidTokenError:
        return "Invalid token. Please log in again."

def get_last_part_after_slash(input_string):
    result = ""

    for char in reversed(input_string):
        if char == '/':
            break
        result = char + result

    return result

def verify_user(username, password):
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM kaptaan WHERE username = %s', (username,))
    user = cursor.fetchone()
    cursor.close()
    if user:
        if verify_password(user['password'], password):
            return True
    return False

def init_db():
    conn = sqlite3.connect('location.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS location
                      (id INTEGER PRIMARY KEY, latitude REAL, longitude REAL, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
log_file = 'app.log'
file_handler = RotatingFileHandler(log_file, maxBytes=1024*1024, backupCount=10)
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO)
app.logger.setLevel(logging.INFO)
app.logger.addHandler(file_handler)

facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

face_descriptors = {}
labels = []

unrecognized_face = None
video_capture = None
frame_queue = queue.Queue(maxsize=1)
event_queue = queue.Queue()
capture_running = False
input_requested = False  
current_user_id = None
last_unrecognized_time = 0
last_input_prompt_time = 0

images_dir = "Face Photo"

def load_face_descriptors(user_id=None):
    global face_descriptors, labels
    
    if user_id is None:
        return
    
    cursor = conn.cursor()
    cursor.execute("SELECT name, face_descriptor FROM face_data WHERE user_id = %s", (user_id,))
    rows = cursor.fetchall()
    cursor.close()

    for row in rows:
        name, face_descriptor_bytes = row
        face_descriptor_np = np.frombuffer(face_descriptor_bytes, dtype=np.float64)
        
        if name not in face_descriptors:
            face_descriptors[name] = []
        
        face_descriptors[name].append(face_descriptor_np)
        if name not in labels:
            labels.append(name)

    app.logger.info(f"Loaded {len(labels)} face descriptors for user {user_id}")

def compare_faces(face_descriptor, tolerance=0.5):
    global face_descriptors
    recognized_labels = []
    for label, descriptors in face_descriptors.items():
        if descriptors:
            distances = np.linalg.norm(np.array(descriptors) - face_descriptor, axis=1)
            min_distance = np.min(distances)
            if min_distance <= tolerance:
                recognized_labels.append((label, min_distance))
    
    if recognized_labels:
        recognized_labels.sort(key=lambda x: x[1])  # Sort by distance
        return recognized_labels[0][0]  # Return the label with the smallest distance
    else:
        return "Not Recognized"

def capture_video():
    global capture_running, unrecognized_face, last_unrecognized_time, current_user_id
    cap = cv2.VideoCapture(0)
    while capture_running:
        success, frame = cap.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            for face in faces:
                shape = predictor(frame, face)
                face_descriptor = facerec.compute_face_descriptor(frame, shape)
                face_descriptor = np.array(face_descriptor)
                
                label = "Not Recognized"
                if current_user_id is not None:  # Only perform recognition if user is logged in
                    label = compare_faces(face_descriptor)
                
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                color = (0, 255, 0) if label != "Not Recognized" else (0, 0, 255)
                x_new = max(0, x - w // 2)
                y_new = max(0, y - h // 2)

                w *= 2
                h *= 2

                cv2.rectangle(frame, (x_new, y_new), (x_new + w, y_new + h), color, 2)

                cv2.putText(frame, label, (x_new, y_new - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                if label == "Not Recognized":
                    current_time = time.time()
                    if current_time - last_unrecognized_time >= 5:
                        if frame_queue.qsize() < 1:  
                            unrecognized_face = frame[y_new:y_new+h, x_new:x_new+w].copy()  
                            last_unrecognized_time = current_time
                            event_queue.put("unrecognized")  
                        else:
                            app.logger.warning("Reached maximum queued photos limit. Ignoring new unrecognized face.")

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            if not frame_queue.full():
                frame_queue.put(frame)

    cap.release()

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'id' not in session:
            flash('You must be logged in to access this page.', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


@app.route('/home', methods=['GET', 'POST'])
@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    global current_user_id
    message = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        username = request.form['username']
        password = request.form['password']
        print(password)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM kaptaan WHERE username = %s', (username,))
        user = cursor.fetchone()
        
        if user and verify_password(user[4], password):
            jwt_token = create_jwt_token(user[1], user[3], user[2], user[0])
            session['jwt_token'] = jwt_token
            session['username'] = user[3]
            session['email'] = user[2]
            session['name'] = user[1]
            session['id'] = user[0]
            current_user_id = user[0]  # Set the global variable
            message = 'Logged in successfully!'
            cursor.close()
            
            # Load face descriptors for the logged-in user
            load_face_descriptors(current_user_id)
            
            return render_template('home.html', username=session['username'], email=session['email'], name=session['name'])
        else:
            message = 'Please enter correct username / password!'
            cursor.close()
            return render_template('login.html', message=message)
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    message = ''
    if request.method == 'POST' and 'name' in request.form and 'password' in request.form and 'email' in request.form and 'username' in request.form :
        name = request.form['name']
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        hashed_password = hash_password(password).decode('utf-8')  # Add .decode('utf-8') here
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM kaptaan WHERE username = %s', (username,))
        account = cursor.fetchone()
        if account:
            message = 'Account already exists !'
            return render_template('signup.html', message=message)
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            message = 'Invalid email address !'
            return render_template('signup.html', message=message)
        elif not username or not password or not email or not name:
            message = 'Please fill out the form !'
            return render_template('signup.html', message=message)
        else:
            cursor.execute('INSERT INTO kaptaan (name, email, username, password) VALUES (%s, %s, %s, %s)',
                       (name, email, username, hashed_password,))
            conn.commit()
            cursor.close()
            message = 'You have successfully registered !'
            return render_template('login.html', message=message)
    elif request.method == 'POST':
        message = 'Please fill out the form !'
        cursor.close()
        return render_template('signup.html',message=message)
    return render_template('signup.html')


@app.route('/logout')
def logout():
    global current_user_id, face_descriptors, labels

    if current_user_id:
        # Clear location data from the database
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM user_locations WHERE user_id = %s", (current_user_id,))
            conn.commit()
        except Exception as e:
            conn.rollback()
            app.logger.error(f"Error clearing location data: {str(e)}")
        finally:
            cursor.close()

    # Clear face recognition data
    face_descriptors.clear()
    labels.clear()

    # Clear all session data
    session.clear()

    # Reset the current user ID
    current_user_id = None

    # Redirect to the home page
    return redirect(url_for('home'))

@app.route('/contact_us')
def contact():
    return render_template('contact_us.html')

@app.route('/face_rec')
@login_required
def index():
    global input_requested
    input_requested = True  
    app.logger.info('Accessed index page')
    return render_template("index4.html")

@app.route('/map')
@login_required
def map_index():
    return render_template('map_index.html')

@app.route('/share_location')
@login_required
def share_location():
    return render_template('share_location.html')

@app.route('/update_location', methods=['POST'])
def update_location():
    data = request.get_json()
    user_id = data.get('userId')
    latitude = data.get('latitude')
    longitude = data.get('longitude')

    if not all([user_id, latitude, longitude]):
        return jsonify({'error': 'Missing required data'}), 400

    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO user_locations (user_id, latitude, longitude)
        VALUES (%s, %s, %s)
    """, (user_id, latitude, longitude))
    conn.commit()
    cursor.close()

    return jsonify({'message': 'Location updated successfully'}), 200

@app.route('/get_user_location/<int:user_id>', methods=['GET'])
def get_user_location(user_id):
    cursor = conn.cursor()
    cursor.execute("""
        SELECT latitude, longitude, timestamp
        FROM user_locations
        WHERE user_id = %s
        ORDER BY timestamp DESC
        LIMIT 1
    """, (user_id,))
    location = cursor.fetchone()
    cursor.close()

    if location:
        return jsonify({
            'latitude': location[0],
            'longitude': location[1],
            'timestamp': location[2].isoformat()
        }), 200
    return jsonify({'error': 'No location found for user'}), 404

@app.route('/video_feed')
def video_feed():
    global capture_running, video_capture
    if not capture_running:
        video_capture = cv2.VideoCapture(0)
        capture_running = True
        threading.Thread(target=capture_video, daemon=True).start()

    def generate():
        while capture_running:
            if not frame_queue.empty():
                frame = frame_queue.get()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                time.sleep(0.1)

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/sse')
def sse():
    def event_stream():
        global last_input_prompt_time, capture_running
        while capture_running:
            try:
                event = event_queue.get(timeout=1)
                if event == "unrecognized":
                    current_time = time.time()
                    if current_time - last_input_prompt_time >= 10:
                        ret, buffer = cv2.imencode('.jpg', unrecognized_face)
                        if ret:
                            unrecognized_face_bytes = base64.b64encode(buffer).decode('utf-8')
                            yield f"event: {event}\ndata: {unrecognized_face_bytes}\n\n"
                            last_input_prompt_time = current_time
                else:
                    yield f"event: {event}\n\n"
            except queue.Empty:
                yield ": heartbeat\n\n"
        yield "event: close\n\n"
    return Response(event_stream(), mimetype='text/event-stream')

@app.route('/stop_video_feed')
def stop_video_feed():
    global capture_running
    capture_running = False
    return jsonify({"status": "success", "message": "Video capture stopped"})

@app.route('/input_name', methods=['POST'])
def input_name():
    global unrecognized_face, face_descriptors, labels, current_user_id
    name = request.form['name']
    if unrecognized_face is not None and current_user_id is not None:
        rgb_face = cv2.cvtColor(unrecognized_face, cv2.COLOR_BGR2RGB)
        faces = detector(rgb_face)
        if len(faces) == 1:
            face = faces[0]
            shape = predictor(rgb_face, face)
            face_descriptor = facerec.compute_face_descriptor(rgb_face, shape)
            face_descriptor_np = np.array(face_descriptor)
            face_descriptor_bytes = face_descriptor_np.tobytes()
            
            _, img_encoded = cv2.imencode('.jpg', unrecognized_face)
            img_bytes = img_encoded.tobytes()
            
            cursor = conn.cursor()
            cursor.execute("INSERT INTO face_data (user_id, name, face_descriptor, image_file) VALUES (%s, %s, %s, %s)",
                           (current_user_id, name, face_descriptor_bytes, img_bytes))
            conn.commit()
            cursor.close()
            
            # Update in-memory face_descriptors immediately
            if name not in face_descriptors:
                face_descriptors[name] = []
            face_descriptors[name].append(face_descriptor_np)
            if name not in labels:
                labels.append(name)
            
            app.logger.info(f"New face added: {name}")
            return jsonify({"status": "success"})
        else:
            return jsonify({"status": "error", "message": "No face detected in the saved image."})
    else:
        return jsonify({"status": "error", "message": "No unrecognized face was captured or user not logged in."})
@app.route('/ignore_face', methods=['POST'])
def ignore_face():
    global unrecognized_face
    unrecognized_face = None
    return jsonify({"status": "ignored"})

init_db()
def log_recognition(recognized_name, confidence):
    cursor = conn.cursor()
    cursor.execute("INSERT INTO recognition_log (user_id, recognized_name, confidence) VALUES (%s, %s, %s)",
                   (session['id'], recognized_name, confidence))
    conn.commit()
    cursor.close()
# Initialize face descriptors
# Initialize face descriptors
face_descriptors = {}
labels = []
if __name__ == '__main__':
    app.run(debug=True, threaded=True)
