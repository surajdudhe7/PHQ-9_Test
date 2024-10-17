from flask import Flask, render_template, redirect, url_for, request
import cv2
from deepface import DeepFace
import sqlite3
from datetime import datetime
import threading
import os

app = Flask(__name__)

# PHQ-9 questions
phq9_questions = [
    "Little interest or pleasure in doing things?",
    "Feeling down, depressed, or hopeless?",
    "Trouble falling or staying asleep, or sleeping too much?",
    "Feeling tired or having little energy?",
    "Poor appetite or overeating?",
    "Feeling bad about yourself – or that you are a failure?",
    "Trouble concentrating on things, such as reading or watching television?",
    "Moving or speaking so slowly that others could have noticed?",
    "Thoughts that you would be better off dead, or of hurting yourself?"
]

# PHQ-9 responses and emotion records
phq9_responses = []
phq9_emotion_records = []
emotion_capture_active = False  # Control flag for facial emotion detection
capture_thread = None  # Thread for capturing emotions

# Initialize the database
def setup_database():
    conn = sqlite3.connect('emotion_detection.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS emotions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        angry REAL,
        disgust REAL,
        fear REAL,
        happy REAL,
        sad REAL,
        surprise REAL,
        neutral REAL,
        dominant_emotion TEXT,
        face_confidence REAL
    )''')
    conn.commit()
    conn.close()

# Insert detected emotions into the database
def insert_result(timestamp, emotions, dominant_emotion, face_confidence):
    conn = sqlite3.connect('emotion_detection.db')
    cursor = conn.cursor()
    cursor.execute('''INSERT INTO emotions (
        timestamp, angry, disgust, fear, happy, sad, surprise, neutral, dominant_emotion, face_confidence
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', (
        timestamp,
        emotions.get('angry', 0),
        emotions.get('disgust', 0),
        emotions.get('fear', 0),
        emotions.get('happy', 0),
        emotions.get('sad', 0),
        emotions.get('surprise', 0),
        emotions.get('neutral', 0),
        dominant_emotion,
        face_confidence
    ))
    conn.commit()
    conn.close()


cap = cv2.VideoCapture(0)


# Function to continuously detect emotions in a background thread
def capture_emotions():
    global emotion_capture_active


    while emotion_capture_active:
        success, frame = cap.read()
        if not success:
            break

        # Detect face in the frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_frame = frame[y:y + h, x:x + w]
            try:
                # Analyze emotions using DeepFace
                results = DeepFace.analyze(face_frame, actions=['emotion'], enforce_detection=False)
                result = results[0] if isinstance(results, list) else results
                emotions = result['emotion']
                dominant_emotion = result['dominant_emotion']
                face_confidence = result.get('face_confidence', 0)

                # Record the emotion data
                phq9_emotion_records.append(result)

                # Insert the emotion data into the database
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                insert_result(timestamp, emotions, dominant_emotion, face_confidence)

            except Exception as e:
                print(f"Emotion detection error: {str(e)}")

    cap.release()

# Start the emotion detection when the first question is accessed
@app.route('/phq9/<int:question_id>', methods=['GET', 'POST'])
def phq9(question_id):
    global emotion_capture_active, capture_thread

    if question_id == 0:
        if not emotion_capture_active:
            # Start the emotion capture in a background thread
            emotion_capture_active = True
            capture_thread = threading.Thread(target=capture_emotions)
            capture_thread.start()

        # Redirect to the first question
        return redirect(url_for('phq9', question_id=1))

    if question_id < len(phq9_questions):
        if request.method == 'POST':
            # Capture the user's response
            response = request.form.get('response')
            if response is not None:
                phq9_responses.append(int(response))

            # Redirect to the next question
            return redirect(url_for('phq9', question_id=question_id + 1))

        # Show the current question
        return render_template('phq9.html', question=phq9_questions[question_id], question_id=question_id)
    else:
        # Stop the emotion capture once the test is done
        emotion_capture_active = False
        capture_thread.join()  # Wait for the thread to finish

        # Calculate PHQ-9 score
        phq9_score = sum(phq9_responses)
        depression_level = calculate_depression_level(phq9_score)

        # Get the dominant emotion during the test
        dominant_emotion = get_dominant_emotion(phq9_emotion_records)

        # Display the results
        return render_template('result.html', depression_level=depression_level, dominant_emotion=dominant_emotion)

# Helper function to calculate depression level based on PHQ-9 score
def calculate_depression_level(score):
    if score <= 4:
        return "Minimal or None"
    elif 5 <= score <= 9:
        return "Mild"
    elif 10 <= score <= 14:
        return "Moderate"
    elif 15 <= score <= 19:
        return "Moderately Severe"
    else:
        return "Severe"

# Helper function to determine the dominant emotion
def get_dominant_emotion(emotions):
    emotion_sums = {emotion: 0 for emotion in ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']}
    for record in emotions:
        for emotion, value in record['emotion'].items():
            emotion_sums[emotion] += value
    return max(emotion_sums, key=emotion_sums.get)

# Main route for the index page
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    # Initialize the database
    setup_database()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
