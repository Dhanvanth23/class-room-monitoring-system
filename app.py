from flask import Flask, render_template, request, redirect, url_for, jsonify, session, abort
from sqlalchemy import func,create_engine
from flask_socketio import SocketIO, emit
from datetime import datetime
import os
import json
import subprocess
import time
import threading
import cv2
from database.models import db, Trainer, Session, Student, Attendance, EngagementReport
from models.FaceAnalyzer import FaceAnalyzer
from models.engagement import calculate_engagement
from models.llm import analyze_classroom
import pandas as pd
from contextlib import contextmanager
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.pool import QueuePool

# Flask App Initialization
app = Flask(__name__)
app.secret_key = "your_secret_key"
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
db_path = os.path.join(BASE_DIR, 'database.sqlite')
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{db_path}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_size': 10,
    'max_overflow': 20,
    'pool_timeout': 60,
    'pool_recycle': 1800,  # Recycle connections after 30 minutes
    'pool_pre_ping': True  # Enable connection health checks
}

socketio = SocketIO(app, cors_allowed_origins="*")
db.init_app(app)

# Create scoped session
with app.app_context():
    engine = create_engine(
        app.config['SQLALCHEMY_DATABASE_URI'],
        poolclass=QueuePool,
        pool_size=10,
        max_overflow=20,
        pool_timeout=60,
        pool_recycle=1800,
        pool_pre_ping=True
    )
    db_session = scoped_session(sessionmaker(bind=engine))

# File Paths
ENGAGEMENT_SCORES_FILE = "static/engagement_scores.json"
PHOTO_UPLOAD_DIR = "uploads/photos"
FACE_ANALYZER_SCRIPT = "FaceAnalyzer.py"
ENGAGEMENT_SCRIPT = "engagement.py"
last_photo_path = "output/annotated_image_1.jpg"
output_folder = "./output"

# Ensure directories exist
os.makedirs(PHOTO_UPLOAD_DIR, exist_ok=True)
os.makedirs("data", exist_ok=True)

@contextmanager
def session_scope():
    """Provide a transactional scope around a series of operations."""
    session = db_session()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

# Add explicit cleanup on application shutdown
@app.teardown_appcontext
def shutdown_session(exception=None):
    db_session.remove()
    if engine:
        engine.dispose()



# Utility Functions
def start_session_if_scheduled():
    while True:
        current_time = datetime.now().strftime("%H:%M")
        with session_scope() as session:
            session_to_start = session.query(Session).filter(
                func.strftime("%H:%M", Session.start_time) == current_time
            ).first()
            
            if session_to_start:
                start_session(session_to_start.id)
                break

def monitor_sessions():
    session_thread = threading.Thread(target=start_session_if_scheduled)
    session_thread.daemon = True
    session_thread.start()

def load_engagement_scores():
    """Load engagement scores from file."""
    if os.path.exists(ENGAGEMENT_SCORES_FILE):
        with open(ENGAGEMENT_SCORES_FILE, "r") as file:
            return json.load(file)
    return []

def capture_photo(photo_number):
    """Capture a photo using the camera and save it to the upload folder."""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        raise Exception("Could not open camera")
    
    ret, frame = cap.read()
    
    if not ret:
        cap.release()
        raise Exception("Failed to capture image")

    photo_path = os.path.join(PHOTO_UPLOAD_DIR, f"photo_{photo_number}.jpg")
    cv2.imwrite(photo_path, frame)
    cap.release()
    
    print(f"Photo {photo_number} captured and saved to {photo_path}")
    return photo_path

def save_engagement_scores(scores):
    """Save engagement scores to file."""
    with open(ENGAGEMENT_SCORES_FILE, "w") as file:
        json.dump(scores, file, indent=4)

def analyze_photo(photo_path):
    """Analyze a photo using FaceAnalyzer.py and engagement.py."""
    try:
        # Step 1: Run FaceAnalyzer.py
        face_analyzer_command = ["python3", FACE_ANALYZER_SCRIPT, photo_path]
        subprocess.run(face_analyzer_command, check=True)

        # Step 2: Run engagement.py
        engagement_command = ["python3", ENGAGEMENT_SCRIPT]
        result = subprocess.run(engagement_command, capture_output=True, text=True, check=True)

        return float(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        return f"Error during photo analysis: {e}"

# Routes for Classroom Monitoring
@app.route("/", methods=["GET", "POST"])
def login():
    """Handle login."""
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if username == "teacher@gmail.com" and password == "password123":
            session["user"] = username
            return redirect(url_for("main"))
        else:
            return render_template("login.html", error="Invalid credentials")
    return render_template("login.html")

@app.route("/home", methods=["GET", "POST"])
def main():
    # Hardcoded institution details
    institution_details = {
        "name": "KGiSL",
        "location": "Coimbatore",
        "contact": "+91 12345 67890",
    }

    with session_scope() as session:
        # Calculate institution engagement score (average of scores in EngagementReport table)
        institution_engagement_score = session.query(
            func.avg(EngagementReport.score)
        ).scalar()
        institution_engagement_score = round(institution_engagement_score, 2) if institution_engagement_score else 0.0

        # Get total number of sessions
        total_sessions = session.query(Session).count()

        # Get all trainers
        trainers = session.query(Trainer).all()
        trainers_list = [{"id": t.id, "name": t.name, "email": t.email, "expertise": t.expertise} for t in trainers]


    return render_template(
        "home.html",
        institution=institution_details,
        engagement_score=institution_engagement_score,
        total_sessions=total_sessions,
        trainers=trainers_list,
    )



@app.route("/dashboard")
def dashboard():
    """Dashboard showing engagement scores."""
    if "user" not in session:
        return redirect(url_for("login"))
    engagement_scores = load_engagement_scores()
    return render_template("dashboard.html", scores=engagement_scores)

@app.route("/start-session", methods=['GET', "POST"])
def start_session(session_id):
    """Start the session and analyze engagement at regular intervals."""
    engagement_scores = []
    
    with session_scope() as db_session:
        session_to_start = db_session.query(Session).get(session_id)
        start = datetime.strptime(session_to_start.start_time, "%H:%M")
        end = datetime.strptime(session_to_start.end_time, "%H:%M")
        minutes = (end - start).seconds // 60

        for i in range(minutes):
            photo_path = capture_photo(i + 1)
            face_analyzer = FaceAnalyzer(photo_path)
            num_faces = face_analyzer.analyze_faces()
            
            face_data_csv = os.path.join(output_folder, f"face_data_{i + 1}.csv")
            face_analyzer.save_results(
                output_image_path=f"{output_folder}/annotated_image_{i + 1}.jpg",
                output_csv_path=face_data_csv
            )
            
            engagement_df, overall_engagement_score = calculate_engagement(face_data_csv)
            
            print(f"Engagement Score for photo {i + 1}: {overall_engagement_score}")
            print(engagement_df)

            engagement_scores.append(overall_engagement_score)
            socketio.emit("score_update", {
                "photo_number": i + 1,
                "engagement_score": overall_engagement_score
            })
            
            if i < 1:
                print("Waiting for 1 minute before the next capture...")
                time.sleep(60)
        
        session_engagement_score = sum(engagement_scores) / len(engagement_scores) if engagement_scores else 0

        all_scores = load_engagement_scores()
        all_scores.append({
            "session": f"Session {len(all_scores) + 1}",
            "score": session_engagement_score
        })
        save_engagement_scores(all_scores)

        try:
            llm_response = analyze_classroom(photo_path)
            llm_content = llm_response["choices"][0]["message"]["content"]
            
            report_path = "static/classroom_analysis_report.json"
            with open(report_path, "w") as f:
                json.dump(llm_response, f, indent=4)

            engagement_report = EngagementReport(
                session_id=session_id,
                score=session_engagement_score,
                report_content=llm_content,
            )
            db_session.add(engagement_report)

        except Exception as e:
            return jsonify({"success": False, "error": f"LLM Analysis failed: {str(e)}"})

@app.route('/session_report')
def session_report():
    with session_scope() as session:
        sessions = session.query(Session).all()
        return render_template('session_report.html', sessions=sessions)

@app.route('/view_report/<int:session_id>')
def view_report(session_id):
    with session_scope() as session:
        session_data = session.query(Session).get(session_id)
        if session_data is None:
            abort(404, description=f"Session with ID {session_id} not found")

        report = session.query(EngagementReport).filter_by(session_id=session_id).first()
        if not report:
            return jsonify({"error": "Engagement report not found"}), 404

        face_data_csv = os.path.join(output_folder, f"face_data_1.csv")
        if not os.path.exists(face_data_csv):
            return jsonify({"error": "Face data not found"}), 404

        face_data = pd.read_csv(face_data_csv)
        engagement_df, institution_engagement_score = calculate_engagement(face_data_csv)

        overall_engagement_score = session.query(
            func.avg(EngagementReport.score)
        ).filter_by(session_id=session_id).scalar()

        return render_template(
            'view_report.html',
            session=session_data,
            engagement_report=report,
            face_data=face_data.to_dict(orient='records'),
            engagement_df=engagement_df.to_dict(orient='records'),
            overall_engagement_score=overall_engagement_score,
        )

@app.route("/logout")
def logout():
    """Logout user."""
    session.pop("user", None)
    return redirect(url_for("login"))



@app.route('/schedule', methods=['GET', 'POST'])
def schedule_session():
    if request.method == 'POST':
        session_data = request.form
        session_date = datetime.strptime(session_data['date'], '%Y-%m-%d').date()

        with session_scope() as session:
            new_session = Session(
                name=session_data['name'],
                trainer_id=session_data['trainer_id'],
                date=session_date,
                start_time=session_data['start_time'],
                end_time=session_data['end_time']
            )
            session.add(new_session)
    
    with session_scope() as session:
        trainers = session.query(Trainer).all()
        return render_template('schedule_sessions.html', trainers=trainers)

@app.route('/add-trainer', methods=['GET', 'POST'])
def add_trainer():
    if request.method == 'POST':
        trainer_data = request.form
        with session_scope() as session:
            new_trainer = Trainer(
                name=trainer_data['name'],
                email=trainer_data['email'],
                expertise=trainer_data['expertise']
            )
            session.add(new_trainer)
            return redirect(url_for('main'))
    return render_template('add_trainer.html')

@app.route('/students', methods=['GET', 'POST'])
def manage_students():
    if request.method == 'POST':
        student_data = request.form
        with session_scope() as session:
            new_student = Student(
                name=student_data['name'],
                email=student_data['email']
            )
            session.add(new_student)
    
    with session_scope() as session:
        students = session.query(Student).all()
        return render_template('student_management.html', students=students)

@app.route('/attendance')
def attendance():
    # Calculate attendance metrics
    total_students = Student.query.count()
    total_sessions = Session.query.count()
    total_attendance = Attendance.query.filter_by(present=True).count()

    # Calculate overall attendance percentage
    overall_attendance = (total_attendance / (total_students * total_sessions)) * 100 if total_sessions > 0 else 0

    return render_template(
        'attendance_dashboard.html',
        total_students=total_students,
        total_sessions=total_sessions,
        overall_attendance=round(overall_attendance, 2)
    )

@app.route('/mark-attendance', methods=['GET', 'POST'])
def mark_attendance():
    if request.method == 'POST':
        session_id = request.form['session_id']  # Get session ID from the form
        student_attendance = request.form.getlist('attendance')  # List of student IDs marked as present

        # Add attendance records for the selected session
        for student_id in student_attendance:
            # Check if the attendance record already exists for the session and student
            existing_record = Attendance.query.filter_by(session_id=int(session_id), student_id=int(student_id)).first()
            
            if not existing_record:
                # If no existing record, create a new attendance record
                attendance = Attendance(
                    session_id=int(session_id),
                    student_id=int(student_id),
                    present=True
                )
                db.session.add(attendance)

        db.session.commit()
        return redirect('/attendance-report')

    # Fetch all sessions and students for the form
    sessions = Session.query.order_by(Session.name, Session.date).all()
    students = Student.query.all()

    return render_template('mark_attendance.html', sessions=sessions, students=students)


@app.route('/attendance-report', methods=['GET'])
def attendance_report():
    # Get all sessions
    sessions = Session.query.order_by(Session.date).all()

    # Check if a specific session is selected via query parameter
    session_id = request.args.get('session_id', type=int)
    selected_session = None
    attendance_data = []

    if session_id:
        # Fetch attendance data for the selected session
        selected_session = Session.query.get(session_id)
        attendance_records = (
            db.session.query(Attendance, Student)
            .join(Student, Attendance.student_id == Student.id)
            .filter(Attendance.session_id == session_id)
            .all()
        )

        # Format attendance records for the template
        attendance_data = [
            {
                "student_name": record.Student.name,
                "present": record.Attendance.present
            }
            for record in attendance_records
        ]
    else:
        selected_session = None
        attendance_data = None

    return render_template(
        'attendance_report.html',
        sessions=sessions,
        selected_session=selected_session,
        attendance_records=attendance_data
    )



@app.route("/exam-mg")
def examManagement():
    return render_template("examManagement.html")



if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    monitor_sessions()
    socketio.run(app, debug=True)