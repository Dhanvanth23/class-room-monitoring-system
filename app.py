from flask import Flask, render_template, request, redirect, url_for, jsonify, session, abort
import secrets
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
os.makedirs(output_folder, exist_ok=True)

# In-memory store for password reset tokens: { token: email }
reset_tokens = {}

# Hardcoded credentials (matches existing login logic)
USERS = {"teacher@gmail.com": "password123"}

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
    with app.app_context():
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

def capture_photo(photo_number, max_retries=3):
    """Capture a photo using the camera and save it to the upload folder."""
    for attempt in range(max_retries):
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print(f"[WARN] Camera not available (attempt {attempt + 1}/{max_retries})")
            time.sleep(2)
            continue
        
        # Allow the camera to warm up by reading a few frames
        for _ in range(5):
            cap.read()
            time.sleep(0.1)
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            print(f"[WARN] Failed to capture image (attempt {attempt + 1}/{max_retries})")
            time.sleep(2)
            continue

        photo_path = os.path.join(PHOTO_UPLOAD_DIR, f"photo_{photo_number}.jpg")
        cv2.imwrite(photo_path, frame)
        
        print(f"Photo {photo_number} captured and saved to {photo_path}")
        return photo_path
    
    raise Exception(f"Failed to capture image after {max_retries} attempts. Check that a camera is connected and not in use by another application.")

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
@app.route("/login", methods=["GET", "POST"])
def login():
    """Handle login."""
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if USERS.get(username) == password:
            session["user"] = username
            return redirect(url_for("main"))
        else:
            return render_template("login.html", error="Invalid credentials")
    return render_template("login.html")

@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    """Show forgot-password form; on POST generate reset token."""
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        # Generate token regardless of whether email exists (security best-practice)
        token = secrets.token_urlsafe(32)
        reset_tokens[token] = email
        reset_url = url_for('reset_password', token=token, _external=True)
        return render_template("email_sent.html", email=email, reset_url=reset_url)
    return render_template("forgot_password.html")

@app.route("/reset-password/<token>", methods=["GET", "POST"])
def reset_password(token):
    """Validate token and let user set a new password."""
    email = reset_tokens.get(token)
    if not email:
        return render_template("login.html", error="Invalid or expired reset link.")
    if request.method == "POST":
        new_password = request.form.get("password", "")
        if email in USERS:
            USERS[email] = new_password
        reset_tokens.pop(token, None)
        return redirect(url_for("login"))
    return render_template("reset_password.html", token=token)

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
        trainers_list = [{"id": t.id, "trainer_id": t.trainer_id, "name": t.name, "email": t.email} for t in trainers]


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
        session_to_start = db_session.get(Session, session_id)
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
        
        # Filter out NaN scores from failed analyses
        import math
        valid_scores = [s for s in engagement_scores if not (isinstance(s, float) and math.isnan(s))]
        session_engagement_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

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
            print(f"[ERROR] LLM Analysis failed for session {session_id}: {str(e)}")
            # Still save a report with just the engagement score (no LLM content)
            engagement_report = EngagementReport(
                session_id=session_id,
                score=session_engagement_score,
                report_content=f"LLM analysis unavailable: {str(e)}",
            )
            db_session.add(engagement_report)

@app.route('/session_report')
def session_report():
    with session_scope() as session:
        sessions = session.query(Session).all()
        return render_template('session_report.html', sessions=sessions)

@app.route('/view_report/<int:session_id>')
def view_report(session_id):
    with session_scope() as session:
        session_data = session.get(Session, session_id)
        if session_data is None:
            abort(404, description=f"Session with ID {session_id} not found")

        report = session.query(EngagementReport).filter_by(session_id=session_id).first()

        overall_engagement_score = session.query(
            func.avg(EngagementReport.score)
        ).filter_by(session_id=session_id).scalar()

        # Try loading face data; fall back to empty lists if unavailable
        face_data_csv = os.path.join(output_folder, f"face_data_1.csv")
        face_data = []
        engagement_data = []
        if os.path.exists(face_data_csv):
            try:
                face_data_df = pd.read_csv(face_data_csv)
                engagement_df, _ = calculate_engagement(face_data_csv)
                face_data = face_data_df.to_dict(orient='records')
                engagement_data = engagement_df.to_dict(orient='records')
            except Exception as e:
                print(f"[WARN] Could not load face data: {e}")

        return render_template(
            'view_report.html',
            session=session_data,
            engagement_report=report,
            face_data=face_data,
            engagement_df=engagement_data,
            overall_engagement_score=overall_engagement_score or 0,
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
                trainer_id=trainer_data['trainer_id'],
                name=trainer_data['name'],
                email=trainer_data['email']
            )
            session.add(new_trainer)
            return redirect(url_for('add_trainer'))
    with session_scope() as session:
        trainers = session.query(Trainer).all()
        return render_template('add_trainer.html', trainers=trainers)

@app.route('/students', methods=['GET', 'POST'])
def manage_students():
    if request.method == 'POST':
        student_data = request.form
        with session_scope() as session:
            new_student = Student(
                name=student_data['name'],
                email=student_data['email'],
                Rollno=student_data['Rollno']
            )
            session.add(new_student)
    
    with session_scope() as session:
        students = session.query(Student).all()
        return render_template('student_management.html', students=students)

@app.route('/attendance')
def attendance():
    with session_scope() as sess:
        total_students = sess.query(Student).count()
        total_sessions = sess.query(Session).count()
        total_attendance = sess.query(Attendance).filter_by(present=True).count()

        if total_students > 0 and total_sessions > 0:
            overall_attendance = (total_attendance / (total_students * total_sessions)) * 100
        else:
            overall_attendance = 0

        return render_template(
            'attendance_dashboard.html',
            total_students=total_students,
            total_sessions=total_sessions,
            overall_attendance=round(overall_attendance, 2)
        )

@app.route('/mark-attendance', methods=['GET'])
def mark_attendance():
    with session_scope() as sess:
        sessions = sess.query(Session).order_by(Session.name, Session.date).all()
        students = sess.query(Student).all()
        return render_template('mark_attendance.html', sessions=sessions, students=students)


@app.route('/api/mark-attendance', methods=['POST'])
def api_mark_attendance():
    """JSON API endpoint for saving attendance from the mark_attendance page."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        sid = data.get('session_id')
        records = data.get('attendance', [])

        if not sid:
            return jsonify({'success': False, 'error': 'session_id is required'}), 400

        with session_scope() as sess:
            for record in records:
                student_id = record.get('student_id')
                status = record.get('status', 'present')
                is_present = status == 'present'

                existing = sess.query(Attendance).filter_by(
                    session_id=int(sid), student_id=int(student_id)
                ).first()

                if existing:
                    existing.present = is_present
                    existing.status = status
                else:
                    new_att = Attendance(
                        session_id=int(sid),
                        student_id=int(student_id),
                        present=is_present,
                        status=status
                    )
                    sess.add(new_att)

        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/get-attendance/<int:session_id>', methods=['GET'])
def api_get_attendance(session_id):
    """JSON API endpoint for loading saved attendance for a session."""
    try:
        with session_scope() as sess:
            records = sess.query(Attendance).filter_by(session_id=session_id).all()
            attendance_map = {}
            for r in records:
                attendance_map[str(r.student_id)] = r.status if r.status else ('present' if r.present else 'absent')

        return jsonify({'success': True, 'attendance': attendance_map})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/attendance-report', methods=['GET'])
def attendance_report():
    with session_scope() as sess:
        sessions = sess.query(Session).order_by(Session.date).all()

        session_id = request.args.get('session_id', type=int)
        selected_session = None
        attendance_data = []

        if session_id:
            selected_session = sess.get(Session, session_id)
            attendance_records = (
                sess.query(Attendance, Student)
                .join(Student, Attendance.student_id == Student.id)
                .filter(Attendance.session_id == session_id)
                .all()
            )

            attendance_data = [
                {
                    "student_name": record.Student.name,
                    "roll_no": record.Student.Rollno,
                    "status": record.Attendance.status if record.Attendance.status else ('present' if record.Attendance.present else 'absent')
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
        # Migrate: add 'status' column to attendance table if it doesn't exist
        from sqlalchemy import inspect, text
        insp = inspect(db.engine)
        columns = [col['name'] for col in insp.get_columns('attendance')]
        if 'status' not in columns:
            with db.engine.connect() as conn:
                conn.execute(text("ALTER TABLE attendance ADD COLUMN status VARCHAR(10) NOT NULL DEFAULT 'present'"))
                conn.commit()
            print("[MIGRATION] Added 'status' column to attendance table.")
    monitor_sessions()
    socketio.run(app, debug=True)