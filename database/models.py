from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlite3 import Connection as SQLite3Connection

db = SQLAlchemy()

# SQLite connection handling
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    if isinstance(dbapi_connection, SQLite3Connection):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA busy_timeout = 60000")  # Set timeout to 60 seconds
        cursor.close()


# Trainer model
class Trainer(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False, unique=True)
    trainer_id = db.Column(db.String(50), nullable=False, unique=True)
    sessions = db.relationship('Session', backref='trainer', lazy=True)

# Session model
class Session(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    date = db.Column(db.Date, nullable=False)
    start_time = db.Column(db.String(5), nullable=False)
    end_time = db.Column(db.String(5), nullable=False)
    trainer_id = db.Column(db.Integer, db.ForeignKey('trainer.id'), nullable=False)
    students = db.relationship('Student', secondary='attendance', backref='sessions', lazy='dynamic', overlaps='attendances,session,student')
    engagement_reports = db.relationship('EngagementReport', back_populates='session')

# Student model
class Student(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False, unique=True)
    Rollno = db.Column(db.String(50), nullable=False)

# Attendance model (many-to-many relationship between Students and Sessions)
class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('session.id'), nullable=False)
    student_id = db.Column(db.Integer, db.ForeignKey('student.id'), nullable=False)
    present = db.Column(db.Boolean, nullable=False)
    status = db.Column(db.String(10), nullable=False, default='present')

    session = db.relationship('Session', backref=db.backref('attendances', lazy=True), overlaps='sessions,students')
    student = db.relationship('Student', backref=db.backref('attendances', lazy=True), overlaps='sessions,students')


class EngagementReport(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('session.id'), nullable=False)
    score = db.Column(db.Float, nullable=False)
    report_content = db.Column(db.Text, nullable=False)
    date_created = db.Column(db.DateTime, default=datetime.utcnow)
    session = db.relationship('Session', back_populates='engagement_reports')
