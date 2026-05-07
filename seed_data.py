import os
import uuid
from datetime import datetime, timedelta
from app import app, db, PredictionHistory, User

# Ensure context
app.app_context().push()

def seed_data():
    print("🌱 Seeding database with demo data...")
    
    # 1. Create Demo Doctor/Admin User
    if not User.query.filter_by(username='doctor').first():
        user = User(username='doctor', role='doctor')
        user.set_password('password')
        db.session.add(user)
        print("Created user: doctor / password")
    if not User.query.filter_by(username='admin').first():
        admin = User(username='admin', role='admin')
        admin.set_password('admin123')
        db.session.add(admin)
        print("Created admin: admin / admin123")

    # 2. Create Patient 'John Doe' History (Progression: Mild -> Severe)
    
    # Visit 1: 6 months ago - Mild DR
    case_id_1 = str(uuid.uuid4())
    date_1 = datetime.utcnow() - timedelta(days=180)
    
    entry_1_left = PredictionHistory(
        case_id=case_id_1,
        filename='demo_mild_left.jpg',
        original_path='uploads/demo_mild_left.jpg', # Assuming these exist or placeholder
        heatmap_path='heatmaps/demo_heatmap_1.jpg',
        prediction='Mild',
        confidence=88.5,
        timestamp=date_1,
        patient_name='John Doe',
        patient_age=55,
        eye_side='Left',
        enhancement_used='None',
        quality_score=95,
        quality_issues='None'
    )
    
    # Visit 2: 3 months ago - Moderate DR
    case_id_2 = str(uuid.uuid4())
    date_2 = datetime.utcnow() - timedelta(days=90)
    
    entry_2_left = PredictionHistory(
        case_id=case_id_2,
        filename='demo_mod_left.jpg',
        original_path='uploads/demo_mod_left.jpg',
        heatmap_path='heatmaps/demo_heatmap_2.jpg',
        prediction='Moderate',
        confidence=92.1,
        timestamp=date_2,
        patient_name='John Doe',
        patient_age=55,
        eye_side='Left',
        enhancement_used='CLAHE',
        quality_score=92,
        quality_issues='None'
    )

    # Visit 3: Today - Severe DR (with Quality Warning)
    case_id_3 = str(uuid.uuid4())
    date_3 = datetime.utcnow()
    
    entry_3_left = PredictionHistory(
        case_id=case_id_3,
        filename='demo_severe_left.jpg',
        original_path='uploads/demo_severe_left.jpg',
        heatmap_path='heatmaps/demo_heatmap_3.jpg',
        prediction='Severe',
        confidence=96.4,
        timestamp=date_3,
        patient_name='John Doe',
        patient_age=56,
        eye_side='Left',
        enhancement_used='None',
        quality_score=45,
        quality_issues='Blurry, Too Dark' # Triggers alert
    )
    
    # Add all to session
    db.session.add(entry_1_left)
    db.session.add(entry_2_left)
    db.session.add(entry_3_left)
    
    db.session.commit()
    print(f"✅ Seeding complete! Added history for 'John Doe'.")
    print(f"   - Visit 1: {date_1.date()} (Mild)")
    print(f"   - Visit 2: {date_2.date()} (Moderate)")
    print(f"   - Visit 3: {date_3.date()} (Severe + Quality Warning)")

if __name__ == '__main__':
    seed_data()
