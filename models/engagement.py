import pandas as pd
import numpy as np
import os

def find_common_viewpoint(csv_file):
    data = pd.read_csv(csv_file)
    
    zones = ["left", "center", "right"]
    zone_median_pose = {}

    for zone in zones:
        zone_data = data[data['zone'] == zone]
        if not zone_data.empty:
            median_pitch = zone_data['pose.pitch'].median()
            median_yaw = zone_data['pose.yaw'].median()
            median_roll = zone_data['pose.roll'].median()
            zone_median_pose[zone] = {
                "median_pitch": median_pitch,
                "median_yaw": median_yaw,
                "median_roll": median_roll
            }
    
    return zone_median_pose

def calculate_engagement(csv_file):
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"The specified file does not exist: {csv_file}")
    
    try:
        data = pd.read_csv(csv_file)
    except Exception as e:
        raise ValueError(f"Failed to read the CSV file: {csv_file}. Error: {e}")    
    data['pose.pitch'] = pd.to_numeric(data['pose.pitch'], errors='coerce')
    data['pose.yaw'] = pd.to_numeric(data['pose.yaw'], errors='coerce')
    data['pose.roll'] = pd.to_numeric(data['pose.roll'], errors='coerce')
    data['confidence'] = pd.to_numeric(data['confidence'], errors='coerce')
    
    emotion_weights = {
        "neutral": 20,    # Baseline - focused
        "happy": -5,      # Positive engagement
        "sad": 20,       # Disengagement potential
        "angry": 5,     # High disengagement
        "surprise": -10,  # Distraction potential
        "fear": -5,      # Discomfort, disengagement
        "disgust": -30,    # Likely disengagement
        "NaN": -100
    }
    
    zone_median_pose = find_common_viewpoint(csv_file)
    
    engagement_scores = []
    
    for _, row in data.iterrows():
        face_id = row["face_id"]
        zone = row["zone"]
        emotion = row["emotion"]
        confidence = row["confidence"]
        emotion_weight = emotion_weights.get(emotion, 0) * confidence  
        
        zone_pose = zone_median_pose.get(zone, {"median_pitch": 0, "median_yaw": 0})
        
        pitch_deviation = abs(row["pose.pitch"] - zone_pose["median_pitch"])
        yaw_deviation = abs(row["pose.yaw"] - zone_pose["median_yaw"])
        
        if yaw_deviation > 90:
            yaw_deviation = 100  # Assign extreme penalty for large yaw deviation
        if pitch_deviation > 100:
            pitch_deviation = 100  # Assign extreme penalty for large pitch deviation
        
        max_deviation = 45
        yaw_score = max(0, 100 - (yaw_deviation / max_deviation) * 100)
        pitch_score = max(0, 100 - (pitch_deviation / max_deviation) * 100)
        
        head_pose_score = (yaw_score * 0.7) + (pitch_score * 0.3)
        
        normalized_emotion = (emotion_weight + 50)
        normalized_emotion = np.clip(normalized_emotion, 0, 100)
        
        total_engagement_score = (head_pose_score * 0.8) + (normalized_emotion * 0.2)
        
        total_engagement_score = np.clip(total_engagement_score, 0, 100)
        
        engagement_scores.append({
            "face_id": face_id,
            "zone": zone,
            "emotion": emotion,
            "confidence": confidence,
            "emotion_weight": emotion_weight,
            "pitch_deviation": pitch_deviation,
            "yaw_deviation": yaw_deviation,
            "pitch_score": pitch_score,
            "yaw_score": yaw_score,
            "head_pose_score": head_pose_score,
            "normalized_emotion": normalized_emotion,
            "engagement_score": total_engagement_score
        })
    
    engagement_df = pd.DataFrame(engagement_scores)
    
    # Calculate overall class engagement score (average of individual engagement scores)
    overall_engagement_score = engagement_df["engagement_score"].mean()

    return engagement_df, overall_engagement_score

