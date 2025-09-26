import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Optional, Dict
import time

class PoseProcessor:
    """MediaPipe pose processor for gait analysis"""
    
    def __init__(self, config):
        self.config = config
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=config.get("model_complexity", 1),
            enable_segmentation=False,
            min_detection_confidence=config.get("min_detection_confidence", 0.7),
            min_tracking_confidence=config.get("min_tracking_confidence", 0.5)
        )
        
        # Define key landmarks for gait analysis
        self.key_landmarks = {
            'LEFT_HIP': 23,
            'RIGHT_HIP': 24,
            'LEFT_KNEE': 25,
            'RIGHT_KNEE': 26,
            'LEFT_ANKLE': 27,
            'RIGHT_ANKLE': 28,
            'LEFT_FOOT_INDEX': 31,
            'RIGHT_FOOT_INDEX': 32,
            'LEFT_SHOULDER': 11,
            'RIGHT_SHOULDER': 12
        }
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[Dict]]:
        """Process single frame and extract pose landmarks"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        pose_data = None
        if results.pose_landmarks:
            pose_data = self._extract_pose_features(results.pose_landmarks)
            # Draw pose on frame
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
        return frame, pose_data
    
    def _extract_pose_features(self, landmarks) -> Dict:
        """Extract relevant pose features for gait analysis"""
        features = {}
        
        # Extract key landmark coordinates
        for name, idx in self.key_landmarks.items():
            landmark = landmarks.landmark[idx]
            features[f"{name}_x"] = landmark.x
            features[f"{name}_y"] = landmark.y
            features[f"{name}_z"] = landmark.z
            features[f"{name}_visibility"] = landmark.visibility
        
        # Calculate additional features
        features.update(self._calculate_joint_angles(landmarks))
        features.update(self._calculate_distances(landmarks))
        
        return features
    
    def _calculate_joint_angles(self, landmarks) -> Dict:
        """Calculate joint angles for gait analysis"""
        angles = {}
        
        def get_angle(p1_idx, p2_idx, p3_idx):
            """Calculate angle between three points"""
            try:
                p1 = landmarks.landmark[p1_idx]
                p2 = landmarks.landmark[p2_idx]
                p3 = landmarks.landmark[p3_idx]
                
                v1 = np.array([p1.x - p2.x, p1.y - p2.y])
                v2 = np.array([p3.x - p2.x, p3.y - p2.y])
                
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)
                return np.degrees(angle)
            except:
                return 0.0
        
        # Hip angles
        angles['left_hip_angle'] = get_angle(11, 23, 25)  # shoulder-hip-knee
        angles['right_hip_angle'] = get_angle(12, 24, 26)
        
        # Knee angles
        angles['left_knee_angle'] = get_angle(23, 25, 27)  # hip-knee-ankle
        angles['right_knee_angle'] = get_angle(24, 26, 28)
        
        # Ankle angles (approximate)
        angles['left_ankle_angle'] = get_angle(25, 27, 31)  # knee-ankle-foot
        angles['right_ankle_angle'] = get_angle(26, 28, 32)
        
        return angles
    
    def _calculate_distances(self, landmarks) -> Dict:
        """Calculate relevant distances"""
        distances = {}
        
        def get_distance(p1_idx, p2_idx):
            try:
                p1 = landmarks.landmark[p1_idx]
                p2 = landmarks.landmark[p2_idx]
                return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
            except:
                return 0.0
        
        # Step width (distance between feet)
        distances['step_width'] = get_distance(27, 28)  # ankle to ankle
        
        # Stride parameters
        distances['left_stride_length'] = get_distance(23, 27)  # hip to ankle
        distances['right_stride_length'] = get_distance(24, 28)
        
        return distances