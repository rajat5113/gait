import pandas as pd
import numpy as np
import cv2
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import json
from typing import List, Dict, Tuple

# Import local modules
from config import Config
from pose_processor import PoseProcessor

class GaitAnalyzer:
    """Main gait analysis class"""
    
    def __init__(self, config):
        self.config = config
        self.pose_processor = PoseProcessor(config)
        self.scaler = StandardScaler()
        
        # Gait data storage
        self.pose_sequences = []
        self.gait_events = []
        self.gait_metrics = {}
        
        # Real-world conversion factors (assuming person ~1.7m tall in frame)
        self.height_conversion = 170  # cm (assumes person is 1.7m tall)
        self.width_conversion = 60   # cm (assumes shoulder width ~60cm)
        
    def process_video(self, video_path: str, save_output: bool = True) -> Dict:
        """Process video file for gait analysis"""
        print(f"🎬 Processing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        pose_data_list = []
        processed_frames = []
        
        # Process frames
        with tqdm(total=frame_count, desc="Processing frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                processed_frame, pose_data = self.pose_processor.process_frame(frame)
                
                if pose_data:
                    pose_data_list.append(pose_data)
                    processed_frames.append(processed_frame)
                
                pbar.update(1)
        
        cap.release()
        
        if not pose_data_list:
            raise ValueError("No pose data extracted from video")
        
        # Convert to DataFrame
        pose_df = pd.DataFrame(pose_data_list)
        
        # Analyze gait
        gait_analysis = self._analyze_gait_sequence(pose_df, fps)
        
        # Add real-world units summary
        gait_analysis['summary_real_units'] = self._calculate_real_world_summary(gait_analysis)
        
        if save_output:
            self._save_analysis_results(video_path, pose_df, gait_analysis)
        
        # Print summary in real units
        self._print_real_units_summary(gait_analysis['summary_real_units'])
        
        return gait_analysis
    
    def process_realtime(self, camera_index: int = 0):
        """Process real-time camera input"""
        print("🎥 Starting real-time gait analysis...")
        print("🎮 Controls: 'q'=quit, 't'=toggle trail, 'r'=reset")
        
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise ValueError(f"Cannot open camera {camera_index}")
        
        # Real-time processing variables
        pose_buffer = []
        trail_points = []
        show_trail = True
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame from camera")
                break
            
            # Process frame
            processed_frame, pose_data = self.pose_processor.process_frame(frame)
            
            if pose_data:
                pose_buffer.append(pose_data)
                
                # Keep buffer size manageable
                if len(pose_buffer) > 300:  # 10 seconds at 30fps
                    pose_buffer.pop(0)
                
                # Add trail points
                if show_trail:
                    left_ankle = (pose_data.get('LEFT_ANKLE_x', 0), pose_data.get('LEFT_ANKLE_y', 0))
                    right_ankle = (pose_data.get('RIGHT_ANKLE_x', 0), pose_data.get('RIGHT_ANKLE_y', 0))
                    trail_points.append((left_ankle, right_ankle))
                    
                    if len(trail_points) > 50:  # Keep trail manageable
                        trail_points.pop(0)
            
            # Draw trail
            if show_trail and trail_points:
                h, w = processed_frame.shape[:2]
                for i, (left_point, right_point) in enumerate(trail_points):
                    alpha = i / len(trail_points)
                    color_intensity = int(255 * alpha)
                    
                    # Convert normalized coordinates to pixel coordinates
                    left_px = (int(left_point[0] * w), int(left_point[1] * h))
                    right_px = (int(right_point[0] * w), int(right_point[1] * h))
                    
                    # Draw trail points
                    cv2.circle(processed_frame, left_px, 3, (0, color_intensity, 255), -1)  # Blue for left
                    cv2.circle(processed_frame, right_px, 3, (0, 255, color_intensity), -1)  # Green for right
            
            # Real-time gait analysis
            if len(pose_buffer) >= 60:  # Analyze every 2 seconds
                try:
                    recent_data = pd.DataFrame(pose_buffer[-60:])
                    gait_metrics = self._calculate_realtime_metrics(recent_data)
                    self._draw_metrics_overlay(processed_frame, gait_metrics)
                except Exception as e:
                    print(f"⚠️ Error in real-time analysis: {e}")
            
            # Add status text
            cv2.putText(processed_frame, f"Buffer: {len(pose_buffer)} frames", 
                       (10, processed_frame.shape[0] - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Real-time Gait Analysis', processed_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("👋 Quitting real-time analysis...")
                break
            elif key == ord('t'):
                show_trail = not show_trail
                if not show_trail:
                    trail_points = []
                print(f"Trail: {'ON' if show_trail else 'OFF'}")
            elif key == ord('r'):
                trail_points = []
                print("🔄 Trail reset")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Return final analysis if enough data collected
        if len(pose_buffer) > 60:
            print("📊 Generating final analysis...")
            final_df = pd.DataFrame(pose_buffer)
            analysis = self._analyze_gait_sequence(final_df, 30)  # Assume 30fps
            analysis['summary_real_units'] = self._calculate_real_world_summary(analysis)
            self._print_real_units_summary(analysis['summary_real_units'])
            return analysis
        
        return None
    
    def _analyze_gait_sequence(self, pose_df: pd.DataFrame, fps: float) -> Dict:
        """Analyze gait sequence from pose data"""
        print("🔍 Analyzing gait sequence...")
        
        analysis = {
            'temporal_metrics': {},
            'spatial_metrics': {},
            'symmetry_metrics': {},
            'stability_metrics': {},
            'gait_events': [],
            'gait_phases': []
        }
        
        try:
            # Temporal analysis
            analysis['temporal_metrics'] = self._calculate_temporal_metrics(pose_df, fps)
            
            # Spatial analysis
            analysis['spatial_metrics'] = self._calculate_spatial_metrics(pose_df)
            
            # Gait event detection
            analysis['gait_events'] = self._detect_gait_events(pose_df, fps)
            
            # Symmetry analysis
            analysis['symmetry_metrics'] = self._calculate_symmetry_metrics(pose_df)
            
            # Stability analysis
            analysis['stability_metrics'] = self._calculate_stability_metrics(pose_df)
            
        except Exception as e:
            print(f"⚠️ Error in gait analysis: {e}")
        
        return analysis
    
    def _calculate_real_world_summary(self, analysis: Dict) -> Dict:
        """Calculate real-world units summary"""
        summary = {}
        
        try:
            # Duration in seconds
            summary['duration_seconds'] = round(analysis['temporal_metrics'].get('duration', 0), 2)
            
            # Cadence in steps/min
            summary['cadence_steps_per_min'] = round(analysis['temporal_metrics'].get('cadence', 0), 1)
            
            # Step width in cm
            step_width_normalized = analysis['spatial_metrics'].get('step_width_mean', 0)
            summary['step_width_cm'] = round(step_width_normalized * self.width_conversion, 1)
            
            # Average stride length in cm
            left_stride = analysis['spatial_metrics'].get('left_stride_length_mean', 0)
            right_stride = analysis['spatial_metrics'].get('right_stride_length_mean', 0)
            avg_stride = (left_stride + right_stride) / 2
            summary['stride_length_cm'] = round(avg_stride * self.height_conversion, 1)
            
            # Joint asymmetries in degrees
            summary['knee_angle_asymmetry_degrees'] = round(analysis['symmetry_metrics'].get('knee_angle_asymmetry', 0), 2)
            summary['hip_angle_asymmetry_degrees'] = round(analysis['symmetry_metrics'].get('hip_angle_asymmetry', 0), 2)
            
            # Calculate balance score (0-100)
            summary['balance_score'] = self._calculate_balance_score(analysis)
            
            # Additional useful metrics
            summary['step_time_variability_seconds'] = round(analysis['temporal_metrics'].get('step_time_variability', 0), 3)
            summary['gait_events_detected'] = len(analysis.get('gait_events', []))
            
        except Exception as e:
            print(f"⚠️ Error calculating real-world summary: {e}")
            summary = {
                'duration_seconds': 0,
                'cadence_steps_per_min': 0,
                'step_width_cm': 0,
                'stride_length_cm': 0,
                'knee_angle_asymmetry_degrees': 0,
                'hip_angle_asymmetry_degrees': 0,
                'balance_score': 0
            }
        
        return summary
    
    def _calculate_balance_score(self, analysis: Dict) -> int:
        """Calculate balance score from 0-100"""
        try:
            score = 100  # Start with perfect score
            
            # Deduct points for various issues
            stability = analysis.get('stability_metrics', {})
            temporal = analysis.get('temporal_metrics', {})
            symmetry = analysis.get('symmetry_metrics', {})
            
            # COM variability (side-to-side movement)
            com_x_var = stability.get('com_variability_x', 0)
            if com_x_var > 0.15:  # High side movement
                score -= min(30, int(com_x_var * 100))
            
            # Step width variability
            step_var = stability.get('step_width_variability', 0)
            if step_var > 0.03:  # Inconsistent stepping
                score -= min(20, int(step_var * 300))
            
            # Step time variability  
            time_var = temporal.get('step_time_variability', 0)
            if time_var > 0.05:  # Timing inconsistency
                score -= min(15, int(time_var * 200))
            
            # Joint asymmetries
            knee_asym = symmetry.get('knee_angle_asymmetry', 0)
            hip_asym = symmetry.get('hip_angle_asymmetry', 0)
            if knee_asym > 5:  # Significant knee asymmetry
                score -= min(10, int(knee_asym))
            if hip_asym > 3:  # Significant hip asymmetry
                score -= min(10, int(hip_asym))
            
            # Cadence issues (too fast or too slow)
            cadence = temporal.get('cadence', 110)
            if cadence < 90 or cadence > 140:  # Outside normal range
                score -= min(15, abs(cadence - 115) // 5)
            
            return max(0, min(100, score))  # Ensure score is 0-100
            
        except Exception as e:
            print(f"⚠️ Error calculating balance score: {e}")
            return 50  # Default middle score
    
    def _print_real_units_summary(self, summary: Dict):
        """Print summary in real-world units"""
        print("\n" + "="*50)
        print("📊 GAIT ANALYSIS SUMMARY (Real Units)")
        print("="*50)
        
        print(f"⏱️  Duration: {summary.get('duration_seconds', 0)} seconds")
        print(f"🚶 Cadence: {summary.get('cadence_steps_per_min', 0)} steps/min")
        print(f"📏 Step width: {summary.get('step_width_cm', 0)} cm")
        print(f"📐 Stride length: {summary.get('stride_length_cm', 0)} cm") 
        print(f"🦵 Knee angle asymmetry: {summary.get('knee_angle_asymmetry_degrees', 0)}°")
        print(f"🦴 Hip angle asymmetry: {summary.get('hip_angle_asymmetry_degrees', 0)}°")
        print(f"⚖️  Balance score: {summary.get('balance_score', 0)}/100")
        print(f"📊 Gait events detected: {summary.get('gait_events_detected', 0)}")
        
        print("="*50)
        
        # Also save as JSON
        json_output = {
            "gait_analysis_summary": {
                "duration_seconds": summary.get('duration_seconds', 0),
                "cadence_steps_per_min": summary.get('cadence_steps_per_min', 0),
                "step_width_cm": summary.get('step_width_cm', 0),
                "stride_length_cm": summary.get('stride_length_cm', 0),
                "knee_angle_asymmetry_degrees": summary.get('knee_angle_asymmetry_degrees', 0),
                "hip_angle_asymmetry_degrees": summary.get('hip_angle_asymmetry_degrees', 0),
                "balance_score_out_of_100": summary.get('balance_score', 0),
                "step_time_variability_seconds": summary.get('step_time_variability_seconds', 0),
                "gait_events_detected": summary.get('gait_events_detected', 0)
            }
        }
        
        print("\n📄 JSON FORMAT:")
        print(json.dumps(json_output, indent=2))
    
    def _calculate_temporal_metrics(self, pose_df: pd.DataFrame, fps: float) -> Dict:
        """Calculate temporal gait metrics"""
        metrics = {}
        
        # Duration
        metrics['duration'] = len(pose_df) / fps
        
        # Cadence (steps per minute) - approximate from ankle movement
        left_ankle_y = pose_df['LEFT_ANKLE_y'].values
        right_ankle_y = pose_df['RIGHT_ANKLE_y'].values
        
        # Find peaks (heel strikes approximation)
        left_peaks, _ = find_peaks(-left_ankle_y, distance=int(fps*0.3))  # Min 0.3s between steps
        right_peaks, _ = find_peaks(-right_ankle_y, distance=int(fps*0.3))
        
        total_steps = len(left_peaks) + len(right_peaks)
        metrics['cadence'] = (total_steps / (len(pose_df) / fps)) * 60 if len(pose_df) > 0 else 0
        
        # Step time variability
        if len(left_peaks) > 1 and len(right_peaks) > 1:
            left_step_times = np.diff(left_peaks) / fps
            right_step_times = np.diff(right_peaks) / fps
            all_step_times = np.concatenate([left_step_times, right_step_times])
            metrics['step_time_variability'] = np.std(all_step_times)
        else:
            metrics['step_time_variability'] = 0
        
        return metrics
    
    def _calculate_spatial_metrics(self, pose_df: pd.DataFrame) -> Dict:
        """Calculate spatial gait metrics"""
        metrics = {}
        
        # Step width (average distance between feet)
        metrics['step_width_mean'] = pose_df['step_width'].mean()
        metrics['step_width_std'] = pose_df['step_width'].std()
        
        # Stride length (approximate from hip-ankle distance changes)
        metrics['left_stride_length_mean'] = pose_df['left_stride_length'].mean()
        metrics['right_stride_length_mean'] = pose_df['right_stride_length'].mean()
        
        return metrics
    
    def _detect_gait_events(self, pose_df: pd.DataFrame, fps: float) -> List[Dict]:
        """Detect gait events (heel strikes, toe offs)"""
        events = []
        
        # Use ankle height for event detection
        left_ankle_y = pose_df['LEFT_ANKLE_y'].values
        right_ankle_y = pose_df['RIGHT_ANKLE_y'].values
        
        # Detect heel strikes (local minima in ankle height)
        left_heel_strikes, _ = find_peaks(-left_ankle_y, distance=int(fps*0.3))
        right_heel_strikes, _ = find_peaks(-right_ankle_y, distance=int(fps*0.3))
        
        # Format events
        for frame in left_heel_strikes:
            events.append({
                'frame': int(frame),
                'time': float(frame / fps),
                'event': 'left_heel_strike',
                'side': 'left'
            })
        
        for frame in right_heel_strikes:
            events.append({
                'frame': int(frame),
                'time': float(frame / fps),
                'event': 'right_heel_strike',
                'side': 'right'
            })
        
        # Sort events by time
        events.sort(key=lambda x: x['time'])
        
        return events
    
    def _calculate_symmetry_metrics(self, pose_df: pd.DataFrame) -> Dict:
        """Calculate gait symmetry metrics"""
        metrics = {}
        
        # Joint angle symmetry
        left_knee_angles = pose_df['left_knee_angle'].values
        right_knee_angles = pose_df['right_knee_angle'].values
        
        metrics['knee_angle_asymmetry'] = abs(np.mean(left_knee_angles) - np.mean(right_knee_angles))
        
        # Hip angle symmetry
        left_hip_angles = pose_df['left_hip_angle'].values
        right_hip_angles = pose_df['right_hip_angle'].values
        
        metrics['hip_angle_asymmetry'] = abs(np.mean(left_hip_angles) - np.mean(right_hip_angles))
        
        return metrics
    
    def _calculate_stability_metrics(self, pose_df: pd.DataFrame) -> Dict:
        """Calculate gait stability metrics"""
        metrics = {}
        
        # Center of mass approximation (midpoint between hips)
        com_x = (pose_df['LEFT_HIP_x'] + pose_df['RIGHT_HIP_x']) / 2
        com_y = (pose_df['LEFT_HIP_y'] + pose_df['RIGHT_HIP_y']) / 2
        
        # COM variability
        metrics['com_variability_x'] = np.std(com_x)
        metrics['com_variability_y'] = np.std(com_y)
        
        # Step width variability
        metrics['step_width_variability'] = np.std(pose_df['step_width'])
        
        return metrics
    
    def _calculate_realtime_metrics(self, pose_df: pd.DataFrame) -> Dict:
        """Calculate real-time gait metrics for overlay"""
        metrics = {}
        
        try:
            # Current cadence estimate
            left_ankle_y = pose_df['LEFT_ANKLE_y'].values
            peaks, _ = find_peaks(-left_ankle_y, distance=10)
            metrics['cadence'] = len(peaks) * 30  # Approximate for 2-second window
            
            # Current step width in cm
            step_width_normalized = pose_df['step_width'].iloc[-1]
            metrics['step_width_cm'] = round(step_width_normalized * self.width_conversion, 1)
            
            # Balance indicator (trunk stability)
            shoulder_x = (pose_df['LEFT_SHOULDER_x'] + pose_df['RIGHT_SHOULDER_x']) / 2
            stability_score = 1.0 - np.std(shoulder_x.iloc[-30:])  # Last 1 second
            metrics['balance'] = round(max(0, min(1, stability_score)) * 100)  # Convert to 0-100
            
        except Exception as e:
            print(f"Error calculating real-time metrics: {e}")
            metrics = {'cadence': 0, 'step_width_cm': 0, 'balance': 0}
        
        return metrics
    
    def _draw_metrics_overlay(self, frame: np.ndarray, metrics: Dict):
        """Draw real-time metrics overlay"""
        y_offset = 30
        overlay_metrics = {
            'Cadence': f"{metrics.get('cadence', 0)} steps/min",
            'Step Width': f"{metrics.get('step_width_cm', 0)} cm",
            'Balance': f"{metrics.get('balance', 0)}/100"
        }
        
        for key, value in overlay_metrics.items():
            try:
                text = f"{key}: {value}"
                cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_offset += 25
            except:
                pass
    
    def _save_analysis_results(self, video_path: str, pose_df: pd.DataFrame, analysis: Dict):
        """Save analysis results to files"""
        output_dir = self.config.get("output_dir", "outputs")
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        session_dir = os.path.join(output_dir, video_name)
        os.makedirs(session_dir, exist_ok=True)
        
        # Save pose data
        if self.config.get("save_pose_data", True):
            pose_df.to_csv(os.path.join(session_dir, "pose_data.csv"), index=False)
        
        # Save analysis results (including real units summary)
        if self.config.get("save_metrics", True):
            with open(os.path.join(session_dir, "gait_analysis.json"), 'w') as f:
                # Convert numpy types to native Python types for JSON serialization
                def convert_numpy(obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return obj
                
                json_analysis = {}
                for key, value in analysis.items():
                    if isinstance(value, dict):
                        json_analysis[key] = {k: convert_numpy(v) for k, v in value.items()}
                    elif isinstance(value, list):
                        json_analysis[key] = [convert_numpy(item) if isinstance(item, (dict, np.integer, np.floating)) else item for item in value]
                    else:
                        json_analysis[key] = convert_numpy(value)
                
                json.dump(json_analysis, f, indent=4)
        
        print(f"💾 Results saved to: {session_dir}")