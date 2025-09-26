import numpy as np
import cv2
import os
import argparse
import sys
from config import Config
from gait_analyzer import GaitAnalyzer

class DemoRunner:
    """Demo runner for testing the gait analysis system"""
    
    def __init__(self):
        try:
            self.config = Config()
            self.analyzer = GaitAnalyzer(self.config)
        except Exception as e:
            print(f"❌ Error initializing demo: {e}")
            sys.exit(1)
    
    def create_sample_video(self, output_path: str = "videos/sample_video.mp4", duration: int = 10):
        """Create a sample video with simulated walking motion"""
        print("🎬 Creating sample gait video...")
        
        # Ensure videos directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Video parameters
        fps = 30
        width, height = 640, 480
        total_frames = duration * fps
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print("❌ Failed to create video writer")
            return None
        
        # Simulate walking motion
        for frame_num in range(total_frames):
            # Create blank frame
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Simulate person walking (simplified stick figure)
            t = frame_num / fps  # time in seconds
            walk_cycle = 2.0  # 2 seconds per walk cycle
            phase = (t % walk_cycle) / walk_cycle * 2 * np.pi
            
            # Body center (moving across screen)
            center_x = int(100 + (t / duration) * (width - 200))
            center_y = int(height * 0.6)
            
            # Draw simplified person
            self._draw_walking_person(frame, center_x, center_y, phase)
            
            out.write(frame)
        
        out.release()
        print(f"✅ Sample video created: {output_path}")
        return output_path
    
    def _draw_walking_person(self, frame: np.ndarray, center_x: int, center_y: int, phase: float):
        """Draw a simplified walking person on the frame"""
        # Head
        cv2.circle(frame, (center_x, center_y - 60), 15, (255, 255, 255), -1)
        
        # Body
        cv2.line(frame, (center_x, center_y - 45), (center_x, center_y + 30), (255, 255, 255), 3)
        
        # Arms (swinging)
        arm_swing = int(20 * np.sin(phase))
        left_arm = (center_x - 15 - arm_swing, center_y - 10)
        right_arm = (center_x + 15 + arm_swing, center_y - 10)
        cv2.line(frame, (center_x, center_y - 20), left_arm, (255, 255, 255), 2)
        cv2.line(frame, (center_x, center_y - 20), right_arm, (255, 255, 255), 2)
        
        # Legs (walking motion)
        leg_forward = int(25 * np.sin(phase))
        leg_up = int(10 * abs(np.sin(phase)))
        
        # Left leg
        left_knee = (center_x - 10 + leg_forward, center_y + 15)
        left_foot = (center_x - 10 + leg_forward * 2, center_y + 45 - leg_up)
        cv2.line(frame, (center_x, center_y + 30), left_knee, (255, 255, 255), 2)
        cv2.line(frame, left_knee, left_foot, (255, 255, 255), 2)
        
        # Right leg
        right_knee = (center_x + 10 - leg_forward, center_y + 15)
        right_foot = (center_x + 10 - leg_forward * 2, center_y + 45 - leg_up)
        cv2.line(frame, (center_x, center_y + 30), right_knee, (255, 255, 255), 2)
        cv2.line(frame, right_knee, right_foot, (255, 255, 255), 2)
    
    def validate_installation(self):
        """Validate that all required packages are installed"""
        print("🔍 Validating installation...")
        success = True
        
        try:
            import cv2
            print("✅ OpenCV installed:", cv2.__version__)
        except ImportError:
            print("❌ OpenCV not found - run: pip install opencv-python")
            success = False
            
        try:
            import mediapipe
            print("✅ MediaPipe installed")
        except ImportError:
            print("❌ MediaPipe not found - run: pip install mediapipe")
            success = False
            
        try:
            import pandas
            print("✅ Pandas installed:", pandas.__version__)
        except ImportError:
            print("❌ Pandas not found - run: pip install pandas")
            success = False
            
        try:
            import numpy
            print("✅ NumPy installed:", numpy.__version__)
        except ImportError:
            print("❌ NumPy not found - run: pip install numpy")
            success = False
            
        try:
            import matplotlib
            print("✅ Matplotlib installed:", matplotlib.__version__)
        except ImportError:
            print("❌ Matplotlib not found - run: pip install matplotlib")
            success = False
            
        try:
            import scipy
            print("✅ SciPy installed:", scipy.__version__)
        except ImportError:
            print("❌ SciPy not found - run: pip install scipy")
            success = False
        
        # Test camera access
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                print("✅ Camera access available")
                cap.release()
            else:
                print("⚠️  Camera not accessible (but system can still process videos)")
        except Exception as e:
            print(f"⚠️  Camera test failed: {e}")
        
        if success:
            print("✅ Installation validation complete!")
        else:
            print("❌ Installation validation failed. Please install missing packages.")
            
        return success
    
    def test_pose_detection(self):
        """Test pose detection on sample video"""
        print("🧪 Testing pose detection...")
        
        # Create sample video if it doesn't exist
        sample_video = "videos/sample_video.mp4"
        if not os.path.exists(sample_video):
            result = self.create_sample_video(sample_video, duration=5)
            if not result:
                print("❌ Failed to create sample video")
                return False
        
        try:
            # Test pose processing
            results = self.analyzer.process_video(sample_video)
            print("✅ Pose detection test passed!")
            print(f"   - Processed {results['temporal_metrics'].get('duration', 0):.1f} seconds of video")
            print(f"   - Detected {len(results['gait_events'])} gait events")
            return True
        except Exception as e:
            print(f"❌ Pose detection test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_realtime(self, duration: int = 10):
        """Test real-time analysis for specified duration"""
        print(f"🎥 Testing real-time analysis for {duration} seconds...")
        print("   Press 'q' to quit early")
        
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("❌ Cannot access camera")
                return False
                
            frame_count = 0
            max_frames = duration * 30  # Assume 30fps
            
            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                processed_frame, pose_data = self.analyzer.pose_processor.process_frame(frame)
                
                # Add test overlay
                cv2.putText(processed_frame, f"Frame: {frame_count}/{max_frames}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(processed_frame, f"Pose: {'Detected' if pose_data else 'Not Detected'}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                           (0, 255, 0) if pose_data else (0, 0, 255), 2)
                
                cv2.imshow('Real-time Test', processed_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
                frame_count += 1
            
            cap.release()
            cv2.destroyAllWindows()
            print("✅ Real-time test completed!")
            return True
            
        except Exception as e:
            print(f"❌ Real-time test failed: {e}")
            return False
    
    def run_complete_demo(self):
        """Run complete demonstration"""
        print("🚀 Starting Complete Gait Analysis Demo")
        print("=" * 50)
        
        # Step 1: Validate installation
        if not self.validate_installation():
            print("❌ Installation validation failed. Please install missing packages.")
            return
        
        # Step 2: Create sample video
        print("\n📹 Creating sample video...")
        sample_video = self.create_sample_video(duration=10)
        if not sample_video:
            print("❌ Failed to create sample video")
            return
        
        # Step 3: Test video analysis
        print("\n🔍 Testing video-based analysis...")
        try:
            results = self.analyzer.process_video(sample_video)
            self._print_analysis_summary(results)
        except Exception as e:
            print(f"❌ Video analysis failed: {e}")
            return
        
        # Step 4: Test real-time (optional)
        print("\n🎥 Real-time analysis test...")
        response = input("Do you want to test real-time camera analysis? (y/n): ").lower()
        if response == 'y':
            self.test_realtime(duration=10)
        else:
            print("⏭️  Skipping real-time test")
        
        print("\n🎉 Demo complete! Check the 'outputs' folder for results.")
        print("📁 Results saved in: outputs/sample_video/")
    
    def _print_analysis_summary(self, results: dict):
        """Print analysis results summary"""
        print("\n📊 Analysis Results Summary")
        print("-" * 40)
        
        if 'temporal_metrics' in results:
            print("⏱️  Temporal Metrics:")
            for key, value in results['temporal_metrics'].items():
                if isinstance(value, (int, float)):
                    print(f"   {key}: {value:.3f}")
                else:
                    print(f"   {key}: {value}")
        
        if 'spatial_metrics' in results:
            print("\n📏 Spatial Metrics:")
            for key, value in results['spatial_metrics'].items():
                if isinstance(value, (int, float)):
                    print(f"   {key}: {value:.3f}")
                else:
                    print(f"   {key}: {value}")
        
        if 'symmetry_metrics' in results:
            print("\n⚖️  Symmetry Metrics:")
            for key, value in results['symmetry_metrics'].items():
                if isinstance(value, (int, float)):
                    print(f"   {key}: {value:.3f}")
                else:
                    print(f"   {key}: {value}")
        
        if 'gait_events' in results:
            print(f"\n🎯 Detected {len(results['gait_events'])} gait events")
            if results['gait_events']:
                event_types = {}
                for event in results['gait_events']:
                    event_type = event['event']
                    event_types[event_type] = event_types.get(event_type, 0) + 1
                for event_type, count in event_types.items():
                    print(f"   - {event_type}: {count}")
        
        print("-" * 40)

def main():
    parser = argparse.ArgumentParser(description='🧪 Gait Analysis Demo')
    parser.add_argument('--validate', action='store_true', help='Validate installation')
    parser.add_argument('--create-sample', action='store_true', help='Create sample video')
    parser.add_argument('--test-pose', action='store_true', help='Test pose detection')
    parser.add_argument('--test-realtime', type=int, metavar='SECONDS', help='Test real-time for N seconds')
    parser.add_argument('--run-demo', action='store_true', help='Run complete demo')
    parser.add_argument('--test-all', action='store_true', help='Run all tests')
    
    args = parser.parse_args()
    
    try:
        demo = DemoRunner()
    except Exception as e:
        print(f"❌ Failed to initialize demo: {e}")
        return
    
    if args.validate:
        demo.validate_installation()
    elif args.create_sample:
        demo.create_sample_video()
    elif args.test_pose:
        demo.test_pose_detection()
    elif args.test_realtime:
        demo.test_realtime(args.test_realtime)
    elif args.run_demo:
        demo.run_complete_demo()
    elif args.test_all:
        print("🧪 Running all tests...")
        success = True
        success &= demo.validate_installation()
        if success:
            demo.create_sample_video()
            success &= demo.test_pose_detection()
        if success:
            print("✅ All tests passed!")
        else:
            print("❌ Some tests failed")
    else:
        print("🧪 Gait Analysis Demo")
        print("Available commands:")
        print("  --validate          Validate installation")
        print("  --create-sample     Create sample video")
        print("  --test-pose         Test pose detection")
        print("  --test-realtime 10  Test real-time for 10 seconds")
        print("  --run-demo          Run complete demo")
        print("  --test-all          Run all tests")
        print("\nQuick start:")
        print("  python demo.py --run-demo")

if __name__ == "__main__":
    main()