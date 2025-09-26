import argparse
import os
import sys
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our modules
try:
    from config import Config
    from gait_analyzer import GaitAnalyzer
    print("✅ All modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all required files are in the current directory:")
    print("- config.py")
    print("- pose_processor.py") 
    print("- gait_analyzer.py")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='🚶 Gait Analysis System')
    parser.add_argument('--mode', choices=['video', 'realtime'], 
                       default='video', help='Analysis mode')
    parser.add_argument('--input', type=str, help='Input video file')
    parser.add_argument('--output', type=str, default='outputs', help='Output directory')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--camera', type=int, default=0, help='Camera index for real-time mode')
    
    args = parser.parse_args()
    
    print("🚀 Starting Gait Analysis System...")
    
    # Load configuration
    try:
        config = Config(args.config)
        if args.output:
            config.config['output_dir'] = args.output
        print("✅ Configuration loaded successfully")
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        print("Creating default configuration...")
        try:
            # Create default config
            os.makedirs("configs", exist_ok=True)
            default_config = {
                "pose_model": "mediapipe",
                "model_complexity": 1,
                "min_detection_confidence": 0.7,
                "min_tracking_confidence": 0.5,
                "output_dir": "outputs",
                "save_pose_data": True,
                "save_visualizations": True,
                "save_metrics": True
            }
            import json
            with open("configs/default_config.json", "w") as f:
                json.dump(default_config, f, indent=4)
            config = Config()
            print("✅ Default configuration created and loaded")
        except Exception as e2:
            print(f"❌ Failed to create default configuration: {e2}")
            return
    
    # Initialize analyzer
    try:
        analyzer = GaitAnalyzer(config)
        print("✅ Gait analyzer initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing analyzer: {e}")
        import traceback
        traceback.print_exc()
        return
    
    try:
        if args.mode == 'video':
            if not args.input:
                print("❌ Error: --input required for video mode")
                print("Usage: python main.py --mode video --input your_video.mp4")
                return
            
            if not os.path.exists(args.input):
                print(f"❌ Error: Video file not found: {args.input}")
                return
            
            print("🎬 Starting video-based gait analysis...")
            results = analyzer.process_video(args.input)
            print("\n✅ Analysis complete!")
            print(f"📊 Results summary:")
            print(f"   - Duration: {results['temporal_metrics'].get('duration', 0):.2f} seconds")
            print(f"   - Cadence: {results['temporal_metrics'].get('cadence', 0):.2f} steps/min")
            print(f"   - Detected events: {len(results['gait_events'])}")
            print(f"   - Results saved to: {config.get('output_dir')}")
            
        elif args.mode == 'realtime':
            print("🎥 Starting real-time gait analysis...")
            print("📹 Make sure your camera is connected")
            print("🎮 Controls:")
            print("   - 'q': Quit")
            print("   - 't': Toggle trail effect")
            print("   - 'r': Reset trail")
            
            results = analyzer.process_realtime(args.camera)
            if results:
                print("\n✅ Real-time analysis complete!")
                print(f"📊 Final analysis:")
                print(f"   - Duration: {results['temporal_metrics'].get('duration', 0):.2f} seconds")
                print(f"   - Average cadence: {results['temporal_metrics'].get('cadence', 0):.2f} steps/min")
                print(f"   - Total events detected: {len(results['gait_events'])}")
            else:
                print("⚠️  No analysis data collected (session too short)")
            
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()