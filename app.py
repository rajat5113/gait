# app.py - Simple Web-based Gait Analysis Platform

from flask import Flask, request, render_template, jsonify, send_from_directory
import os
import json
import tempfile
import shutil
from werkzeug.utils import secure_filename
import traceback
from datetime import datetime

# Import your gait analysis modules
from config import Config
from gait_analyzer import GaitAnalyzer

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Configuration
UPLOAD_FOLDER = 'web_uploads'
RESULTS_FOLDER = 'web_results'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'm4v'}

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_video_analysis(video_path):
    """Process video and return analysis results"""
    try:
        # Initialize gait analyzer
        config = Config()
        analyzer = GaitAnalyzer(config)
        
        # Process video
        analysis = analyzer.process_video(video_path, save_output=False)
        
        # Get real-world summary
        summary = analyzer._calculate_real_world_summary(analysis)
        
        return {
            'success': True,
            'results': summary,
            'full_analysis': analysis,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'timestamp': datetime.now().isoformat()
        }

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and analysis"""
    if 'video' not in request.files:
        return jsonify({'success': False, 'error': 'No video file provided'})
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
    
    if file and allowed_file(file.filename):
        try:
            # Save uploaded file
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_filename = f"{timestamp}_{filename}"
            video_path = os.path.join(UPLOAD_FOLDER, unique_filename)
            file.save(video_path)
            
            # Process the video
            result = process_video_analysis(video_path)
            
            # Save results
            result_filename = f"{timestamp}_results.json"
            result_path = os.path.join(RESULTS_FOLDER, result_filename)
            with open(result_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            # Clean up uploaded file (optional)
            # os.remove(video_path)
            
            return jsonify(result)
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Processing failed: {str(e)}',
                'traceback': traceback.format_exc()
            })
    else:
        return jsonify({'success': False, 'error': 'Invalid file type. Please upload MP4, AVI, MOV, MKV, WMV, or M4V files.'})

@app.route('/results/<filename>')
def download_result(filename):
    """Download result file"""
    return send_from_directory(RESULTS_FOLDER, filename)

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    print("🚀 Starting Gait Analysis Web Platform...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("💡 Upload a video and get instant gait analysis results!")
    app.run(debug=True, host='0.0.0.0', port=5000)