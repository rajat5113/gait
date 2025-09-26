import json
import os
from pathlib import Path
from typing import Dict, Any

class Config:
    """Configuration management for gait analysis system"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or "configs/default_config.json"
        self.config = self._load_default_config()
        if os.path.exists(self.config_path):
            self.config.update(self._load_config())
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration"""
        return {
            # Pose estimation settings
            "pose_model": "mediapipe",
            "model_complexity": 1,
            "min_detection_confidence": 0.7,
            "min_tracking_confidence": 0.5,
            
            # Gait analysis settings
            "sequence_length": 60,
            "overlap": 0.5,
            "sampling_rate": 30,
            
            # TCN model settings
            "num_classes": 4,  # stance_left, stance_right, swing_left, swing_right
            "num_filters": 64,
            "kernel_size": 3,
            "num_blocks": 4,
            "dropout_rate": 0.2,
            
            # Training settings
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100,
            "validation_split": 0.2,
            
            # Output settings
            "output_dir": "outputs",
            "save_pose_data": True,
            "save_visualizations": True,
            "save_metrics": True
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        with open(self.config_path, 'r') as f:
            return json.load(f)
    
    def get(self, key: str, default=None):
        """Get configuration value"""
        return self.config.get(key, default)
    
    def save_config(self):
        """Save current configuration to file"""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=4)