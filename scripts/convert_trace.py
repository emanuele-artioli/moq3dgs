#!/usr/bin/env python3
"""Convert EyeNavGS (Rutgers) CSV traces to 3DGS-MoQ JSON format.

Usage::

    python scripts/convert_trace.py assets/bicycle_traces/user101_bicycle.csv assets/bicycle_trace.json
"""

import argparse
import csv
import json
import numpy as np
from scipy.spatial.transform import Rotation as R

def quaternion_to_view_matrix(pos, quat_xyzw):
    """Convert position and quaternion to a 4x4 view matrix.
    
    EyeNavGS uses (X, Y, Z, W) quaternions and world coordinates.
    View matrix is World -> Camera.
    """
    rot = R.from_quat(quat_xyzw)
    # Rotation matrix (Camera -> World)
    rmat = rot.as_matrix()
    
    # In OpenGL/3DGS convention, camera looks down -Z.
    # If the quaternion represents the camera orientation in world space,
    # the view matrix is the inverse.
    view = np.eye(4)
    view[:3, :3] = rmat.T
    view[:3, 3] = -rmat.T @ np.array(pos)
    
    return view.tolist()

def convert_csv_to_json(csv_path, json_path):
    frames = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # EyeNavGS has multiple ViewIndices per timestamp (stereo/left-right), 
            # we just take one (e.g., ViewIndex 0)
            if row['ViewIndex'] != '0':
                continue
                
            pos = [float(row['PositionX']), float(row['PositionY']), float(row['PositionZ'])]
            quat = [float(row['QuaternionX']), float(row['QuaternionY']), 
                    float(row['QuaternionZ']), float(row['QuaternionW'])]
            
            # Rutgers dataset FOV values are half-angles in radians usually? 
            # FOV1..4 are likely left, right, top, bottom.
            # For simplicity, we'll use a standard FOV or calculate it if possible.
            # Based on Rutgers docs, FOV values are related to the projection.
            # We'll use a default 60.0 or estimate from FOV2/FOV4 (y-axis).
            fov_y = np.degrees(float(row['FOV2']) + float(row['FOV4']))
            if fov_y <= 0 or fov_y > 170:
                fov_y = 60.0
                
            frames.append({
                "timestamp_ms": int(row['Timestamp']),
                "camera_position": pos,
                "view_matrix": quaternion_to_view_matrix(pos, quat),
                "fov": fov_y
            })
            
    with open(json_path, 'w') as f:
        json.dump({"frames": frames}, f, indent=2)
    print(f"Converted {len(frames)} frames to {json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv")
    parser.add_argument("output_json")
    args = parser.parse_args()
    convert_csv_to_json(args.input_csv, args.output_json)
