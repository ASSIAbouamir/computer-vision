#!/usr/bin/env python
"""
Generate videos from frame sequences
Creates MP4 videos from tracking frames and graph visualizations
"""

import cv2
import os
import glob
import numpy as np

def create_video_from_images(image_pattern, output_path, fps=25):
    """
    Create video from sequence of images
    
    Args:
        image_pattern: Glob pattern for images (e.g., 'frame_*.png')
        output_path: Output video file path
        fps: Frames per second
    """
    print(f"📹 Creating video: {output_path}")
    
    # Get sorted list of images
    images = sorted(glob.glob(image_pattern))
    
    if not images:
        print(f"  ❌ No images found for pattern: {image_pattern}")
        return False
    
    print(f"  Found {len(images)} frames")
    
    # Read first image to get dimensions
    first_frame = cv2.imread(images[0])
    if first_frame is None:
        print(f"  ❌ Could not read first image: {images[0]}")
        return False
    
    height, width, layers = first_frame.shape
    print(f"  Resolution: {width}x{height}")
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write all frames
    for i, img_path in enumerate(images):
        frame = cv2.imread(img_path)
        if frame is not None:
            video.write(frame)
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{len(images)} frames")
        else:
            print(f"  ⚠️ Could not read: {img_path}")
    
    video.release()
    
    # Get output file size
    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    print(f"  ✅ Video created: {file_size:.2f} MB")
    print(f"  Duration: {len(images)/fps:.2f} seconds @ {fps} fps")
    
    return True


def create_side_by_side_video(pattern1, pattern2, output_path, fps=25):
    """
    Create side-by-side comparison video
    
    Args:
        pattern1: Pattern for left side images
        pattern2: Pattern for right side images
        output_path: Output video path
        fps: Frames per second
    """
    print(f"📹 Creating side-by-side video: {output_path}")
    
    images1 = sorted(glob.glob(pattern1))
    images2 = sorted(glob.glob(pattern2))
    
    if not images1 or not images2:
        print(f"  ❌ Missing images")
        return False
    
    if len(images1) != len(images2):
        print(f"  ⚠️ Mismatch: {len(images1)} vs {len(images2)}, using minimum")
    
    num_frames = min(len(images1), len(images2))
    print(f"  Processing {num_frames} frames")
    
    # Read first frames
    frame1 = cv2.imread(images1[0])
    frame2 = cv2.imread(images2[0])
    
    h1, w1 = frame1.shape[:2]
    h2, w2 = frame2.shape[:2]
    
    # Match heights
    target_height = min(h1, h2)
    w1_scaled = int(w1 * target_height / h1)
    w2_scaled = int(w2 * target_height / h2)
    
    width = w1_scaled + w2_scaled
    height = target_height
    
    print(f"  Combined resolution: {width}x{height}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for i in range(num_frames):
        img1 = cv2.imread(images1[i])
        img2 = cv2.imread(images2[i])
        
        # Resize to match height
        img1_resized = cv2.resize(img1, (w1_scaled, target_height))
        img2_resized = cv2.resize(img2, (w2_scaled, target_height))
        
        # Concatenate horizontally
        combined = np.hstack([img1_resized, img2_resized])
        
        video.write(combined)
        
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{num_frames} frames")
    
    video.release()
    
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  ✅ Video created: {file_size:.2f} MB")
    
    return True


def main():
    """Main function to generate all videos"""
    
    print("🎬 VIDEO GENERATION STARTING")
    print("=" * 60)
    
    output_dir = "output_results"
    
    # Check if output directory exists
    if not os.path.exists(output_dir):
        print(f"❌ Output directory not found: {output_dir}")
        return
    
    videos_created = 0
    
    # 1. Tracking Video
    print("\n1️⃣ TRACKING VIDEO")
    pattern_tracking = os.path.join(output_dir, "frame_*.png")
    output_tracking = os.path.join(output_dir, "tracking_video.mp4")
    if create_video_from_images(pattern_tracking, output_tracking, fps=25):
        videos_created += 1
    
    # 2. Graph Interaction Video
    print("\n2️⃣ GRAPH INTERACTION VIDEO")
    pattern_graph = os.path.join(output_dir, "graph_*.png")
    output_graph = os.path.join(output_dir, "graph_interaction_video.mp4")
    if create_video_from_images(pattern_graph, output_graph, fps=25):
        videos_created += 1
    
    # 3. Side-by-Side Comparison Video (bonus)
    print("\n3️⃣ SIDE-BY-SIDE COMPARISON VIDEO")
    output_comparison = os.path.join(output_dir, "tracking_vs_graph_video.mp4")
    if create_side_by_side_video(pattern_tracking, pattern_graph, output_comparison, fps=25):
        videos_created += 1
    
    print("\n" + "=" * 60)
    print(f"🎉 VIDEO GENERATION COMPLETE!")
    print(f"   Created {videos_created} videos in {output_dir}/")
    print("\n📁 Generated files:")
    
    video_files = [
        "tracking_video.mp4",
        "graph_interaction_video.mp4", 
        "tracking_vs_graph_video.mp4"
    ]
    
    for video in video_files:
        path = os.path.join(output_dir, video)
        if os.path.exists(path):
            size = os.path.getsize(path) / (1024 * 1024)
            print(f"   ✅ {video} ({size:.2f} MB)")
        else:
            print(f"   ❌ {video} (not created)")


if __name__ == "__main__":
    main()
