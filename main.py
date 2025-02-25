import os
import cv2
import numpy as np
from keras.src.applications.vgg16 import VGG16, preprocess_input
from concurrent.futures import ThreadPoolExecutor

# Load pre-trained VGG16 model for feature extraction
model = VGG16(weights='imagenet', include_top=False, pooling='avg')

VIDEO_FOLDER = 'videos'
FRAME_INTERVAL = 2
SIMILARITY_THRESHOLD = 0.95
NUM_THREADS = 12

def extract_frames(video_path, interval=FRAME_INTERVAL):
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    success, frame = cap.read()
    count = 0
    while success:
        if int(count % (interval * fps)) == 0:
            frames.append(frame)
        success, frame = cap.read()
        count += 1
    cap.release()
    return frames

def extract_features(frames):
    features = []
    for frame in frames:
        img = cv2.resize(frame, (224, 224))
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        feature = model.predict(img)
        features.append(feature.flatten())
    return features

def calculate_similarity(features1, features2):
    # Use cosine similarity as the similarity metric
    mean_features1 = np.mean(features1, axis=0)
    mean_features2 = np.mean(features2, axis=0)
    
    norm1 = mean_features1 / np.linalg.norm(mean_features1)
    norm2 = mean_features2 / np.linalg.norm(mean_features2)
    
    return np.dot(norm1, norm2)

def process_video(video):
    video_path = os.path.join(VIDEO_FOLDER, video)
    frames = extract_frames(video_path)
    features = extract_features(frames)
    return video, features

def detect_duplicate_videos():
    video_files = [f for f in os.listdir(VIDEO_FOLDER) if f.endswith('.mp4')]
    video_features = {}

    # Step 1: Extract features from each video using threads
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        for video, features in executor.map(process_video, video_files):
            video_features[video] = features

    # Step 2: Compare each pair of videos for similarity
    duplicate_videos = []
    video_names = list(video_features.keys())
    for i in range(len(video_names)):
        for j in range(i + 1, len(video_names)):
            video1 = video_names[i]
            video2 = video_names[j]
            similarity_score = calculate_similarity(video_features[video1], video_features[video2])
            if similarity_score > SIMILARITY_THRESHOLD:
                print(f"Video '{video1}' is a duplicate of Video '{video2}' with similarity: {similarity_score:.2f}")
                duplicate_videos.append((video1, video2, similarity_score))

    # Step 3: Print the result
    if duplicate_videos:
        print("\nList of duplicate videos:")
        for video1, video2, score in duplicate_videos:
            print(f"Video: {video1} is a duplicate of Video: {video2} (Similarity: {score * 100:.2f}%)")
    else:
        print("No duplicate videos found.")

detect_duplicate_videos()