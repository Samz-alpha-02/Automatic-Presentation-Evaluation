import json
import cv2
from fer import FER

def process_video(video_path):
    detector = FER(mtcnn=True)

    cap = cv2.VideoCapture(video_path)

    # Initialize dictionary to store emotion counts with the correct labels used by FER
    emotion_counts = {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 0}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect emotions in the frame
        result = detector.detect_emotions(frame)

        if result:
            # Get the detected emotion with the highest confidence
            dominant_emotion, score = detector.top_emotion(frame)
            if score > 0.4:  # consider only emotions with a confidence level above 40%
                emotion_counts[dominant_emotion] += 1

    cap.release()
    cv2.destroyAllWindows()

    # Find the dominant emotion
    dominant_emotion = max(emotion_counts, key=emotion_counts.get)

    # Create a dictionary containing the final emotion data
    final_emotion_data = {'dominant_emotion': dominant_emotion}

    # Wrap the final emotion data inside a dictionary with the key 'video_result'
    result = {'video_result': final_emotion_data}

    # Convert the dictionary to JSON format
    json_data_recorded = json.dumps(result)

    print(json_data_recorded)

    return json_data_recorded

