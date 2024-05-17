import json
import cv2
import numpy as np
import tensorflow as tf

def process_video(video_path):
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path='Model/model.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(video_path)

    # Initialize variables to store emotion percentages
    emotion_percentages = {'neutral': 0, 'happiness': 0, 'surprise': 0, 'sadness': 0, 'anger': 0, 'disgust': 0, 'fear': 0}

    # Initialize dictionary to store emotion counts
    emotion_counts = {emotion: 0 for emotion in emotion_percentages}

    while True:
        ret, img = cap.read()
        if not ret:
            break

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.2, 6)

        for (x, y, w, h) in faces_detected:
            roi_gray = gray_img[y:y + w, x:x + h]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            img_pixels = np.expand_dims(roi_gray, axis=0)
            img_pixels = np.expand_dims(img_pixels, axis=-1)
            img_pixels = img_pixels.astype(np.float32) / 255.0

            interpreter.set_tensor(input_details[0]['index'], img_pixels)
            interpreter.invoke()
            predictions = interpreter.get_tensor(output_details[0]['index'])[0]

            for i, emotion in enumerate(['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']):
                emotion_percentages[emotion] = predictions[i] * 100
                if emotion_percentages[emotion] > 40:  # consider only emotions with a confidence level above 40%
                    emotion_counts[emotion] += 1

    cap.release()
    cv2.destroyAllWindows()

    # Initialize an empty list to store emotion-percentage pairs
    emotion_data = []

    # Iterate over the emotion percentages dictionary
    for emotion, percentage in emotion_percentages.items():
        # Append the emotion and percentage as a tuple to the list
        emotion_data.append((emotion, percentage))

    # Sort the emotion data based on percentage in descending order
    emotion_data.sort(key=lambda x: x[1], reverse=True)

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

