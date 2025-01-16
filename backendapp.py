import tensorflow as tf
from tensorflow import keras
import cv2
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from keras.utils import to_categorical
# import pyautogui
import time
from pynput.keyboard import Controller, Key
import time
import pickle
import os
import numpy as np 
import subprocess
import webbrowser

try:
    functions = pickle.load(open('functions.dat', 'rb')) 
except:
    functions = [['None','None'],['None','None'],['None','None'],['None','None'],['None','None'],['None','None'],['None','None'],['None','None'],['None','None'],['None','None']] 
functions = [['None','None'],['keys','cmd shift ]'],['keys','cmd shift ]'],['keys','cmd m'],['keys','cmd z'],['None','None'],['cmd','b'],['None','None'],['None','None'],['None','None'],['None','None']] 
print(functions)


actions = np.array(['none','close', 'swiperight','zoom', 'swipeleft','cut', 'threefinger', 'twofinger', 'fivefingerleft', 'fivefingerright'])
sequence_length = 12
no_sequence = 35

model = tf.keras.models.load_model('handmodel.h5') 
model.summary() 

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_styled_landmarks(image, results):
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             )

def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])

colors = [(245,117,16), (117,245,16), (16,117,245), (0,117,245),(16,117,245),(16,117,245)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[0], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame

def shortcut_click(keyslist):
    keyboard = Controller()
    time.sleep(2)
    print(keyslist)
    # Press keys from the list
    for key in keyslist:
        # Check if the key is a special key in the Key class
        if hasattr(Key, key):
            keyboard.press(getattr(Key, key))
        else:
            keyboard.press(key)
        time.sleep(0.1)

    # Release keys in reverse order
    for key in reversed(keyslist):
        if hasattr(Key, key):
            keyboard.release(getattr(Key, key))
        else:
            keyboard.release(key)

label_map = {label:num for num, label in enumerate(actions)}

def action(result):
    action_type = functions[label_map[result]][0]
    action_content = functions[label_map[result]][1]
    print(action_content)
    if action_type == 'keys':
        shortcut_click(action_content.split())
    if action_type == 'text':
        keyboard = Controller()
        keyboard.type(action_content)
        print(action_content)
    if action_type == 'file':
        subprocess.Popen(['open', action_content])
    if action_type == 'website':
        webbrowser.open(action_content)
    else:
        pass

sequence = []
sentence = []
predictions = []
threshold = 0.85

cap = cv2.VideoCapture(0)
# Set mediapipe model 
count = 0
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        if count >0:
            count -= 1
        # Read feed
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        #print(results)
        
        # Draw landmarks
        draw_styled_landmarks(image, results)
        
        # 2. Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-12:]
        
        if len(sequence) == 12:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            #print(actions[np.argmax(res)])
            predictions.append(np.argmax(res))
            
            
        #3. Viz logic
            if np.unique(predictions[-10:])[0]==np.argmax(res) and count == 0: 
                if res[np.argmax(res)] > threshold: 
                    if len(sentence) > 0: 
                        if actions[np.argmax(res)] != sentence[-1] and actions[np.argmax(res)] == 'none':
                            preaction = actions[np.argmax(res)] 
                            sentence.append(actions[np.argmax(res)])
                    if len(sentence) > 0: 
                        if actions[np.argmax(res)] != sentence[-1] and preaction == "none":
                            preaction = actions[np.argmax(res)] 
                            sentence.append(actions[np.argmax(res)])
                            action(actions[np.argmax(res)])
                    else:
                        preaction = actions[np.argmax(res)]
                        sentence.append(actions[np.argmax(res)])
                        action(actions[np.argmax(res)])
                        count = 10
    

            if len(sentence) > 5: 
                sentence = sentence[-5:]

            # Viz probabilities
            image = prob_viz(res, actions, image, colors)
            
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
