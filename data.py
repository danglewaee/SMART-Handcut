import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
 
actions = np.array(['test1', 'test2', 'test3'])

#how many videos per action 
dataset_size = 45
#how many frames per video
number_of_frames = 12


DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# ------------Mediapiipe tools--------
mp_holistic = mp.solutions.holistic #holistic model
mp_drawing = mp.solutions.drawing_utils #drawing utitlity

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #Convert bgr to rgb for mediapipe
    image.flags.writeable = False #Save memory
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    print(np.concatenate([lh, rh]))
    return(np.concatenate([lh, rh]))

# --------------------

lable_map = {label:num for num, label in enumerate(actions)}

cap = cv2.VideoCapture(0)

fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
frame_width = int(cap.get(3)) 
frame_height = int(cap.get(4)) 
size = (frame_width, frame_height)

#path to save numerical data
DATA_PATH = os.path.join("MP_Data")

for action in actions:
    for sequence in range(dataset_size):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

for action in actions:
    #save visual data for double check
    if not os.path.exists(os.path.join(DATA_DIR, action)):
        os.makedirs(os.path.join(DATA_DIR, action))
    print("collecting data for classs {}".format(action))

    #waiting screen for each action
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, "Ready? Press 'Q' ! :", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA) 
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break
    
    #Collecting data screen - click C to collect and wait until the red text turn off
    for sequence in range(dataset_size):
        out = cv2.VideoWriter(os.path.join(DATA_DIR, action, '{}.mp4'.format(sequence)), fourcc, fps, size)
        frame_count = 31
        with mp_holistic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence=0.5) as holistic:
            while True:
                ret, frame = cap.read()
                image = cv2.flip(frame, 1)
                cv2.putText(image, "Ready? Press 'c' ! :", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA) 
                #make prediction
                image, results = mediapipe_detection(image, holistic)
                #draw landmarks
                draw_landmarks(image, results)

                if frame_count == 0:
                    cv2.putText(image, "Starting collection", (120,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 4, cv2.LINE_AA)
                    cv2.putText(image, f"Collecting frames for {action} video number {sequence}", (15,12), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 1, cv2.LINE_AA)
                elif frame_count >0 and frame_count < 30:
                    cv2.putText(image, f"Collecting frames for {action} video number {sequence}", (15,12), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 1, cv2.LINE_AA)
                cv2.imshow('frame', image)
                if frame_count < 30:
                    out.write(frame)
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_count))
                    np.save(npy_path, keypoints)
                    frame_count += 1


                elif frame_count == 30:
                    break
                if cv2.waitKey(25) == ord('c'):
                    frame_count = 0

cap.release()
cv2.destroyAllWindows()
    
