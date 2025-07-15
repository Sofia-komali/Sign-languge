import cv2
import mediapipe as mp
import time

cap=cv2.VideoCapture(0)
mp_hand=mp.solutions.hands
hand=mp_hand.Hands(max_num_hands=2,min_detection_confidence=0.7,min_tracking_confidence=0.5)
mp_draw=mp.solutions.drawing_utils
gesture_sequence = []
full_sentence = ""
sentence_timestamp=0

gesture_dict = {
                    ((0, 0, 0, 0, 0), (1, 0, 0, 0, 0)): "Hello",
                    ((1, 0, 0, 0, 1), (1, 1, 1, 1, 1)): "Goodbye",
                    ((0, 1, 1, 0, 0), (1, 0, 0, 0, 1)): "Thank You",
                    ((0, 1, 0, 0, 1), (0, 0, 1, 1, 1)): "Sorry",
                    ((1, 0, 0, 0, 0), (0, 1, 1, 0, 0)): "Yes",
                    ((0, 1, 0, 0, 1), (1, 0, 0, 0, 0)): "No",
                    ((0, 0, 1, 1, 1), (1, 0, 0, 0, 0)): "Maybe",
                    ((1, 0, 0, 0, 1), (0, 1, 1, 0, 0)): "Please",
                    ((0, 1, 1, 0, 0), (1, 0, 0, 0, 1)): "Good Morning",
                    ((1, 0, 0, 0, 1), (0, 1, 1, 0, 0)): "Good Night",
                    ((0, 1, 0, 0, 1), (1, 0, 0, 0, 1)): "How are you?",
                    ((1, 0, 0, 0, 1), (0, 1, 0, 0, 1)): "I love you",
                    ((0, 1, 0, 0, 1), (1, 1, 1, 1, 1)): "Good Luck",
                    ((1, 1, 1, 1, 1), (0, 0, 0, 0, 0)): "Goodbye",
                    ((0, 0, 0, 0, 0), (1, 1, 1, 1, 1)): "Welcome",
                    (("call", "peace"),): "Hello",
                    (("peace", "Thumbs Up"),): "What's your name?",
                    (("peace", "peace"),): "Nice to meet you",
                    (("call", "call"),): "See you later",
                    (("peace", "call"),): "Take care",
                    (("call", "rock on"),): "Have a good day",
                    (("rock on", "rock on"),): "Goodbye",
                    (("rock on", "peace"),): "See you soon",
                    (("rock on", "call"),): "Take care",
                    (("rock on", "Thumbs Up"),): "Have a nice day",
                    (("Thumbs Up", "Thumbs Up"),): "Goodbye",
                    (("Thumbs Up", "peace"),): "See you later",
                    (("Thumbs Up", "call"),): "Take care",
                    (("Thumbs Up", "rock on"),): "Have a great day",
                    }
while True:
    success,img=cap.read()
    
    if not success:
        break
    
    img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    result=hand.process(img_rgb)
    img_height,img_width,_=img.shape
    

    if result.multi_hand_landmarks:
        num_hands=len(result.multi_hand_landmarks)
        hand_gestures=[]
        
        
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(img,hand_landmarks,mp_hand.HAND_CONNECTIONS)
                
            thumb_tip_x = hand_landmarks.landmark[4].x * img_width
            thumb_base_x = hand_landmarks.landmark[2].x * img_width
            thumb_up=thumb_tip_x>thumb_base_x
                
            index_tip_y = hand_landmarks.landmark[8].y * img_height
            index_pip_y = hand_landmarks.landmark[6].y * img_height
            index_up=index_tip_y<index_pip_y
                
            middle_tip_y = hand_landmarks.landmark[12].y * img_height
            middle_pip_y = hand_landmarks.landmark[10].y * img_height
            middle_up=middle_tip_y<middle_pip_y
                
            ring_tip_y = hand_landmarks.landmark[16].y * img_height
            ring_pip_y = hand_landmarks.landmark[14].y * img_height
            ring_up=ring_tip_y<ring_pip_y
                
            pinky_tip_y = hand_landmarks.landmark[20].y * img_height
            pinky_pip_y = hand_landmarks.landmark[18].y * img_height
            pinky_up=pinky_tip_y<pinky_pip_y
                
            fingers=[thumb_up,index_up,middle_up,ring_up,pinky_up]
                
            gestures={
                    (0,0,0,0,0):"Fist",
                    (1,0,0,0,0):"Thumbs Up",
                    (0,1,1,0,0):"Peace",
                    (1,0,0,0,1):"Call",
                    (0,1,0,0,1):"Rock on",
                    (0,0,1,1,1):"OK",
                    (1,1,1,1,1):"High Five",
                    
                }
            gesture = gestures.get(tuple(fingers))
            
            
                
            if gesture:
                min_x = min([landmark.x for landmark in hand_landmarks.landmark]) * img_width
                max_x = max([landmark.x for landmark in hand_landmarks.landmark]) * img_width
                min_y = min([landmark.y for landmark in hand_landmarks.landmark]) * img_height
                max_y = max([landmark.y for landmark in hand_landmarks.landmark]) * img_height
                    
                cv2.rectangle(img, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (0, 255, 0), 2)
                cv2.putText(img, gesture, (int(min_x), int(min_y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                center_x = int(hand_landmarks.landmark[0].x * img_width)
                center_y = int(hand_landmarks.landmark[0].y * img_height)
                cv2.circle(img, (center_x, center_y), 5, (0, 255, 0), -1)
                hand_gestures.append(gesture)
                    
        if len(gesture_sequence) > 0:
            for pattern, sentence in gesture_dict.items():
                pattern_len = len(pattern)
            if len(gesture_sequence) >= pattern_len:
                if tuple(gesture_sequence[-pattern_len:]) == pattern:
                    full_sentence = sentence
                    gesture_sequence.clear()
                    break

            
        elif num_hands == 2 and len(hand_gestures) == 2:
            gesture_pair = tuple(hand_gestures)
            if gesture_pair in gesture_dict:
                full_sentence = gesture_dict[gesture_pair]

    
    if full_sentence:
        cv2.putText(img, full_sentence, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)


                    
        
    cv2.imshow("Image",img)
    key= cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
