import numpy as np
from keras.models import model_from_json
import operator
import cv2
import sys, os

# Loading the model
json_file = open("modelHand.json", "r")
model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(model_json)
# load weights into new model
loaded_model.load_weights("modelHand.h5")
print("Loaded model from disk")

cap = cv2.VideoCapture(0)

# Category dictionary
#categories = {01_palm: 'Palm', 02_l: 'Loser', 03_fist: 'Fist', 05_thumb: 'Thumbs Up', 06_index: 'Index Up', 07_ok: 'Okay', 09_c: 'C'}

while True:
    _, frame = cap.read()
    # Simulating mirror image
    frame = cv2.flip(frame, 1)
    
    # Got this from collect-data.py
    # Coordinates of the ROI
    x1 = int(0.5*frame.shape[1])
    y1 = 10
    x2 = frame.shape[1]-10
    y2 = int(0.5*frame.shape[1])
    # Drawing the ROI
    # The increment/decrement by 1 is to compensate for the bounding box
    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
    # Extracting the ROI
    roi = frame[y1:y2, x1:x2]
    
    # Resizing the ROI so it can be fed to the model for prediction
    roi = cv2.resize(roi, (64, 64)) 
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, test_image = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
    cv2.imshow("test", test_image)
    # Batch of 1
    result = loaded_model.predict(test_image.reshape(1, 64, 64, 1))
    prediction = {'Palm': result[0][0], 
                  'Loser': result[0][1], 
                  'Fist': result[0][2],
                  'Thumbs Up': result[0][3],
                  'Index Up': result[0][4],
                  'Okay': result[0][5],
                  'C': result[0][6]}
    # Sorting based on top prediction
    prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
    
    if prediction[0][0]=='Palm':
        emoji=cv2.imread('Emoji/palm.png')
    elif prediction[0][0]=='Loser':
        emoji=cv2.imread('Emoji/loser.png')
    elif prediction[0][0]=='Fist' or prediction[0][0]=='C':
        emoji=cv2.imread('Emoji/fist.png')
    elif prediction[0][0]=='Index Up' or prediction[0][0]=='Thumbs Up':
        emoji=cv2.imread('Emoji/index.png')
    elif prediction[0][0]=='Loser' :
        emoji=cv2.imread('Emoji/loser.png')
    else:
        emoji=cv2.imread('Emoji/okay.png')
        
    width = 500
    height = 500
    dim = (width, height)
        
    emoji = cv2.resize(emoji, dim, interpolation = cv2.INTER_AREA)
        
        
    cv2.imshow("Emoji", emoji)
    
    # Displaying the predictions
    cv2.putText(framess, prediction[0][0], (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)    
    cv2.imshow("Frame", frame)
    
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == ord('s'): # esc key
        break
        
 
cap.release()
cv2.destroyAllWindows()

