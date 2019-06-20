import cv2
import os
#from google.colab.patches import cv2_imshow
import os.path
import numpy as np
import tensorflow as tf
from imageio import imread


cap = cv2.VideoCapture(0)
(ret, frame) = cap.read()
prediction = 'n.a.'

#from keras.models import Sequential
from keras.models import load_model
cnn_model = load_model('/Users/mihirmahajan/desktop/project/keras_cnn_model.hdf5')
#print(cnn_model.class_indices)

def test(frame):
    x_t = []
    #img = imread(frame)
    img = frame
    #print (frame)
    #print (frame)
    width = 128
    height =128
    dim = (width, height)
    # resize image
    img_RGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_RGB, dim, interpolation = cv2.INTER_AREA)
    x_t.append(resized)
    #print (x_t)
    x_t = np.array(x_t)
    x_t = x_t/255
    #print (x_t.shape)
    #x_t = np.array(x_t)
    #print (x_t)
    y_pred = cnn_model.predict(x_t)
    predct = np.argmax(y_pred,axis=1) 
    pred = int(predct)
    return pred

while True:
    

    # Capture frame-by-frame
    (ret, frame) = cap.read()
    #'Prediction: ' + prediction,
    cv2.putText(
        img=frame,
        text=prediction,
        org=(30, 70),
        fontFace=cv2.FONT_HERSHEY_PLAIN,
        fontScale=6.0,
        color=(255, 255, 255),
        thickness=2
        )


    # Display the resulting frame
    cv2.imshow('color detection',frame)

    testout=test(frame)
    d = {0:'black',1:'blue',2:'green',3:'orange',4:'red',5:'violet',6:'white',7:'yellow'}
    prediction= d[testout]
    """
    if testout == 0:
        prediction = "Black"
    elif testout == 1:
        prediction = "Blue"
    elif testout == 2:
        prediction = "Green"
    elif testout == 3:
        prediction = "Orange"
    elif testout == 4:
        prediction = "Red"
    elif testout == 5:
        prediction = "Violet"
    elif testout == 6:
        prediction = "White"
    elif testout == 7:
        prediction = "Yellow"                
    """

    #prediction = knn_classifier.main('training.data', 'test.data')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()	
