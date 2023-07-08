import cv2
import cvzone
import tensorflow as tf
from cvzone.ClassificationModule import Classifier
import os
import pickle
import numpy as np

cap = cv2.VideoCapture(0)
classifier = Classifier('Resources/Model/keras_model.h5', 'Resources/Model/labels.txt')
filename = 'model/test_model.sav'
classifier_v2 =  tf.keras.models.load_model('saved_model/my_model')
imgArrow = cv2.imread('Resources/arrow.png', cv2.IMREAD_UNCHANGED)
classIDBin = 0
imgWasteList = []
# labels
predicted_labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
pathFolderWaste = "Resources/Waste" # 1: 'cardboard', 2: 'glass', 3: 'metal', 4: 'paper', 5: 'plastic', 6: 'trash'

pathList = os.listdir(pathFolderWaste)
for path in pathList:
    imgWasteList.append(cv2.imread(os.path.join(pathFolderWaste, path), cv2.IMREAD_UNCHANGED))

# Import all the waste images
imgBinsList = []
pathFolderBins = "Resources/Bins"
pathList = os.listdir(pathFolderBins)
print(pathList)
for path in pathList:
    imgBinsList.append(cv2.imread(os.path.join(pathFolderBins, path), cv2.IMREAD_UNCHANGED))
# 0 = Paper/Cardboard
# 1 = Plastic
# 2 = Metal
# 3 = Glass
# 4 = Trash
classDic = {0: None,
            1: 0,
            2: 3,
            3: 2,
            4: 0,
            5: 1,
            6: 4}
while True:
    _, img = cap.read()
    imgResize = cv2.resize(img, (454, 340))
    imgForModel = cv2.resize(img, (256, 256))
    imgBackground = cv2.imread('Resources/background.png')
    #img = tf.keras.preprocessing.image.load_img(imgResize, target_size=(256, 256))
    img_array = tf.keras.preprocessing.image.img_to_array(imgForModel)
    img_array = tf.expand_dims(img_array, 0) 
    prediction = classifier_v2.predict(img_array)
    index = np.argmax(prediction)
    item = predicted_labels[index]
    #classID = predection[1]
    print(index)
    if index != 5:
        imgBackground = cvzone.overlayPNG(imgBackground, imgWasteList[index], (909, 127))
        imgBackground = cvzone.overlayPNG(imgBackground, imgArrow, (978, 320))

        classIDBin = classDic[index+1]
    else: 
        imgBackground = cvzone.overlayPNG(imgBackground, imgWasteList[5], (909, 127))
        imgBackground = cvzone.overlayPNG(imgBackground, imgArrow, (978, 320))
        classIDBin = classDic[6]
    imgBackground = cvzone.overlayPNG(imgBackground, imgBinsList[classIDBin], (895, 374))

    imgBackground[148:148 + 340, 159:159 + 454] = imgResize
    # Displays
    # cv2.imshow("Image", img)
    cv2.imshow("Output", imgBackground)
    cv2.waitKey(1)