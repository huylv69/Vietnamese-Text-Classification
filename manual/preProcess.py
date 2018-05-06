import numpy as np
import os
import cv2
from skimage import io

# rename data to numbering type, ex: 1.jpg, 2.jpg,...
def renameData(inFolder, outFolder):
    print('--STARTING RENAME DATA---')
    listDirs = os.listdir(inFolder)
    for subDir in listDirs:
        number = 1
        path = inFolder + subDir + '/'
        for imgfile in os.listdir(path):
            img = cv2.imread(path + imgfile)
            name, ext = os.path.splitext(imgfile)
            print(path + imgfile)
            pathOut = outFolder + subDir
            if not os.path.exists(pathOut):
                os.makedirs(pathOut)
            try:
                cv2.imwrite(pathOut + '/' + str(number) + ext, img)
                number += 1
            except Exception as ex:
                print('Cannot save: ' + path + imgfile)
    print('---RENAME COMPETED---\n')

# Detect face and resize image to (32x32) px.
def faceDetectAndResizeImg(inputFolder, outFaceFolder):
    faceDetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    print('\n---STARTING DETECT AND RESIZE FACE---')
    listDirs = os.listdir(inputFolder)
    for subDir in listDirs: 
        path = inputFolder + subDir + '/'
        for imgfile in os.listdir(path):
            if os.path.isfile(path + imgfile):
                img = cv2.imread(path + imgfile)
                # Detect Faces
                faces = faceDetect.detectMultiScale(img)
                
                # IF CAN'T DETECT FACE --> CONTINUE!
                if (isinstance(faces, tuple)):
                    continue
                
                print(path + imgfile)
                num = ''
                for x, y, w, h in faces:
                    if(faces.shape != (1, 4)):
                        num = num + 'a'
                    if(h < 32 or w < 32): # face detect maybe wrong --> not save!
                        continue

                    faceCrop = img[y:y + h, x:x + w]
                    croppedImg = cv2.resize(faceCrop, (32, 32)) # resize to 32x32 px
                    if not os.path.exists(outFaceFolder + subDir):
                        os.makedirs(outFaceFolder + subDir)
                    cv2.imwrite(outFaceFolder + subDir + '/' +num + imgfile, croppedImg)

    print('---FACE DETECT COMPLETED!---\n')


def writeVector(faceFolder, fileVector):

    listFace  = []
    listLabel = []

    listDirs = os.listdir(faceFolder)
    print('---READING IMAGE---')
    for subDir in listDirs:
        path = faceFolder + subDir + '/'
        for imgfile in os.listdir(path):
            img = io.imread(path + imgfile, as_grey=True)
            # print(path + imgfile)
            label = listDirs.index(subDir)
            if(img.shape != (32, 32)):
                continue

            listFace.append(img)
            listLabel.append(label)
    print('---READ IMAGE COMPLETED---\n')
    
    XdataTemp = np.array(listFace)
    Xdata = XdataTemp.reshape(len(XdataTemp), 32 * 32)
    Ylabel = np.array(listLabel)

    print('dataShape: ', Xdata.shape)
   
    print('   WRITING VECTOR...')
    if not os.path.exists('vector'):
        os.makedirs('vector')
    writeFile(Xdata, Ylabel, 'vector/' + fileVector)

    print('---SAVE VECTOR COMPLETE!---\n')


def writeFile(Xdata, Ylabel, filename):
    fileWrite = open(filename, 'w')
    for i in range(0, len(Ylabel)):
        fileWrite.write(str(Ylabel[i]) + ' ')
        for i2 in range(0, 1024):
            fileWrite.write(str(i2) + ':' + str(Xdata[i][i2]) + ' ')

        fileWrite.write('\n')
    fileWrite.close()

# renameData('rawdata/', 'dataset/')
faceDetectAndResizeImg('dataset/', 'face/')
# renameData('face/', 'faceData/')
# renameData('facefilter/', 'faceData/')
writeVector('face/', 'vectorFull')

