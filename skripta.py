# u 96. redu obratiti paznju na putanju

import os.path

import numpy as np
from skimage import img_as_ubyte
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize
from sklearn.datasets import fetch_mldata
from sklearn.neighbors import KNeighborsClassifier

import cv2

def prepareTrainData(data):
    close_kernel = np.ones((5, 5), np.uint8)
    len_data=len(data)
    for i in range(0, len_data):
        number = data[i].reshape(28, 28)
        th = cv2.inRange(number, 150, 255)
        closing = cv2.morphologyEx(th, cv2.MORPH_CLOSE, close_kernel)
        labeled = label(closing)
        regions = regionprops(labeled)
        len_regions=len(regions)
        if(len_regions > 1):
            max_height = 0
            max_width = 0
            for region in regions:
                t_bbox = region.bbox
                t_height = t_bbox[2] - t_bbox[0]
                t_width = t_bbox[3] - t_bbox[1]
                if(max_height < t_height and max_width < t_width):
                    bbox = t_bbox
                    max_width = t_width
                    max_height = t_height
        else:
            bbox = regions[0].bbox
        x = 0
        img = np.zeros((28, 28))
        bbx1=bbox[1]
        bbx2=bbox[2]
        bbx0=bbox[0]
        bbx3=bbox[3]
        y = 0
        for row in range(bbx0, bbx2):
            for col in range(bbx1, bbx3):
                img[x, y] = number[row, col]
                y_ink=y+1
                y=y_ink
            x += 1
        data[i] = img.reshape(1, 784)

def getNumberImage(bbox, img):
    bbx1 = bbox[1]
    bbx2 = bbox[2]
    bbx0 = bbox[0]
    bbx3 = bbox[3]
    height = bbx2 - bbox[0]  #bbox[0] je min_row
    width = bbx3 - bbox[1]  #bbox[0] je min_col
    img_number = np.zeros((28, 28))
    for x in range(0, height):
        for y in range(0, width):
            x_dekr=x-1
            y_dekr=y-1
            img_number[x, y] = img[x_dekr + bbox[0], y_dekr + bbox[1]]
    return img_number

def presekSaPravom(bbox, height, width):
    bbx3 = bbox[3]
    bbx0 = bbox[0]
    bbx1 = bbox[1]
    bbx2 = bbox[2]
    if( x2<(bbx1) or x1>(4+bbx3) or y2>(4+bbx2)):
        return False
    recEnd = (y1*x2-y2*x1)+ (x1-x2)*(1+bbx2) + (y2-y1)*(1+bbx3)
    if(recEnd <= 0): #Ukoliko je kraj regiona ispod linije racunati kao da je ceo pravouganik presao liniju
        return False
    TL = (y1*x2-y2*x1)+ (x1-x2)*bbx0 + (y2-y1)*bbx1
    TR = (y1*x2-y2*x1)+ (x1-x2)*bbx0 + (y2-y1)*(4+bbx3)
    BL = (y1*x2-y2*x1)+ (x1-x2)*(4+bbx2) + (y2-y1)*bbx1
    BR = (y1*x2-y2*x1)+ (x1-x2)*(4+bbx2) + (y2-y1)*(4+bbx3)
    if (0 < BR and 0 < TR and 0 < TL and 0 < BL):
        return False
    elif (0 > BR and 0 > TR and 0 > TL and 0 > BL):
        return False
    else:
        return True

def addNumber(width, lista, bbox, broj):
    for tup in lista:
        t2=tup[2]
        t1=tup[1]
        t3=tup[3]
        t0=tup[0]
        if(width==t3 and (5+bbox[1])>t1 and broj==t0 and (5+bbox[0])>t2 ):
            lista.remove(tup)
            lista.append((broj, bbox[1], bbox[0], width))
            return False
    bbx0=bbox[0]
    bbx1=bbox[1]
    lista.append((broj, bbx1, bbx0, width))

mnistOrString = 'MNIST original'
mnistOriginal = fetch_mldata(mnistOrString)

DIRString = 'C:\Users\Mutic\Desktop\SoftProjekatDjole'
DIR = DIRString
if(os.path.exists(os.path.join(DIR, 'mnistPrepared')+'.npy')):
    train = np.load(os.path.join(DIR, 'mnistPrepared')+'.npy')
else:
    train = mnistOriginal.data
    np.save(os.path.join(DIR, 'mnistPrepared'), train)
    prepareTrainData(train)
train_labels = mnistOriginal.target
knn = KNeighborsClassifier(n_neighbors=1, algorithm='brute').fit(train, train_labels)

eros_kernel = np.ones((2, 2), np.uint8)#Kernel koriscen prilikom erozije (priprema za skeleton operaciju)
close_kernel = np.ones((4, 4), np.uint8)

video_imena = [os.path.join(DIR+'\\Videos', naziv) for naziv in os.listdir(DIR+'\\Videos') if os.path.isfile(os.path.join(DIR+'\\Videos', naziv))]
#ODAVDE
izlazniFajl = open("out.txt","w")
izlazniFajl.write("Suma brojeva sa obradjenih video materijala: \n")
izlazniFajl.write("RA 65/2013 Djordje Mutic\n")
#DOVDE
broj_snimaka=len(video_imena)
for redniBrojSnimka in range(0, broj_snimaka):
    videoPutanja=video_imena[redniBrojSnimka]
    print 'Putanja do videa: ' + videoPutanja
    cap = cv2.VideoCapture(videoPutanja)
    frameNum = 0
    lista_brojeva=[]
    while(cap.isOpened()):
        ret, frame = cap.read()
        if(frameNum%2 != 0):
            frameNum += 1
            continue
        if(ret == False):
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if(frameNum == 0):
            line_th = cv2.inRange(gray, 4, 55)
            erosion = cv2.erode(line_th, eros_kernel, iterations=1)
            skeleton = skeletonize(erosion/255.0)
            cv_skeleton = img_as_ubyte(skeleton)
            lines = cv2.HoughLinesP(cv_skeleton, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
            x1, y1, x2, y2 = lines[0][0]
        th = cv2.inRange(gray, 163, 255)
        closing = cv2.morphologyEx(th, cv2.MORPH_CLOSE, close_kernel)
        gray_labeled = label(closing)
        regions = regionprops(gray_labeled)
        for region in regions:
            bbox = region.bbox
            height = bbox[2]-bbox[0]
            width = bbox[3]-bbox[1]
            if(height <= 10):
                continue
            if(presekSaPravom(bbox, height, width) == False):
                continue
            img_number = getNumberImage(bbox, gray)
            num = int(knn.predict(img_number.reshape(1, 784)))
            #cv2.line(gray,(x1,y1),(x2,y2),(255,255,255),1)
            #plt.imshow(gray, 'gray')
            #plt.show()
            if (addNumber(width, lista_brojeva, bbox, num) == False):
                continue
            #print 'U frejmu '+ str(frameNum)+ '. prepoznat broj '+str(num)
        frameNum += 1
        cv2.imshow('frame', frame)
    suma=0
    for tup in lista_brojeva:
        suma += tup[0]
    print 'Suma: '+str(suma)+'\n'
    #OVDE da upisem u out.txt resenje za video
    videoSnimak = "videos/video-" + format(redniBrojSnimka) + ".avi"
    nazivVideoSnimka = "video-" + format(redniBrojSnimka) + ".avi"
    izlazniFajl.write("Naziv video snimka: " + str(videoSnimak) + " \n")
    izlazniFajl.write("Rezultat: " + str(suma) + " \n")
    #DOVDE
cap.release()
cv2.destroyAllWindows()