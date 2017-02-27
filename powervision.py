from __future__ import unicode_literals
from skimage.filters import threshold_adaptive
import numpy as np
import cv2
import imutils
import os
import pytesseract
from matplotlib import pyplot as plt
import pyttsx
import imutils
import glob
from PIL import Image
from nltk import tokenize
import serial
from textblob import TextBlob
import language_check
import subprocess

pushlist =[]
pointer =0
pointer1 =0
poplist =[]
# ser = serial.Serial('/dev/ttyUSB0', 9600)
# a=glob.glob('/dev/ttyUSB*')    
# ser = serial.Serial(a[0], 9600)

def poplist_function(text):
    global poplist
    global pointer1
    global pushlist
    global pointer
   
    poplist.append(text)
    pointer1 =pointer1 +1

def emptyfunction():
    global poplist
    global pointer1
    global pushlist
    global pointer
    global ser
    if (pointer1>=0):
        pointer1 =pointer1 -1
        ter = poplist[pointer1]
        ter ='"' + ter + '"'
        subprocess.call('echo '+ter+'|festival --tts', shell=True)
        pointer = pointer +1
        # print ("this is pointer in empty ") 
        print pointer
        x= ser.inWaiting()
        print x
        if (x != 0):
            print ("went into loop")
            serVal = ser.readline()
            print serVal
            if('D' in serVal):
                print ("pause is pressed")
                serVal = ser.readline()
                if('C' in serVal):
                    pop_function()
                else :
                    emptyfunction()
                
            else :
                pop_function()

def pop_function():
    global poplist
    global pointer1
    global pushlist
    global pointer
    pointer = pointer-1
    # print ("this is pointer in pop") 
    print pointer
    text = pushlist[pointer]
    poplist_function(text)
    emptyfunction()

def push_function(text):
    global poplist
    global pointer1
    global pushlist
    global pointer
    pushlist.append(text)
    pointer = pointer +1
    # print ("this is pointer")
    print pointer

def Camera_Capture(N=1):
    cap = cv2.VideoCapture(1)
    ret = cap.set(3, 1920)
    ret = cap.set(4,1080)
    ret,frame = cap.read()
    cap.release()
    cv2.destroyAllWindows()
    return frame

def init():
    os.system('v4l2-ctl -d /dev/video1 -c brightness=180')
    os.system('v4l2-ctl -d /dev/video1 -c contrast=100')
    os.system('v4l2-ctl -d /dev/video1 -c saturation=0')
    os.system('v4l2-ctl -d /dev/video1 -c sharpness=220')

def Capture_Vlc_Img(path):
    a='vlc -I dummy v4l2:///dev/video1:width=1920:height=1080 --video-filter scene --no-audio --scene-path '+path+' --scene-prefix image_prefix --scene-format tiff --scene-replace  vlc://quit --run-time=6'
    os.system(a)

def order_points(pts):
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    # warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight),flags=cv2.INTER_CUBIC)
    return warped

def total_col(warped_new,col,y1_row,y2_row):
    total=[]
    for i in range(0,col):
        j=0
        for a in warped_new[y1_row:y2_row,i]:
            if a < 255:
                j=j+1
        total.append(j)
    return total

def total_row(warped_new,row,x1_col,x2_col):
    total=[]
    for i in range(0,row):
        j=0
        for a in warped_new[i,x1_col:x2_col]:
            if a < 255:
                j=j+1
        total.append(j)
    return total

def show(image,Name='Outline'):
    cv2.imshow(Name,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def Row_Cropping(total,row,col,N=10):
    breaking_pt1_row=[]
    breaking_pt2_row=[]
    row_new=col/N
    i=0
    while i<row-1:
        i+=1
        # print " outer col"
        # print i
        if total[i]<row_new:
            while total[i]<row_new and i<row-1:
                # print "breaking_pt1 loop, i =" + str(i)
                i+=1
                if i==row-1:
                    break
            breaking_pt1_row.append(i)
            while total[i]>row_new and i<row-1:
                # print "breaking_pt2 loop, i =" + str(i)
                i+=1
                if i==row-1:
                    break
            breaking_pt2_row.append(i)
    return breaking_pt1_row,breaking_pt2_row

def Column_Cropping(total,row,col,N=10):
    breaking_pt1_row=[]
    breaking_pt2_row=[]
    row_new=row/N
    i=0
    while i<col:
        # print " outer col"
        # print i
        # print "total = " + str(total[i])  
        if total[i]<row_new:
            while total[i]<row_new and i<col:
                # print "breaking_pt1 loop, i = " + str(i)+ "total = "+str(total[i])+"threshold = "+str(row_new)
                i+=1
                if i==col:
                    break
            breaking_pt1_row.append(i)
            while i<col and total[i]>row_new:
                # print "breaking_pt2 loop, i =" + str(i)+ "total = "+str(total[i])+" threshold = "+str(row_new)
                i+=1
                if i==col:
                    break
            breaking_pt2_row.append(i)
        i+=1
    return breaking_pt1_row,breaking_pt2_row

def speak(image):
    a=pytesseract.image_to_string(Image.open(image))
    #print a
    #print ("ekkada nunchi tesuko ra rei vinnava ledha ")
    #print unidecode(a)
    #tyu =re.sub(r'[^\x00-\x7F]+',' ',a)
    #print tyu
    #a=a.encode('ascii',errors='ignore')
    #a =''.join([i if ord(i) < 128 else ' ' for i in a])
    sentences_list = tokenize.sent_tokenize(a.decode("ascii", 'ignore'))
    # print sentences_list
    tool=language_check.LanguageTool('en-US')
    for text in  sentences_list :
        cleaned_text = text.replace('\n','')
        cleaned_text = text.replace('/','')
        cleaned_text = text.replace('.','')
        cleaned_text = text.replace('?','')
        #cleaned_text = ["\" + x for x in cleaned_text.split()]
        
        # print cleaned_text
        text_blob=TextBlob(cleaned_text)
        text_blob_correct=text_blob.correct()
        text_str=str(text_blob_correct)
        # print ("vgybhjn")
        #print text_str
        matches = tool.check(text_str)
        text_f=language_check.correct(text_str, matches)
        push_function(text_f)
        text_f = '"' + text_f + '"'
        print text_f
        subprocess.call('echo '+text_f+'|festival --tts', shell=True)
        if (ser.inWaiting() != 0):
            serVal = ser.readline()
            if('D' in serVal):
                print ("pause is pressed")
                serVal = ser.readline()
                if('C' in serVal):
                    pop_function()
                elif('D' in serVal):
                    continue
            elif('C' in serVal):
                pop_function()

def kernel(N=3,Type=1):
    kernel = np.zeros((N,N), dtype=np.uint8)
    if Type==1 :
        ''' Vertical kernel '''
        kernel[:,(N-1)/2] = 1
        return kernel
    if Type==2 :
        ''' Horizontal Kernel '''
        kernel[(N-1)/2,:] = 1
        return kernel
    if Type==3 :
        ''' Star Kernel  '''
        kernel[:,(N-1)/2] = 1
        kernel[(N-1)/2,:] = 1
        return kernel
    if Type==4 :
        ''' Box Kernel '''
        kernel = np.ones((N,N),np.uint8)
        return kernel

def Serial_Cap(path):
    a=glob.glob('/dev/ttyUSB*')    
    ser = serial.Serial(a[0], 9600)
    while (1):
      serVal = ser.readline()
      print serVal
      if ('D' in serVal):
          Capture_Vlc_Img(path)
          # image = Camera_Capture()
          break
    # return image

def Bounding(image,X=20):
    row,col,lol=image.shape
    image[0:X,:,:]=[0,0,0]
    image[row-X:row,:,:]=[0,0,0]
    image[:,0:X,:]=[0,0,0]
    image[:,col-X:col,:]=[0,0,0]
    return image

def PreProcessing(image):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(5,5), 0)
    ret3,th3 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return th3

def Bounding_Box(th3,x=0.02):
    (_,cnts, _) = cv2.findContours(th3, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE )
    cnts=sorted(cnts,key=cv2.contourArea, reverse=True)[:5]
    c=cnts[0]
    peri=cv2.arcLength(c,True)
    approx=cv2.approxPolyDP(c,x*peri,True)
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    if len(approx)==4 :
        box=approx
    return box

def Scanned(orig,box,threshold=251,offset_adaptive=10,ratio=1):
    warped =four_point_transform(orig,box.reshape(4,2)*ratio)
    warped = cv2.bilateralFilter(warped,9,75,75) #Bilateral Filtering
    warped =cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    # ret2,th2 = cv2.threshold(warped,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    warped = threshold_adaptive(warped, threshold, offset = offset_adaptive)
    warped = warped.astype("uint8")*255
    return warped

def compute_skew_angle(img):
    N=3
    kernel = np.zeros((N,N), dtype=np.uint8)
    kernel[(N-1)/2,:] = 1
    img = cv2.bitwise_not(img)
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.dilate(img,kernel,iterations=10)
    # show(gray)
    ret2,th2 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # show(th2)
    height, width = gray.shape
    angle = 0
    lines = cv2.HoughLinesP(th2, 1, np.pi/180, 100, minLineLength=width / 3.0, maxLineGap=10)
    nlines = lines.size/4
    for i in range(0,nlines):
        for x1,y1,x2,y2 in lines[i]:
            if (abs(np.arctan2(y2 - y1,x2 - x1))<1.04):
                # print np.arctan2(y2 - y1,x2 - x1)
                # cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
                angle += np.arctan2(y2 - y1,x2 - x1)
    angle = angle/nlines*180/np.pi
    print angle
    return angle

def deskew(img,angle):    
    height, width = img.shape
    img = cv2.bitwise_not(img)
    image= cv2.copyMakeBorder(img,50,50,50,50,cv2.BORDER_CONSTANT,value=0)
    center = (width / 2, height / 2)
    maxHeight,maxWidth =  image.shape[:2]
    M = cv2.getRotationMatrix2D(center,angle, 1.0)
    rotated = cv2.warpAffine(image, M, (maxWidth,maxHeight), flags=cv2.INTER_CUBIC)
    lite=cv2.getRectSubPix(rotated, (maxWidth,maxHeight), center)
    image = cv2.bitwise_not(lite)
    # image =  lite
    show(image,'rotated')

    return image

# def Crop(warped_new,warped):
#     row,col=warped_new.shape
#     total = total_col(warped_new,col,0,row)
#     breaking_pt1_col,breaking_pt2_col=Column_Cropping(total,row,col,10)
#     k=0
#     for i in range(len(breaking_pt1_col)):
#         diff = breaking_pt2_col[i]-breaking_pt1_col[i]
#         if diff>col/20:
#             lol=warped[:,breaking_pt1_col[i]:breaking_pt2_col[i]]
#             show(lol)
#             angle =  compute_skew_angle(lol)
#             if abs(angle)>5 and abs(angle)< 15:
#                 lol =  deskew(lol,angle)
#                 show(lol,'deskew')
#             cv2.imwrite("crop_"+str(k)+".jpg",lol)
#             k=k+1
#     return k

def Crop(warped_new,warped):
    row,col=warped_new.shape
    print row 
    print col
    total = total_col(warped_new,col,0,row)
    breaking_pt1_col,breaking_pt2_col=Column_Cropping(total,row,col,10)
    k=0
    for i in range(len(breaking_pt1_col)):
        diff = breaking_pt2_col[i]-breaking_pt1_col[i]
        if diff>col/20:
            total1=total_row(warped_new,row,breaking_pt1_col[i],breaking_pt2_col[i])
            breaking_pt1_row,breaking_pt2_row=Row_Cropping(total1,row,diff,25)
            print breaking_pt1_row
            for j in range(len(breaking_pt1_row)):
                diff_row = breaking_pt2_row[j]-breaking_pt1_row[j]
                if diff_row>row/100:
                    print  breaking_pt1_row[j]
                    lol = warped[breaking_pt1_row[j]:breaking_pt2_row[j],breaking_pt1_col[i]:breaking_pt2_col[i]]
                    show(lol)
                    angle =  compute_skew_angle(warped)
                    if abs(angle)>3:
                        lol =  deskew(lol,angle/2)
                        show(lol,'deskew')
                    cv2.imwrite("crop_"+str(k)+".jpg",warped[breaking_pt1_row[j]:breaking_pt2_row[j],breaking_pt1_col[i]:breaking_pt2_col[i]])
                    k=k+1
            # lol=warped[:,breaking_pt1_col[i]:breaking_pt2_col[i]]
            # show(lol)
            # angle =  compute_skew_angle(lol)
            # if abs(angle)>5 and abs(angle)< 15:
            #     lol =  deskew(lol,angle)
            #     show(lol,'deskew')
            # cv2.imwrite("crop_"+str(k)+".jpg",lol)
            # k=k+1
    return k

def Sharpen(image):
    kernel_sharpen = np.array([[-1,-1,-1,-1,-1],
                                 [-1,2,2,2,-1],
                                 [-1,2,8,2,-1],
                                 [-1,2,2,2,-1],
                                 [-1,-1,-1,-1,-1]])/8.0
    output_3 = cv2.filter2D(image, -1, kernel_sharpen_3)
    return output_3




if __name__ == "__main__":

	'''Capture Serial Input and Image '''

	# init()
	# print 'initilizing Camera'
	# Serial_Cap('/home/rapter/GOR')


	# image = Serial_Cap()
	# Capture_Vlc_Img()
	# image = Camera_Capture()
	''' Read Image .... useful for testing with a static image '''

	image =  cv2.imread('half.jpg')
	ratio = image.shape[0]/500.0
	orig= imutils.resize(image,height = 500)

	# image =  cv2.imread('image_prefix.tiff')

	show(orig)

	os.system('rm image_prefix.tiff')
	os.system('rm crop_*')

	orig = image
	''' Create Initial Boundary  '''

	image = Bounding(image)

	''' PreProcessing '''

	image =  PreProcessing(image)
	# show(image,'PreProcessing')
	''' Bounding Box Coorinates '''

	box=Bounding_Box(image)

	''' Scanned Output '''

	warped = Scanned(orig,box,51,10)
	show(warped,'Scanned')
	angle =  compute_skew_angle(warped)
	# if abs(angle)>3:
	#     warped =  deskew(warped,angle)
	#     show(warped,'deskew')
	'''Erosion operations on otsu image '''
	ret2,th2 = cv2.threshold(warped,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	show(th2,'otsu on deskewed image')
	warped_new=th2
	warped_new = cv2.dilate(warped_new,kernel(N=3,Type=4),iterations=1) 
	show(warped,'dilate')
	warped_new = cv2.erode(warped_new,kernel(N=5,Type=4),iterations=8)
	show(warped_new,'Erosion')
	# warped_new = cv2.erode(warped,kernel(N=5),iterations=10)

	''' Crop '''

	k = Crop(warped_new,warped)
	print k
	''' Speak '''
	if k ==0 :
	    cv2.imwrite('crop_0.jpg',warped)
	    speak("crop_0.jpg")
	for i in range(k):
	    print speak
	    speak("crop_"+str(i)+".jpg")