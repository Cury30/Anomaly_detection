import os 
import sys
import math
from shutil import rmtree
import cv2
from imageai.Detection import ObjectDetection

#VIDEOPATH=os.environ['AICITYVIDEOPATH'] + "/test-data/"

model_path = str(sys.argv[1])

VIDEOPATH = model_path + "/Dataset/"
WEIGHTPATH = model_path + "/yolo.h5"
TXTOUTPATH = model_path + "/Detections/"

def listar_archivos(ruta):
    return [obj.name for obj in os.scandir(ruta) if obj.is_file()]

def renombrar(files, path):
    contador = 0
    for file in files:
        contador+=1
        delimitador = file.find('.')
        nuevo_nombre = path + str(contador) + file[delimitador:]
        os.rename(path + file, nuevo_nombre)

archivos = listar_archivos(VIDEOPATH)
renombrar(archivos,VIDEOPATH)
    

if(os.path.exists(TXTOUTPATH)):
    rmtree(TXTOUTPATH)
else:
    os.mkdir(TXTOUTPATH)

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(WEIGHTPATH)
detector.loadModel()
custom = detector.CustomObjects(car=True, bus=True,truck=True)
video_amount = len(next(os.walk(VIDEOPATH))[2]) + 1
print("Using Input Video Path : "+VIDEOPATH)

for video_num in range(1,video_amount):
    print(VIDEOPATH+str(video_num)+'.mp4')
    cap = cv2.VideoCapture(VIDEOPATH+str(video_num)+'.mp4')
    
    if not cap.isOpened():
        raise IOError("Couldn't open webcam or video")
    framecount=0
    writelist=[]
    while(cap.isOpened()):			
        ret, frame = cap.read()
        framecount+=1
        print(framecount)
        if(not ret):
            break
        ret_img,detections = detector.detectCustomObjectsFromImage( custom_objects=custom, input_type="array",input_image=frame, output_type="array", minimum_percentage_probability=10)
        for eachObject in detections:				
            if eachObject["percentage_probability"]>10.0:
                writelist.append([video_num,framecount,eachObject["box_points"],eachObject["percentage_probability"],eachObject["name"]])
    with open(TXTOUTPATH+str(video_num)+".txt","w") as outtextfile:
        for lines in writelist:
            outtextfile.write(str(lines) + "\n")
    cap.release()
