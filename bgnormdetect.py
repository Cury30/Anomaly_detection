from imageai.Detection import ObjectDetection
import os
import cv2
import sys


model_path = str(sys.argv[1])
dataset_path = model_path + '/Dataset'
inpath = model_path + "/MinuteMask/"
outpath = model_path + "/BGDetections/"
execution_path = os.getcwd()
video_amount = len(next(os.walk(dataset_path))[2]) + 1

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath( os.path.join(execution_path , "./yolo.h5"))
detector.loadModel()
custom = detector.CustomObjects(car=True, bus=True,truck=True)

def listar_archivos(ruta):
    return [obj.name for obj in os.scandir(ruta) if obj.is_file()]

def renombrar(files, path):
    contador = 0
    for file in files:
        contador+=1
        delimitador = file.find('.')
        nuevo_nombre = path + "\\" + str(contador) + file[delimitador:]
        os.rename(path + "\\" + file, nuevo_nombre)

archivos = listar_archivos(dataset_path)
renombrar(archivos,dataset_path)


for i in range(1,video_amount):
    Len = len(os.listdir(os.path.join(inpath,str(i))))
    if(not os.path.exists(outpath+str(i))): 
        os.mkdir(outpath+str(i))
    texfile=open(outpath+str(i)+"/out.txt","w")
    for q in range(Len):            
        detections = detector.detectCustomObjectsFromImage( custom_objects=custom,input_image=os.path.join(execution_path , inpath+str(i)+"/"+str(q+1)+'.png'), output_image_path=os.path.join(execution_path , outpath+str(i)+"/"+str(q+1)+'.png'), minimum_percentage_probability=10)
        for eachObject in detections:					
            if eachObject["percentage_probability"]>10.0:
                texfile.write(str(i)+"," +str(q)+", "+str(eachObject["percentage_probability"])+","+str(eachObject["box_points"])+","+str(eachObject["name"] )+ "\n")
