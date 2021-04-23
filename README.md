# AICity-2020-CETCVLAB
Code for the Submission to AI City Challenge Track 4 by Computer Vision LAB, College of Engineering, Trivandrum which obtained third position in leaderboard.
The leader board statistics for the proposed method are :

* F1-Score : 0.7018
* RMSE : 67.5044
* S4 Score : 0.5438

## Requirements

* Python 3
* [ImageAI](https://github.com/OlafenwaMoses/ImageAI#installation)
* TensorFlow 1.x
* SciPy
* MatPlotlib
* Opencv-Contrib

## Pre-Trained Weights

The detection model YOLOv3 uses the pretrained weights provided here : [OlafenwaMoses/ImageAI](https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/yolo.h5) , the weight file is expected to be placed in the same working directory as the code.

## Reproducing the Results without Running the detections

We have provided the detections text files both in the Original Video, Background Images, and the Detections on the Zoomed(Cropped Video) under the releases, please download the [Release](https://github.com/cetcvlab/AICity-2020-CETCVLAB/releases/download/v1.0/AICity2020.zip) and unzip to your workspace and run the anomaly extractor only to extract the anomalies from the detection text files, the anomaly extractor module extracts the anomaly time from the videos using the detection text files, the anomaly extractor can be tried after [downloading and unzipping the release](https://github.com/cetcvlab/AICity-2020-CETCVLAB/releases/download/v1.0/AICity2020.zip) by running :

```
python3 CombinedExtractor.py
```

## Reproducing the Results after running the detections

Export the Environment variable ```AICITYVIDEOPATH``` to the input video directory (test and train) 
```
export AICITYVIDEOPATH="/path/to/track4/dataset/"
```

### Step 1 Running the detections on Original Video and Background Generation (to save time both of these can be run in parallel)

```
python3 normdetect.py
python3 createbg.py
```

### Step 2 Run the detections on the averaged background frames

```
python3 bgnormdetect.py
```

### Step 3 Run the normal extractor to identify which all videos to run the crop detections

This will partially fill the results and provide a text file zoomcheck.txt which gives the video numbers for the videos to check for zoom/cropped detections.

```
python3 CombinedExtractor.py normal
```

### Step 4 Run the Crop Detector on the videos and the background samples (can be run in parallel)

```
python3 cropdetect.py
python3 bgcropdetect.py
```

### Step 3 Run the zoom extractor to get the final results

This will fill up the rest of the Result.txt by running the anomaly extraction algorithm on the detection text files

```
python3 CombinedExtractor.py zoom
```

Result.txt will contain the results in the AI City Challenge Track 4 format

## References

* Background Generation : Using same parameters as [2019-CVPR-AIC-Track-3-UWIPL](https://github.com/ipl-uw/2019-CVPR-AIC-Track-3-UWIPL/blob/master/candidate%20selection/bgfg%20generate%20v2.py)
* Freeze detection on videos : [2019-CVPR-AIC-Track-3-UWIPL](https://github.com/ipl-uw/2019-CVPR-AIC-Track-3-UWIPL/blob/master/time%20decision/get_freeze_period.py)