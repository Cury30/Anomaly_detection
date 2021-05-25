import cv2
from google.colab.patches import cv2_imshow

print("From first video")
cv2_imshow(cv2.imread('MinuteMask/1/1.png', cv2.IMREAD_UNCHANGED))

print("From second video")
cv2_imshow(cv2.imread('MinuteMask/2/1.png', cv2.IMREAD_UNCHANGED))