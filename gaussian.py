import cv2
import matplotlib.pyplot as plt
from PIL import Image

img = cv2.imread('data_with_seg/train/RP/176.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
blur = cv2.GaussianBlur(img,(31, 31),0)
plt.imshow(blur)
plt.show()
blur = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)
