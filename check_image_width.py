# %%
import numpy as np
import cv2
img = cv2.imread('check_image.jpg')
np.shape(img)
x = 15
img = img[:,0+x:400-x,:]
# %%
cv2.namedWindow('My Image', cv2.WINDOW_NORMAL)

cv2.imshow('My Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# %%
r = 370/4.8
a = 5.66*r
h = 2.83*r
print(h)
# %%
4.8/370
# %%
