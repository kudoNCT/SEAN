import cv2
import numpy as np




#empty.fill(255)

label_org = cv2.imread("../data_set/CelebA_HQ/CelebA-HQ-binarymask/test/labels/29258.png")

label = label_org.copy()
label[np.where(label==1)] = 255
#cv2.imshow('label',label)
label_gray = cv2.cvtColor(label,cv2.COLOR_BGR2GRAY)
a, thresh = cv2.threshold(label_gray,60, 255, cv2.THRESH_BINARY)
contours, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
empty = np.zeros(label.shape,dtype = np.uint8)
label_unchange = label_org.copy()
cv2.drawContours(image=label_org, contours=contours, contourIdx=-1, color=(1, 1, 1), thickness=5,lineType=cv2.LINE_AA)
cv2.imwrite('new_label_29258.png',label_org)
compare = label_unchange == label_org
equal_arr = compare.all()
print(f'equal_arr:{equal_arr}')
#cv2.imshow('empty',empty)


cv2.waitKey(0)
cv2.destroyAllWindows()