import cv2
import os
txt_name='trainval.txt'
floder='/home/bz/mx-rcnn-master11.2_only_rotate/data/VOCdevkit/VOC2007/JPEGImages'
for files in os.listdir(floder):
    with open (txt_name,'a+') as f:
        file=files.split('.jpg')[0]
        print (file)
        f.write(file+'\n')
