
"""
def vis_detection(im_orig, detections, class_names, thresh=0.7):
    #visualize [cls, conf, x1, y1, x2, y2]
    import matplotlib.pyplot as plt
    import random

    plt.imshow(im_orig)
    colors = [(random.random(), random.random(), random.random()) for _ in class_names]
    for [cls, conf, x1, y1, x2, y2] in detections:
        cls = int(cls)
        if cls > 0 and conf > thresh:
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 fill=False, edgecolor=colors[cls], linewidth=3.5)
            plt.gca().add_patch(rect)
            plt.gca().text(x1, y1 - 2, '{:s} {:.3f}'.format(class_names[cls], conf),
                           bbox=dict(facecolor=colors[cls], alpha=0.5), fontsize=12, color='white')
    plt.show()
"""
import cv2
import numpy as np
import random

def draw_rotate_box_cv(boxes, labels,thresh=0.95):
    img_orig = '/home/bz/mx-rcnn-master11.2_only_rotate/data/VOCdevkit/VOC2007/JPEGImages/13004.jpg'
    #colors = [random.random(0,255), random.random(0,255), random.random(0,255)]
    img_orig=cv2.imread(img_orig)
    for [cls, conf, x_c, y_c, w, h,theta] in boxes:
        cls = int(cls)
        if cls > 0 and conf > thresh:
            #rect=((x_c/1.5625,y_c/1.5625),(w/1.5625,h/1.5625),theta)
            rect = ((x_c / 1.5625, y_c / 1.5625), (w / 1.5625, h / 1.5625), theta)
            #rect=cv2.minAreaRect(rect)
            rect=cv2.boxPoints(rect)#/1.25  #这个系数需要随图像改变
            rect = np.int0(rect)

            cv2.drawContours(img_orig,[rect],-1,(0,255,0))
            #print(rect)
        cv2.imwrite('13004.jpg',img_orig)

        #cv2.waitKey(100000)

"""
    for i, box in enumerate(boxes):
        x_c, y_c, w, h, theta = box[0], box[1], box[2], box[3], box[4]

        label = labels[i]
        if label != 0:
            print("y_center=", y_c , "x_center=", x_c , 'h=',h , 'w=', w, 'thata=', theta)
            num_of_object += 1
            # color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))
            color = (0, 255, 0)
            rect = ((x_c, y_c), (w, h), theta)
            rect = cv2.boxPoints(rect)
            rect = np.int0(rect)
            cv2.drawContours(img, [rect], -1, color, 2)

            category = LABEl_NAME_MAP[label]

            if scores is not None:
                cv2.rectangle(img,
                               pt1=(x_c, y_c),
                               pt2=(x_c + 120, y_c + 15),
                               color=color,
                               thickness=-1)
                cv2.putText(img,
                             text=category+": "+str(scores[i]),
                             org=(x_c, y_c+10),
                             fontFace=1,
                             fontScale=1,
                             thickness=2,

                             color=(color[1], color[2], color[0]))
            else:
                 cv2.rectangle(img,
                               pt1=(x_c, y_c),
                               pt2=(x_c + 40, y_c + 15),
                               color=color,
                               thickness=-1)
                 cv2.putText(img,
                             text=category,

                             org=(x_c, y_c + 10),
                             fontFace=1,
                             fontScale=1,
                             thickness=2,
                             color=(color[1], color[2], color[0]))


    cv2.putText(img,
                text=str(num_of_object),
                org=((img.shape[1]) // 2, (img.shape[0]) // 2),
                fontFace=3,
                fontScale=1,
                color=(255, 0, 0))
    cv2.imwrite('1',img)

    #return img
"""