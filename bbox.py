# -*- coding:utf-8 -*-
import numpy as np
import math
import cv2
from symdata.rotate_polygon_nms import rotate_gpu_nms
import time
np.set_printoptions(suppress=True)
def bbox_overlaps(boxes, query_boxes):
    """
    determine overlaps between boxes and query_boxes
    :param boxes: n * 4 bounding boxes
    :param query_boxes: k * 4 bounding boxes
    :return: overlaps: n * k overlaps
    """
    n_ = boxes.shape[0]
    #print("n_=",n_)
    k_ = query_boxes.shape[0]
    #print("k_=",k_)
    overlaps = np.zeros((n_, k_), dtype=np.float)
    for k in range(k_):
        query_box_area = (query_boxes[k, 2] - query_boxes[k, 0] + 1) * (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        for n in range(n_):
            iw = min(boxes[n, 2], query_boxes[k, 2]) - max(boxes[n, 0], query_boxes[k, 0]) + 1 #判断anchor的bbox和gt_bbox的任意右下角的x坐标要大于左上角的x坐标
            if iw > 0:
                ih = min(boxes[n, 3], query_boxes[k, 3]) - max(boxes[n, 1], query_boxes[k, 1]) + 1 #判断右下角的y坐标要大于左上角的y坐标
                if ih > 0:
                    box_area = (boxes[n, 2] - boxes[n, 0] + 1) * (boxes[n, 3] - boxes[n, 1] + 1)
                    all_area = float(box_area + query_box_area - iw * ih)
                    #print('iw*ih=', iw * ih)
                    #print('box_area =', box_area)
                    #print(' query_box_area=', query_box_area)
                    overlaps[n, k] = iw * ih / all_area

    #print("overlaps=",overlaps)
    return overlaps

def bbox_transform(ex_rois, gt_rois, box_stds):#对应tf中encode
#计算与anchor有最大IOU的GT的偏移量
#ex_rois：表示anchor；gt_rois：表示GT
    """
    compute bounding box regression targets from ex_rois to gt_rois
    :param ex_rois: [N, 4]
    :param gt_rois: [N, 4]
    :return: [N, 4]
    """
    assert ex_rois.shape[0] == gt_rois.shape[0], 'inconsistent rois number'

    #ex_rois=(x1,y1,x2,y2),#得到anchor的（x,y,w,h）
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * (ex_widths - 1.0)
    ex_ctr_y = ex_rois[:, 1] + 0.5 * (ex_heights - 1.0)
    # gt_rois=(x1,x2,x3,x4),得到GT的（x,y,w,h）
    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    #print('gt_widths=',gt_widths)
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * (gt_widths - 1.0)
    gt_ctr_y = gt_rois[:, 1] + 0.5 *(gt_heights - 1.0)
 #按照损失函数中的计算公式，计算，得到对应的偏移量
    targets_dx = (gt_ctr_x - ex_ctr_x) / (ex_widths + 1e-14) / box_stds[0]
    targets_dy = (gt_ctr_y - ex_ctr_y) / (ex_heights + 1e-14) / box_stds[1]
    targets_dw = np.log(gt_widths / ex_widths) / box_stds[2]
    targets_dh = np.log(gt_heights / ex_heights) / box_stds[3]
    targets = np.vstack((targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
    return targets

def encode_boxes_rotate( ex_rois,gt_rois, box_stds=[0.1,0.1,0.2,0.2,0.2]):
    '''
    :param unencode_boxes: [batch_size*H*W*num_anchors_per_location, 5]
    :param reference_boxes: [H*W*num_anchors_per_location, 5]
    :return: encode_boxes [-1, 5]
    '''
    #print('gt_rois=',gt_rois.shape)
    #x_center, y_center, w, h, theta = gt_rois[:, 0], gt_rois[:, 1], gt_rois[:, 2], gt_rois[:, 3], gt_rois[:, 4]
    x_min=gt_rois[:,0]
    y_min=gt_rois[:,1]
    x_max=gt_rois[:,2]
    y_max=gt_rois[:,3]
   ##########################角度问题？？？
    theta=gt_rois[:,4]


    w=x_max-x_min+1
    h=y_max-y_min+1
    x_center=x_min+0.5*(w-1)
    y_center=y_min+0.5*(h-1)

    #reference_xmin, reference_ymin, reference_xmax, reference_ymax = ex_rois[:, 0], ex_rois[:, 1], ex_rois[:, 2], ex_rois[:, 3]
    #reference_x_center = (reference_xmin + reference_xmax) / 2.
    #reference_y_center = (reference_ymin + reference_ymax) / 2.
    # here maybe have logical error, reference_w and reference_h should exchange,
    # but it doesn't seem to affect the result.
    #reference_w = reference_xmax - reference_xmin
    #reference_h=reference_ymax-reference_ymin

    reference_w = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    reference_h = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    reference_x_center = ex_rois[:, 0] + 0.5 * (reference_w - 1.0)
    reference_y_center = ex_rois[:, 1] + 0.5 * (reference_h - 1.0)
    #print("reference_w",reference_w[:50])
    #reference_h = reference_ymax - reference_ymin
    reference_theta= np.ones(ex_rois[:,0].shape) * -90

    targets_dx = (x_center - reference_x_center) / (reference_w + 1e-14) / box_stds[0]
    targets_dy = (y_center - reference_y_center) / (reference_h+ 1e-14) / box_stds[1]
    targets_dw = np.log(w / reference_w) / box_stds[2]
    targets_dh = np.log(h / reference_h) / box_stds[3]
    t_theta = (theta - reference_theta) * math.pi / 180/box_stds[4]

    targets = np.vstack((targets_dx, targets_dy, targets_dw, targets_dh,t_theta)).transpose()
    return targets

#根据anchor和偏移量计算proposals
def bbox_pred_rotate(boxes, box_deltas, box_stds=[0.1,0.1,0.2,0.2,0.2]):#对应tf中decode_boxes_rotate
    """
    Transform the set of class-agnostic boxes into class-specific boxes
    by applying the predicted offsets (box_deltas)
    :param boxes: !important [N 5]
    :param box_deltas: [N, 5* num_classes]
    :return: [N 5 * num_classes]
    """
    #print('boxes=',boxes)
    if boxes.shape[0] == 0:
        return np.zeros((0, box_deltas.shape[1]))
#anchor到bouning boxes regression
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    #print("widths=",widths)
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    #print("height=",heights)
    ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
    #print("ctr_x=",ctr_x)
    #print("ctr_x=",ctr_x)
    ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)
    #print("ctr_y=",ctr_y)
    reference_theta = np.ones((boxes[:,0].shape)) * -90
    #reference_theta=boxes[:,5]
#得到（x,y,w,h）方向上的偏移量
    dx = box_deltas[:, 0::5] * box_stds[0]#########问题关键点
    dy = box_deltas[:, 1::5] * box_stds[1]
    dw = box_deltas[:, 2::5] * box_stds[2]
    dh = box_deltas[:, 3::5] * box_stds[3]
    d_theta=box_deltas[:,4::5]*box_stds[4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis] #np.newaxis,表示将widths增加一维，使得其能够相加
    #print(" pred_ctr_x=", pred_ctr_x)
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]
    predict_theta=d_theta*180/math.pi+reference_theta[:,np.newaxis]#+90

    pred_boxes = np.zeros(box_deltas.shape)
    # x_c
    pred_boxes[:, 0::5] = pred_ctr_x
    # y_c
    pred_boxes[:, 1::5] = pred_ctr_y

    # w
    pred_boxes[:, 2::5] = pred_w
    #h
    pred_boxes[:, 3::5] = pred_h
    #theta
    pred_boxes[:, 4::5]=predict_theta
    return pred_boxes

def nms_rotate_gpu(boxes_list, scores, iou_threshold, use_angle_condition=False, angle_gap_threshold=0, device_id=0):
    if use_angle_condition:
        x_c=boxes_list[:,0]
        y_c=boxes_list[:,1]
        w=boxes_list[:,2]
        h=boxes_list[:,3]
        theta=boxes_list[:,4]

        boxes_list = np.transpose(np.stack([x_c, y_c, w, h, theta]))
        #det_tensor = tf.concat([boxes_list, tf.expand_dims(scores, axis=1)], axis=1)
        det_tensor=np.concatenate([boxes_list, np.expand_dims(scores, axis=1)], axis=1)
        keep=rotate_gpu_nms(det_tensor, iou_threshold, device_id)
        return keep
    else:
        x_c = boxes_list[:, 0]
        y_c = boxes_list[:, 1]
        w = boxes_list[:, 2]
        h = boxes_list[:, 3]
        theta = boxes_list[:, 4]
        boxes_list = np.transpose(np.stack([x_c, y_c, w, h, theta]))
        det_tensor = np.concatenate([boxes_list, np.expand_dims(scores, axis=1)], axis=1)
        keep = rotate_gpu_nms(det_tensor, iou_threshold, device_id)
        keep = np.reshape(keep, [-1])
        return keep


def nms_rotate_cpu(det, iou_threshold):
    """
       :param boxes: format [x_c, y_c, w, h, theta,scores]
       :param threshold: iou threshold (0.7 or 0.5)
       :param max_output_size: max number of output
       :return: the remaining index of boxes
       """
    keep = []
    x_c = det[:, 0]
    y_c= det[:, 1]
    w = det[:, 2]
    h = det[:, 3]
    theta=det[:,4]
    scores = det[:, -1]
    order = scores.argsort()[::-1]
    #print("order=",order.shape)
    num = det.shape[0]

    suppressed = np.zeros((num), dtype=np.int)

    for _i in range(num):
        if len(keep) >= 150:
            break
        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        r1 = ((det[i, 0], det[i, 1]), (det[i, 2], det[i, 3]), det[i, 4])
        area_r1 = det[i, 2] * det[i, 3]
        for _j in range(_i + 1, num):
            #print("j_=",_j)
            j = order[_j]
            if suppressed[i] == 1:
                continue
            r2 = ((det[j, 0], det[j, 1]), (det[j, 2], det[j, 3]), det[j, 4])
            area_r2 = det[j, 2] * det[j, 3]
            inter = 0.0

            int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
            if int_pts is not None:
                order_pts = cv2.convexHull(int_pts, returnPoints=True)

                int_area = cv2.contourArea(order_pts)

                inter = int_area * 1.0 / (area_r1 + area_r2 - int_area + 0.00001)

            #if inter >= iou_threshold:
            #    suppressed[j] = 1
            inds=np.where(inter<=iou_threshold)[0]
            order=order[inds+1]

    return keep

#在test时用到预测的函数
def im_detect(rois, scores, bbox_deltas, im_info,
              bbox_stds, nms_thresh, conf_thresh):
    """rois (nroi, 4), scores (nrois, nclasses), bbox_deltas (nrois, 5* nclasses), im_info (3)"""
    rois = rois.asnumpy()

    scores = scores.asnumpy()
    bbox_deltas = bbox_deltas.asnumpy()
    #print("bbox_deltas=",bbox_deltas.shape)

    im_info = im_info.asnumpy()
    height, width, scale = im_info
    # post processing
    #pred_boxes = bbox_pred(rois, bbox_deltas, bbox_stds)
    #print("rois=",rois)
    pred_boxes=bbox_pred_rotate(rois,bbox_deltas,bbox_stds) #得到预测的（x_c,y_c,w,h,theta)
    # we used
    #  scaled image & roi to train, so it is necessary to transform them back
    #print("pred_boxes=",pred_boxes[:,0:4])

    pred_boxes[:,0:4] = pred_boxes[:,0:4] / scale
    #print("pred_boxes[:,0:4]=",pred_boxes[:,0:4])
    # convert to per class detection results
    det = []
    for j in range(1, scores.shape[-1]):
        indexes = np.where(scores[:, j] > conf_thresh)[0]
        cls_scores = scores[indexes, j, np.newaxis]
        #print("pred_boxes.shape=",pred_boxes.shape)
        cls_boxes = pred_boxes[indexes, j * 5:(j + 1) * 5]
        #print("cls_boxes=",cls_boxes.astype(np.int32))
        cls_dets = np.hstack((cls_boxes, cls_scores)).astype(np.float32)
        #keep=nms(cls_dets,0.7)
        keep = rotate_gpu_nms(cls_dets,nms_thresh)

        cls_id = np.ones_like(cls_scores) * j
        det.append(np.hstack((cls_id, cls_scores, cls_boxes))[keep, :])

    # assemble all classes
    det = np.concatenate(det, axis=0)
    #det=det/scale
    return det
