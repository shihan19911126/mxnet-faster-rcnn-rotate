"""
Proposal Target Operator selects foreground and background roi and assigns label, bbox_transform to them.
"""

import mxnet as mx
import numpy as np

from symdata.bbox import encode_boxes_rotate,bbox_overlaps
from symdata.coordinate_convert import back_forward_convert,forward_convert
#from box_utils.cython_utils.cython_bbox import bbox_overlaps

def sample_rois(rois, gt_boxes, num_classes, rois_per_image, fg_rois_per_image, fg_overlap, box_stds=None):
    """
    generate random sample of ROIs comprising foreground and background examples
    :param rois: [n, 5] (batch_index, x1, y1, x2, y2)
    :param gt_boxes: [n, 9] (x1, y1, x2, y2,x3,y3,x4,y4, cls)
    :param num_classes: number of classes
    :param rois_per_image: total roi number
    :param fg_rois_per_image: foreground roi number
    :param fg_overlap: overlap threshold for fg rois
    :param box_stds: std var of bbox reg
    :return: (labels, rois, bbox_targets, bbox_weights)
    """
    gt_boxes_coodinate_convert = back_forward_convert(gt_boxes, True)  # return [x_c,y_c,w,h,theta,label]
    theta=gt_boxes_coodinate_convert[:,4]
    real_label=gt_boxes_coodinate_convert[:,5]


    gt_boxes_rec_with_label=np.zeros((gt_boxes.shape[0],6),dtype=np.float32)
    gt_boxes_rec_with_label[:,0]=np.min(gt_boxes[:,0:8:2])#x_min
    gt_boxes_rec_with_label[:,1]=np.min(gt_boxes[:,1:8:2])# y_min
    gt_boxes_rec_with_label[:,2]=np.max(gt_boxes[:,0:8:2])#x_max
    gt_boxes_rec_with_label[:,3]=np.max(gt_boxes[:,1:8:2])#y_max

    gt_boxes_rec_with_label[:,4] = theta                 #真实的旋转角度
    gt_boxes_rec_with_label[:,5]=real_label#gt_boxes[:,-1]#真实的标签

    overlaps = bbox_overlaps(rois[:, 1:], gt_boxes_rec_with_label[:, :4])

    #overlaps = bbox_overlaps(
     #   np.ascontiguousarray(rois, dtype=np.float),
      #  np.ascontiguousarray(gt_boxes[:, :-1], dtype=np.float))
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    #print('mx_overlap=',max_overlaps)
    labels = gt_boxes_rec_with_label[gt_assignment, -1]#
    # select foreground RoI with FG_THRESH overlap

    fg_indexes = np.where(max_overlaps >= fg_overlap)[0]
    # guard against the case when an image has fewer than fg_rois_per_image foreground RoIs
    fg_rois_this_image = min(fg_rois_per_image, len(fg_indexes))
    # sample foreground regions without replacement
    if len(fg_indexes) > fg_rois_this_image:
        fg_indexes = np.random.choice(fg_indexes, size=fg_rois_this_image, replace=False)

    # select background RoIs as those within [0, FG_THRESH)
    bg_indexes = np.where(max_overlaps < fg_overlap)[0]
    # compute number of background RoIs to take from this image (guarding against there being fewer than desired)
    bg_rois_this_image = rois_per_image - fg_rois_this_image
    bg_rois_this_image = min(bg_rois_this_image, len(bg_indexes))
    # sample bg rois without replacement
    if len(bg_indexes) > bg_rois_this_image:
        bg_indexes = np.random.choice(bg_indexes, size=bg_rois_this_image, replace=False)

    # indexes selected
    keep_indexes = np.append(fg_indexes, bg_indexes)
    # pad more bg rois to ensure a fixed minibatch size
    while len(keep_indexes) < rois_per_image:
        gap = min(len(bg_indexes), rois_per_image - len(keep_indexes))
        gap_indexes = np.random.choice(range(len(bg_indexes)), size=gap, replace=False)
        keep_indexes = np.append(keep_indexes, bg_indexes[gap_indexes])

    # sample rois and labels
    rois = rois[keep_indexes]
    labels = labels[keep_indexes]
    # set labels of bg rois to be 0
    labels[fg_rois_this_image:] = 0

    targets = encode_boxes_rotate(ex_rois=rois[:, 1:], gt_rois=gt_boxes_rec_with_label[gt_assignment[keep_indexes], :5])
    bbox_targets = np.zeros((rois_per_image, 5 * num_classes), dtype=np.float32)
    bbox_weights = np.zeros((rois_per_image, 5 * num_classes), dtype=np.float32)
    for i in range(fg_rois_this_image):
        cls_ind = int(labels[i])
        bbox_targets[i, cls_ind * 5:(cls_ind + 1) * 5] = targets[i]
        bbox_weights[i, cls_ind * 5:(cls_ind + 1) * 5] = 1
    return rois,labels,bbox_targets,bbox_weights



    """
    gt_boxes_rec=np.zeros((gt_boxes.shape[0],5),dtype=np.float32)
    #print('gt_boxes=',gt_boxes)
    #print('gt_boxes[:,0:8:2]=',gt_boxes[:,0:8:2] )
    #print('max_x=',np.max(gt_boxes[:,0:8:2]))
    #gt_boxes=back_forward_convert(gt_boxes,True)
    #gt_boxes=forward_convert(gt_boxes,False)
    #
    #print("gt_boxes=",gt_boxes)

    gt_boxes_rec[:,0]=np.min(gt_boxes[:,0:8:2])#x_min
    gt_boxes_rec[:,1]=np.min(gt_boxes[:,1:8:2])# y_min
    gt_boxes_rec[:,2]=np.max(gt_boxes[:,0:8:2])#x_max
    gt_boxes_rec[:,3]=np.max(gt_boxes[:,1:8:2])#y_max
    gt_boxes_rec[:,4]=gt_boxes[:,-1]

    overlaps = bbox_overlaps(rois[:, 1:], gt_boxes_rec[:, :4])#######问题也在这里,带标签

    #overlaps = bbox_overlaps(
     #   np.ascontiguousarray(rois, dtype=np.float),
      #  np.ascontiguousarray(gt_boxes[:, :-1], dtype=np.float))
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    #print('mx_overlap=',max_overlaps)
    labels = gt_boxes_rec[gt_assignment, -1]#
    # select foreground RoI with FG_THRESH overlap

    fg_indexes = np.where(max_overlaps >= fg_overlap)[0]
    # guard against the case when an image has fewer than fg_rois_per_image foreground RoIs
    fg_rois_this_image = min(fg_rois_per_image, len(fg_indexes))
    # sample foreground regions without replacement
    if len(fg_indexes) > fg_rois_this_image:
        fg_indexes = np.random.choice(fg_indexes, size=fg_rois_this_image, replace=False)

    # select background RoIs as those within [0, FG_THRESH)
    bg_indexes = np.where(max_overlaps < fg_overlap)[0]
    # compute number of background RoIs to take from this image (guarding against there being fewer than desired)
    bg_rois_this_image = rois_per_image - fg_rois_this_image
    bg_rois_this_image = min(bg_rois_this_image, len(bg_indexes))
    # sample bg rois without replacement
    if len(bg_indexes) > bg_rois_this_image:
        bg_indexes = np.random.choice(bg_indexes, size=bg_rois_this_image, replace=False)

    # indexes selected
    keep_indexes = np.append(fg_indexes, bg_indexes)
    # pad more bg rois to ensure a fixed minibatch size
    while len(keep_indexes) < rois_per_image:
        gap = min(len(bg_indexes), rois_per_image - len(keep_indexes))
        gap_indexes = np.random.choice(range(len(bg_indexes)), size=gap, replace=False)
        keep_indexes = np.append(keep_indexes, bg_indexes[gap_indexes])

    # sample rois and labels
    rois = rois[keep_indexes]
    labels = labels[keep_indexes]
    # set labels of bg rois to be 0
    labels[fg_rois_this_image:] = 0

    targets = encode_boxes_rotate(ex_rois=rois[:, 1:], gt_rois=gt_boxes_rec[gt_assignment[keep_indexes], :5])
    bbox_targets = np.zeros((rois_per_image, 5 * num_classes), dtype=np.float32)
    bbox_weights = np.zeros((rois_per_image, 5 * num_classes), dtype=np.float32)
    for i in range(fg_rois_this_image):
        cls_ind = int(labels[i])
        bbox_targets[i, cls_ind * 5:(cls_ind + 1) * 5] = targets[i]
        bbox_weights[i, cls_ind * 5:(cls_ind + 1) * 5] = 1
    return rois,labels,bbox_targets,bbox_weights
    """

class ProposalTargetOperator(mx.operator.CustomOp):
    def __init__(self, num_classes, batch_images, batch_rois, fg_fraction, fg_overlap, box_stds):
        super(ProposalTargetOperator, self).__init__()
        self._num_classes = num_classes
        self._batch_images = batch_images
        self._batch_rois = batch_rois
        self._rois_per_image = int(batch_rois / batch_images)
        self._fg_rois_per_image = int(round(fg_fraction * self._rois_per_image))
        self._fg_overlap\
            = fg_overlap
        self._box_stds = box_stds

    def forward(self, is_train, req, in_data, out_data, aux):
        assert self._batch_images == in_data[1].shape[0], 'check batch size of gt_boxes'

        all_rois = in_data[0].asnumpy()
        #print("in_data[0]=",in_data[0])
        #all_rois=[0,x0,y0,x1,y1]
        all_gt_boxes = in_data[1].asnumpy()
        #print("in_data[1]=",in_data[1])
        #all_gt_boxes=[x0,y0,x1,y1,x2,y2,x3,y3,1]

        rois = np.empty((0, 5), dtype=np.float32)#5改4
        labels = np.empty((0, ), dtype=np.float32)
        bbox_targets = np.empty((0, 5* self._num_classes), dtype=np.float32)
        bbox_weights = np.empty((0, 5 * self._num_classes), dtype=np.float32)
        for batch_idx in range(self._batch_images):
            b_rois = all_rois[np.where(all_rois[:, 0] == batch_idx)[0]]
            #print("b_rois=",b_rois)

            b_gt_boxes = all_gt_boxes[batch_idx]
            b_gt_boxes = b_gt_boxes[np.where(b_gt_boxes[:, -1] > 0)[0]]
            #print("b_gt_boxes=",b_gt_boxes)
            b_rois, b_labels, b_bbox_targets, b_bbox_weights = sample_rois(b_rois, b_gt_boxes, num_classes=self._num_classes, rois_per_image=self._rois_per_image,
                            fg_rois_per_image=self._fg_rois_per_image, fg_overlap=self._fg_overlap, box_stds=self._box_stds)
            rois = np.vstack((rois, b_rois))
            labels = np.hstack((labels, b_labels))
            bbox_targets = np.vstack((bbox_targets, b_bbox_targets))
            bbox_weights = np.vstack((bbox_weights, b_bbox_weights))

        self.assign(out_data[0], req[0], rois)
        self.assign(out_data[1], req[1], labels)
        self.assign(out_data[2], req[2], bbox_targets)
        self.assign(out_data[3], req[3], bbox_weights)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)

@mx.operator.register('proposal_target')
class ProposalTargetProp(mx.operator.CustomOpProp):
    def __init__(self, num_classes='21', batch_images='1', batch_rois='128', fg_fraction='0.25',
                 fg_overlap='0.5', box_stds='(0.1, 0.1, 0.2, 0.2)'):
        super(ProposalTargetProp, self).__init__(need_top_grad=False)
        self._num_classes = int(num_classes)
        self._batch_images = int(batch_images)
        self._batch_rois = int(batch_rois)
        self._fg_fraction = float(fg_fraction)
        self._fg_overlap = float(fg_overlap)
        self._box_stds = tuple(np.fromstring(box_stds[1:-1], dtype=float, sep=','))

    def list_arguments(self):
        return ['rois', 'gt_boxes']

    def list_outputs(self):
        return ['rois_output', 'label', 'bbox_target', 'bbox_weight']

    def infer_shape(self, in_shape):
        assert self._batch_rois % self._batch_images == 0, 'BATCHIMAGES {} must devide BATCH_ROIS {}'.format(self._batch_images, self._batch_rois)

        rpn_rois_shape = in_shape[0]
        gt_boxes_shape = in_shape[1]

        output_rois_shape = (self._batch_rois, 5)
        label_shape = (self._batch_rois, )
        bbox_target_shape = (self._batch_rois, self._num_classes * 5)
        bbox_weight_shape = (self._batch_rois, self._num_classes * 5)

        return [rpn_rois_shape, gt_boxes_shape], [output_rois_shape, label_shape, bbox_target_shape, bbox_weight_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return ProposalTargetOperator(self._num_classes, self._batch_images, self._batch_rois, self._fg_fraction,
                                      self._fg_overlap, self._box_stds)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []