import numpy as np
from symdata.bbox import bbox_transform,bbox_overlaps
from symdata.coordinate_convert import back_forward_convert,forward_convert
#from box_utils.cython_utils.cython_bbox import bbox_overlaps



class AnchorGenerator:
    def __init__(self, feat_stride=16, anchor_scales=(8,16,32), anchor_ratios=(0.5,1,2)):
        self._num_anchors = len(anchor_scales) * len(anchor_ratios)
        self._feat_stride = feat_stride
        self._base_anchors = self._generate_base_anchors(feat_stride, np.array(anchor_scales), np.array(anchor_ratios))

    def generate(self, feat_height, feat_width):
        shift_x = np.arange(0, feat_width) * self._feat_stride
        shift_y = np.arange(0, feat_height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = self._num_anchors
        K = shifts.shape[0]
        all_anchors = self._base_anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
        all_anchors = all_anchors.reshape((K * A, 4))
        #print("all_anchors=",all_anchors)
        return all_anchors

    @staticmethod
    def _generate_base_anchors(base_size, scales, ratios):
        """
        Generate anchor (reference) windows by enumerating aspect ratios X
        scales wrt a reference (0, 0, 15, 15) window.
        """
        base_anchor = np.array([1, 1, base_size, base_size]) - 1
        ratio_anchors = AnchorGenerator._ratio_enum(base_anchor, ratios)
        anchors = np.vstack([AnchorGenerator._scale_enum(ratio_anchors[i, :], scales)
                             for i in range(ratio_anchors.shape[0])])
        return anchors

    @staticmethod
    def _whctrs(anchor):
        """
        Return width, height, x center, and y center for an anchor (window).
        """
        w = anchor[2] - anchor[0] + 1
        h = anchor[3] - anchor[1] + 1
        x_ctr = anchor[0] + 0.5 * (w - 1)
        y_ctr = anchor[1] + 0.5 * (h - 1)
        return w, h, x_ctr, y_ctr

    @staticmethod
    def _mkanchors(ws, hs, x_ctr, y_ctr):
        """
        Given a vector of widths (ws) and heights (hs) around a center
        (x_ctr, y_ctr), output a set of anchors (windows).
        """
        ws = ws[:, np.newaxis]
        hs = hs[:, np.newaxis]
        anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                             y_ctr - 0.5 * (hs - 1),
                             x_ctr + 0.5 * (ws - 1),
                             y_ctr + 0.5 * (hs - 1)))
        return anchors

    @staticmethod
    def _ratio_enum(anchor, ratios):
        """
        Enumerate a set of anchors for each aspect ratio wrt an anchor.
        """
        w, h, x_ctr, y_ctr = AnchorGenerator._whctrs(anchor)
        size = w * h
        size_ratios = size / ratios
        ws = np.round(np.sqrt(size_ratios))
        hs = np.round(ws * ratios)
        anchors = AnchorGenerator._mkanchors(ws, hs, x_ctr, y_ctr)#返回（x0,y0,x1,y1)
        return anchors

    @staticmethod
    def _scale_enum(anchor, scales):
        """
        Enumerate a set of anchors for each scale wrt an anchor.
        """
        w, h, x_ctr, y_ctr = AnchorGenerator._whctrs(anchor)
        ws = w * scales
        hs = h * scales
        anchors = AnchorGenerator._mkanchors(ws, hs, x_ctr, y_ctr)
        return anchors


class AnchorSampler:   #分配标签同时剔除不符合标准的anchor=tf中anchor_target和proposal_opr
    def __init__(self, allowed_border=0, batch_rois=256, fg_fraction=0.5, fg_overlap=0.7, bg_overlap=0.3):
        self._allowed_border = allowed_border
        self._num_batch = batch_rois
        self._num_fg = int(batch_rois * fg_fraction)
        self._fg_overlap = fg_overlap
        self._bg_overlap = bg_overlap

    def assign(self, anchors, gt_boxes, im_height, im_width):
        num_anchors = anchors.shape[0]
        #print("gt_boxes=",gt_boxes)
        # filter out padded gt_boxes
        valid_labels = np.where(gt_boxes[:, -1] > 0)[0]  #判断gt_boxes是否是目标
        #print("valid_labels=",valid_labels)
        gt_boxes = gt_boxes[valid_labels]                #得到目标gt_boxes的索引
        #print("gt_boxes2=", gt_boxes)

        # filter out anchors outside the region  #过滤超出边界的anchor
        inds_inside = np.where((anchors[:, 0] >= -self._allowed_border) &
                               (anchors[:, 2] < im_width + self._allowed_border) &
                               (anchors[:, 1] >= -self._allowed_border) &
                               (anchors[:, 3] < im_height + self._allowed_border))[0]
        anchors = anchors[inds_inside, :]
        num_valid = len(inds_inside)
        #print('num_valid=',num_valid)

        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = np.ones((num_valid,), dtype=np.float32) * -1
        bbox_targets = np.zeros((num_valid, 4), dtype=np.float32)
        bbox_weights = np.zeros((num_valid, 4), dtype=np.float32)

        # sample for positive labels
        if gt_boxes.size > 0:
            # overlap between the anchors and the gt boxes
            # overlaps (ex, gt)
            #print('anchor=',anchors)
            #print('anchor_gt_boxes=', gt_boxes)####问题就在这,不带标签

            gt_boxes_rec = np.zeros((gt_boxes.shape[0], 4), dtype=np.float32)


            gt_boxes_rec[:, 0] = np.min(gt_boxes[:, 0:8:2])  # x_min
            gt_boxes_rec[:, 1] = np.min(gt_boxes[:, 1:8:2])  # y_min
            gt_boxes_rec[:, 2] = np.max(gt_boxes[:, 0:8:2])  # x_max
            gt_boxes_rec[:, 3] = np.max(gt_boxes[:, 1:8:2])  # y_max
            overlaps = bbox_overlaps(anchors.astype(np.float),gt_boxes_rec.astype(np.float))

            # fg anchors: anchor with highest overlap for each gt
            gt_max_overlaps = overlaps.max(axis=0)
            #print("gt_max_overlaps=",gt_max_overlaps)
            argmax_inds = np.where(overlaps == gt_max_overlaps)[0]
            labels[argmax_inds] = 1

################################问题就在这里###########################################################
            # fg anchors: anchor with overlap > iou thresh
            max_overlaps = overlaps.max(axis=1)
            #print('max_overlaps=',max_overlaps)
            # bg anchors: anchor with overlap < iou thresh

            labels[max_overlaps < self._bg_overlap] = 0
            #print("labels_0=", labels.shape)
            labels[max_overlaps >= self._fg_overlap] = 1
            #print("labels_1=",labels.shape)


            # sanity check
            fg_inds = np.where(labels == 1)[0]
            #print(" fg_inds=", fg_inds.shape)
            bg_inds = np.where(labels == 0)[0]
            #print("bg_inds=",bg_inds.shape)
            assert len(np.intersect1d(fg_inds, bg_inds)) == 0

            # subsample positive anchors
            cur_fg = len(fg_inds)
            if cur_fg > self._num_fg:
                disable_inds = np.random.choice(fg_inds, size=(cur_fg - self._num_fg), replace=False)
                labels[disable_inds] = -1

            # subsample negative anchors
            cur_bg = len(bg_inds)
            max_neg = self._num_batch - min(self._num_fg, cur_fg)
            #print("max_neg=",max_neg)
            if cur_bg > max_neg:
                disable_inds = np.random.choice(bg_inds, size=(cur_bg - max_neg), replace=False)
                labels[disable_inds] = -1


                # assign to argmax overlap
                fg_inds = np.where(labels == 1)[0]
                #print(" fg_inds=",fg_inds)
                argmax_overlaps = overlaps.argmax(axis=1)
                #print("anchors[fg_inds, :]=",anchors[fg_inds, :])
                #print("gt_boxes[argmax_overlaps[fg_inds], :]=",gt_boxes[argmax_overlaps[fg_inds], :])
                bbox_targets[fg_inds, :] = bbox_transform(anchors[fg_inds, :], gt_boxes_rec[argmax_overlaps[fg_inds], :],
                                                          box_stds=(1.0, 1.0, 1.0, 1.0))
                # only fg anchors has bbox_targets
                bbox_weights[fg_inds, :] = 1
                ###前面和tensorflow都一样，但是缺少_unmap函数
            else:
                # randomly draw bg anchors
                bg_inds = np.random.choice(np.arange(num_valid), size=self._num_batch, replace=False)
                labels[bg_inds] = 0

            all_labels = np.ones((num_anchors,), dtype=np.float32) * -1
            all_labels[inds_inside] = labels
            all_bbox_targets = np.zeros((num_anchors, 4), dtype=np.float32)
            all_bbox_targets[inds_inside, :] = bbox_targets
            all_bbox_weights = np.zeros((num_anchors, 4), dtype=np.float32)
            all_bbox_weights[inds_inside, :] = bbox_weights

            return all_labels, all_bbox_targets, all_bbox_weights