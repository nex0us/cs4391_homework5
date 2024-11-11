"""
CS 4391 Homework 5 Programming
Implement the compute_loss() function in this python script
"""
import os
import torch
import torch.nn as nn


# compute Intersection over Union (IoU) of two bounding boxes
# the input bounding boxes are in (cx, cy, w, h) format
def compute_iou(pred, gt):
    x1p = pred[0] - pred[2] * 0.5
    x2p = pred[0] + pred[2] * 0.5
    y1p = pred[1] - pred[3] * 0.5
    y2p = pred[1] + pred[3] * 0.5
    areap = (x2p - x1p + 1) * (y2p - y1p + 1)    
    
    x1g = gt[0] - gt[2] * 0.5
    x2g = gt[0] + gt[2] * 0.5
    y1g = gt[1] - gt[3] * 0.5
    y2g = gt[1] + gt[3] * 0.5
    areag = (x2g - x1g + 1) * (y2g - y1g + 1)

    xx1 = max(x1p, x1g)
    yy1 = max(y1p, y1g)
    xx2 = min(x2p, x2g)
    yy2 = min(y2p, y2g)

    w = max(0.0, xx2 - xx1 + 1)
    h = max(0.0, yy2 - yy1 + 1)
    inter = w * h
    iou = inter / (areap + areag - inter)    
    return iou

# TODO: finish the implementation of this loss function for YOLO training
# output: (batch_size, num_boxes * 5 + num_classes, 7, 7), raw output from the network
# pred_box: (batch_size, num_boxes * 5 + num_classes, 7, 7), predicted bounding boxes from the network (see the forward() function)
# gt_box: (batch_size, 5, 7, 7), ground truth bounding box target from the dataloader
# gt_mask: (batch_size, 7, 7), ground truth bounding box mask from the dataloader
# num_boxes: number of bounding boxes per cell
# num_classes: number of object classes for detection
# grid_size: YOLO grid size, 64 in our case
# image_size: YOLO image size, 448 in our case
def compute_loss(output, pred_box, gt_box, gt_mask, num_boxes, num_classes, grid_size, image_size):
    batch_size = output.shape[0]
    num_grids = output.shape[2]
    
    # initialize tensors for the mask and confidence
    box_mask = torch.zeros(batch_size, num_boxes, num_grids, num_grids)
    box_confidence = torch.zeros(batch_size, num_boxes, num_grids, num_grids)

    # compute assignment of predicted bounding boxes for ground truth bounding boxes
    for i in range(batch_size):
        for j in range(num_grids):
            for k in range(num_grids):
 
                # if the ground truth mask is 1
                if gt_mask[i, j, k] > 0:
                    # transform ground truth box
                    gt = gt_box[i, :, j, k].clone()
                    gt[0] = gt[0] * grid_size + k * grid_size
                    gt[1] = gt[1] * grid_size + j * grid_size
                    gt[2] = gt[2] * image_size
                    gt[3] = gt[3] * image_size

                    select = 0
                    max_iou = -1
                    # select the box with maximum IoU
                    for b in range(num_boxes):
                        pred = pred_box[i, 5*b:5*b+4, j, k].clone()
                        iou = compute_iou(gt, pred)
                        if iou > max_iou:
                            max_iou = iou
                            select = b
                    box_mask[i, select, j, k] = 1
                    box_confidence[i, select, j, k] = max_iou

    # compute YOLO loss components
    weight_coord = 5.0
    weight_noobj = 0.5

    # loss for coordinates (x, y, w, h)
    loss_x = torch.sum(box_mask * torch.pow(box_confidence - output[:, 0:1*num_boxes:5], 2.0))
    loss_y = torch.sum(box_mask * torch.pow(box_confidence - output[:, 1:2*num_boxes:5], 2.0))
    loss_w = torch.sum(box_mask * torch.pow(box_confidence - output[:, 2:3*num_boxes:5], 2.0))
    loss_h = torch.sum(box_mask * torch.pow(box_confidence - output[:, 3:4*num_boxes:5], 2.0))
    
    # loss for confidence (object or non-object)
    loss_obj = torch.sum(box_mask * torch.pow(box_confidence - output[:, 4:5*num_boxes:5], 2.0))
    
    # loss for non-object confidence (weight_noobj)
    loss_noobj = torch.sum((1 - box_mask) * torch.pow(box_confidence - output[:, 4:5*num_boxes:5], 2.0)) * weight_noobj

    # loss for class predictions
    class_pred = output[:, 5:5+num_classes, :, :]  # Slice the class probabilities
    loss_cls = torch.sum(gt_mask * torch.pow(gt_box[:, 4:, :, :] - class_pred, 2.0))

    # the total loss
    loss = loss_x + loss_y + loss_w + loss_h + loss_obj + loss_noobj + loss_cls

    return loss
