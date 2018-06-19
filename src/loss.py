import torch
import model_utils
import model

import torch.nn.functional as F


class RPNLoss(torch.Module):
    def __init__(self, cache_anchor_space = False, lambda_multiplicator = 10):
        super(RPNLoss, self).__init__()
        self.lambda_multiplicator = lambda_multiplicator
        self.cache_anchor_space = cache_anchor_space
        # TODO: implement anchor space caching

    def forward(self, input, target):
        feature_size = input.size()[1:2]
        size = input.size()[1:2]
        batch_size = input.size()[0]

        anchor_space = model_utils.create_anchor_space(size, feature_size, model.anchors)
        anchor_space_size = feature_size[0] * feature_size[1] * len(model.anchors)

        iou = calculate_iou(anchor_space, target)
        max_iou, max_iou_indices = torch.max(iou, 3)

        # Calculates the cls loss
        positive_labels, label_mask = model_utils.calculate_anchor_detection_mask(size, target["boxes"], anchor_space, iou, max_iou, ignore_out_of_range=True)        cls_loss_prep = F.cross_entropy(input[:,:,:,0:1], positive_labels, reduce=False)
        cls_loss_prep *= label_mask
        cls_loss = torch.sum(cls_loss_prep)
        
        # Calculate the regression loss
        max_gt_box = target.view(1,1,1,-1,4).gather(2, max_iou_indices)
        t = transform_parameters(max_gt_box, anchor_space)
        tstar = transform_parameters(input[:,:,:,2:], anchor_space)
        reg_loss_prep = F.smooth_l1_loss(t, tstar, reduce = False)
        reg_loss_prep *= positive_labels
        reg_loss = torch.sum(reg_loss_prep)

        ncls = batch_size 
        nreg = anchor_space_size #number of anchor locations
        loss = ncls * cls_loss + nreg * reg_loss * self.lambda_multiplicator
        return loss