import torch
from model_utils import calculate_iou, create_anchor_space, calculate_anchors, transform_parameters
import model
import torch.nn as nn
import torch.nn.functional as F

TOTAL_ANCHORS = 256

def balance_anchors(positive_anchors, negative_anchors):
    # We randomly disable some positive and some negative anchors
    p_samples = (positive_anchors == 1).nonzero()
    n_samples = (negative_anchors == 1).nonzero()
    pos_count = p_samples.size(0)
    neg_count = n_samples.size(0)
    negative_anchors = negative_anchors.fill_(0)
    # Set those indices to 0
    non_p_samples = p_samples[torch.randperm(pos_count)[(TOTAL_ANCHORS // 2):]]
    n_samples = n_samples[torch.randperm(neg_count)[:(TOTAL_ANCHORS - (min(TOTAL_ANCHORS // 2, pos_count)))]]
    # Note: there is a place for optimization
    for elem in non_p_samples:
        positive_anchors[elem[0], elem[1], elem[2], elem[3]] = 0
    for elem in n_samples:
        negative_anchors[elem[0], elem[1], elem[2], elem[3]] = 1
    return (positive_anchors, negative_anchors, )




class RPNLoss(nn.Module):
    def __init__(self, cache_anchor_space = False, lambda_multiplicator = 10):
        super(RPNLoss, self).__init__()
        self.lambda_multiplicator = lambda_multiplicator
        self.cache_anchor_space = cache_anchor_space
        # TODO: implement anchor space caching

    def forward(self, input, target):
        feature_size = input.size()[2:4]
        size = target["size"]
        batch_size = input.size()[0]
        input = input.view(batch_size * len(model.anchors), -1, feature_size[0], feature_size[1])

        anchor_space = create_anchor_space(size, feature_size, model.anchors)
        anchor_space_size = feature_size[0] * feature_size[1] * len(model.anchors)

        iou = calculate_iou(anchor_space.view(1, len(model.anchors), feature_size[0], feature_size[1], 4).repeat(batch_size, 1, 1, 1, 1), target["boxes"])
        max_iou, max_iou_indices = torch.max(iou, -1)

        # Calculates the cls loss
        positive_labels, negative_labels = calculate_anchors(size, target["boxes"], anchor_space, iou, max_iou, ignore_out_of_range=True)        

        # Balance anchors
        positive_labels, negative_labels = balance_anchors(positive_labels, negative_labels)
        positive_labels = positive_labels.view(-1, feature_size[0], feature_size[1])
        negative_labels = negative_labels.view(-1, feature_size[0], feature_size[1])

        cls_loss_prep = F.cross_entropy(input[:,0:2,:,:], positive_labels.type(torch.LongTensor), reduce=False)
        cls_loss_prep *= (positive_labels | negative_labels).type(torch.FloatTensor)
        cls_loss = torch.sum(cls_loss_prep)
        
        # Calculate the regression loss
        max_gt_box = target["boxes"].view(batch_size, -1,1,4,1,1).repeat(1,1,len(model.anchors), 1, feature_size[0], feature_size[1]).gather(1, 
            max_iou_indices.view(batch_size, 1, len(model.anchors), 1, feature_size[0], feature_size[1]).repeat(1,1,1,4,1,1)) \
            .view(batch_size, len(model.anchors), 4, feature_size[0], feature_size[1])
        max_gt_box = max_gt_box.view(batch_size * len(model.anchors), -1, feature_size[0], feature_size[1])

        tstar = transform_parameters(max_gt_box, anchor_space)
        t = transform_parameters(input[:,2:,:,:], anchor_space)
        reg_loss_prep = F.smooth_l1_loss(t, tstar, reduce = False)
        reg_loss_prep *= positive_labels.view(len(model.anchors), feature_size[0], feature_size[1], 1).type(torch.FloatTensor)
        reg_loss = torch.sum(reg_loss_prep)

        ncls = batch_size * TOTAL_ANCHORS
        nreg = anchor_space_size #number of anchor locations
        loss = ncls * cls_loss + nreg * reg_loss * self.lambda_multiplicator
        return loss