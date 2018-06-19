import torch

def make_absolute_boxes(size, features):
    pass




def get_filter_in_range(size, anchor_space):
    width, height = size
    filter_mask = anchor_space[:,:,:,2].gt(width) | anchor_space[:,:,:,3].gt(height) | anchor_space[:,:,:,0].lt(0) \
        | anchor_space[:,:,:,1].lt(0)
    return ~filter_mask

def create_anchor_space(size, feature_size, anchors, use_cache = False):
    # TODO: add cache of anchor space for same image sizes

    wfeature, hfeature = feature_size
    width, height = size
    # Generate anchors first
    anchors_view = torch.Tensor(anchors)
    anchors_view = torch.cat((anchors_view * -0.5, anchors_view * 0.5,), 1).view(1,1,-1,4)

    xspace = torch.linspace(width / wfeature / 2, width - width / wfeature / 2, wfeature).repeat(hfeature, 1)
    yspace = torch.linspace(height / hfeature / 2, height - height / hfeature/ 2, hfeature).view(-1,1).repeat(1, wfeature)
    
    origins = torch.cat((xspace.view(hfeature, wfeature, 1),yspace.view(hfeature, wfeature, 1),),-1) \
        .repeat(1,1,2) \
        .view(hfeature, wfeature, 1, 4)

    anchor_space = anchors_view + origins
    return anchor_space

def calculate_iou(anchor_space, gtboxes):
    height, width, anchors, _ = anchor_space.shape
    anchor_space = anchor_space.view(height, width, anchors, 1, 4)
    gtboxes = gtboxes.view(1, 1, 1, -1, 4)

    #calculate intersection
    mind = torch.min(anchor_space, gtboxes)
    maxd = torch.max(anchor_space, gtboxes)
    intersection = torch.clamp(mind[:,:,:,:,2] - maxd[:,:,:,:,0], min = 0) * torch.clamp(mind[:,:,:,:,3] - maxd[:,:,:,:,1], min = 0)
    areas = (anchor_space[:,:,:,:,2] - anchor_space[:,:,:,:,0]) * (anchor_space[:,:,:,:,3] - anchor_space[:,:,:,:,1]) + \
        (gtboxes[:,:,:,:,2] - gtboxes[:,:,:,:,0]) * (gtboxes[:,:,:,:,3] - gtboxes[:,:,:,:,1])

    return intersection / (areas - intersection)

'''
Takes detected boxes and gtboxes at specific location and calculates for which boxes the loss function applies
'''
def calculate_anchor_detection_mask(size, gtboxes, anchor_space, iou, max_iou, ignore_out_of_range = True, **kwargs):
    gtboxescount = gtboxes.shape[0]

    # Not we will remove everything and keep
    # (i) anchor with the highest IoU with a GT
    # (ii) anchor with IoU greater equal 0.7
    positive_anchors = max_iou.gt(0.7) # (ii)

    # (ii)
    positive_anchors_indices = torch.argmax(iou.view(-1, gtboxescount), 0)
    idx = []
    for adim in list(iou.size())[:-1]:
        idx.append(positive_anchors_indices % adim)
        positive_anchors_indices = positive_anchors_indices / adim

    idx = torch.cat(idx).view(3,-1).transpose(0,1)
    
    # Note: there is a place for optimization
    for elem in idx:
        positive_anchors[elem[0], elem[1], elem[2]] = 1

    # Now, we calculate the negative samples
    # Its iou with any GT box is < 0.3
    negative_anchors = positive_anchors & max_iou.lt(0.3)

    # For training we ignore anchors that are not fully visible
    if ignore_out_of_range:
        anchor_filter = get_filter_in_range(size, anchor_space)
        positive_anchors &= anchor_filter
        negative_anchors &= anchor_filter

    feature_mask = positive_anchors | negative_anchors    
    return (positive_anchors, feature_mask)


def transform_parameters(box, anchor_space):
    wa = anchor_space[:,:,:,2] - anchor_space[:,:,:,0]
    ha = anchor_space[:,:,:,3] - anchor_space[:,:,:,1]
    xa = anchor_space[:,:,:,0] + wa / 2
    ya = anchor_space[:,:,:,1] + ha / 2

    w = box[:,:,:,2] - box[:,:,:,0]
    h = box[:,:,:,3] - box[:,:,:,1]
    x = box[:,:,:,0] + w / 2
    y = box[:,:,:,1] + h / 2

    tx = (x - xa) / wa
    ty = (y - ya) / ha
    tw = torch.log(w/wa)
    th = torch.log(h/ha)
    return torch.cat((tx,ty,tw,th,), 3)
