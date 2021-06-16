from __future__ import  absolute_import
import torch as t
from torch import nn
from torchvision.models import vgg16
from torchvision.ops import RoIPool

from model.region_proposal_network import RegionProposalNetwork
from model.faster_rcnn import FasterRCNN
from utils import array_tool as at
from utils.config import opt


def decom_vgg16():
    if opt.caffe_pretrain:
        model = vgg16(pretrained=False)
        if not opt.load_path:
            model.load_state_dict(t.load(opt.caffe_pretrain_path))
            # pretrain 된 모델이 있으면 가져와서 대체한다.
    else:
        model = vgg16(not opt.load_path)
        # 없다면 vgg16 architecture를 가져온다.
    features = list(model.features)[:30] # 특징맵의 30개까지만 가져온다.
    classifier = model.classifier

    classifier = list(classifier)
    del classifier[6]
    if not opt.use_drop:
        del classifier[5]
        del classifier[2]
    # dropout 기법을 사용하지 않는 다면 classifier
    classifier = nn.Sequential(*classifier)

    # freeze top4 conv
    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad = False

    return nn.Sequential(*features), classifier


class FasterRCNNVGG16(FasterRCNN):
    """Faster R-CNN based on VGG-16.
    For descriptions on the interface of this model, please refer to
    :class:`model.faster_rcnn.FasterRCNN`.

    Args:
        n_fg_class (int): The number of classes excluding the background.
        ratios (list of floats): This is ratios of width to height of
            the anchors.
        anchor_scales (list of numbers): This is areas of anchors.
            Those areas will be the product of the square of an element in
            :obj:`anchor_scales` and the original area of the reference
            window.

    """

    feat_stride = 16  # vgg16구조를 거쳐서 나온 특징맵을 downsampling하기 위해 feat stride를 16을 설정.

    def __init__(self,
                 n_fg_class=20,
                 ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32]
                 ): # background class가 포함되어있지 않은 class 개수와 anchor box를 만들 때 쓰일 scale과 ratio를 넣어 생성함수 호출.
                 
        extractor, classifier = decom_vgg16()
        # 특징맵과 classification을 도출하게 하는 모듈을 decom_vgg16을 통해 가져온다.

        rpn = RegionProposalNetwork(
            512, 512,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride,
        )
        # input channel 개수와 output channel 개수를 512로 설정.
        # ration, scale, stride를 input으로 넣어서 anchor box를 만들도록 한다. -> 이 후 region proposals 생성.

        head = VGG16RoIHead(
            n_class=n_fg_class + 1,
            roi_size=7,
            spatial_scale=(1. / self.feat_stride),
            classifier=classifier
        )
        # Roi를 받아 마지막으로 classifier와 regressor로 들어가도록 하는 VGG16RoIHead.
        
        super(FasterRCNNVGG16, self).__init__(
            extractor,
            rpn,
            head,
        )
        # Faster RCNN모듈을 상속받아와서 특징맵과 rpn, head를 넣어 생성함수 호출.

class VGG16RoIHead(nn.Module):
    """Faster R-CNN Head for VGG-16 based implementation.
    This class is used as a head for Faster R-CNN.
    This outputs class-wise localizations and classification based on feature
    maps in the given RoIs.
    
    Args:
        n_class (int): The number of classes possibly including the background.
        roi_size (int): Height and width of the feature maps after RoI-pooling.
        spatial_scale (float): Scale of the roi is resized.
        classifier (nn.Module): Two layer Linear ported from vgg16

    """

    def __init__(self, n_class, roi_size, spatial_scale,
                 classifier):
        # 여기서의 n_class는 background class를 포함한 총 class 개수이다.
        super(VGG16RoIHead, self).__init__()

        self.classifier = classifier
        self.cls_loc = nn.Linear(4096, n_class * 4) 
        # 총 class갯수에 anchor box의 좌표 x,y, 박스의 너비와 높이 w,h를 위한(x4 부분) 공간할당 후 Fullyconnected로 계산.
        self.score = nn.Linear(4096, n_class)
        # 객체가 있는지 없는지에 대한 여부만 지표로 삼기 때문에 n개의 class만 있으면 됨.

        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)
        # 각 classification, locatlization의 초기값을 평균이 0, 분산이 0.001로 초기화.
        # score에 대한 부분도 마찬가지로 초기값을 평균이 0, 분산이 0.001로 초기화.
        
        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.roi = RoIPool( (self.roi_size, self.roi_size),self.spatial_scale)
        # 마지막으로 RoiPooling을 통해 고정적인 결과 dimension이 나오도록 한다.
        
    def forward(self, x, rois, roi_indices):
        """Forward the chain.

        We assume that there are :math:`N` batches.

        Args:
            x (Variable): 4D image variable.
            rois (Tensor): A bounding box array containing coordinates of
                proposal boxes.  This is a concatenation of bounding box
                arrays from multiple images in the batch.
                Its shape is :math:`(R', 4)`. Given :math:`R_i` proposed
                RoIs from the :math:`i` th image,
                :math:`R' = \\sum _{i=1} ^ N R_i`.
            roi_indices (Tensor): An array containing indices of images to
                which bounding boxes correspond to. Its shape is :math:`(R',)`.

        """
       
        roi_indices = at.totensor(roi_indices).float() # roi와 이미지와 대응되는 index들을 float으로 캐스팅, tensor형태로 변환.
        rois = at.totensor(rois).float() # 실제 roi들도 마찬가지로 float으로 캐스팅 후 tensor형태로 변환.
        indices_and_rois = t.cat([roi_indices[:, None], rois], dim=1) # roi_indices와 rois를 tensor형태로 묶는다. concatenate과 비슷한 함수.
        
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]] # 첫번째 요소는 그대로 두고 (2,3), (4,5) index에 해당하는 요소를 switch.
        indices_and_rois =  xy_indices_and_rois.contiguous() # 0개씩 연속적으로 묶으면서 tensor를 만듬. 0개씩이므로 묶지 않고 개별적으로 tensor내부에 []가 있다.

        pool = self.roi(x, indices_and_rois) # input 이미지에 roi pooling을 적용시킴으로써 roi를 이미지에 투영시킨다.
        pool = pool.view(pool.size(0), -1) # 투영시킨 이미지 dimension 변경.
        fc7 = self.classifier(pool) # 투영시킨 이미지를 classifier에 대입.
        roi_cls_locs = self.cls_loc(fc7)  # fully connected layer를 통해 localization 연산.
        roi_scores = self.score(fc7) # fully connected layer를 통해 classification 연산.
        return roi_cls_locs, roi_scores


def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
    
    # 가중치 w와 bias를 입력으로 들어오는 평균과 분산으로 초기화시키는 함수.
    
