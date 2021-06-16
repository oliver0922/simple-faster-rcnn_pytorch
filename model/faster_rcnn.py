from __future__ import  absolute_import
from __future__ import division
import torch as t
import numpy as np
from utils import array_tool as at
from model.utils.bbox_tools import loc2bbox
from torchvision.ops import nms
# from model.utils.nms import non_maximum_suppression

from torch import nn
from data.dataset import preprocess
from torch.nn import functional as F
from utils.config import opt


def nograd(f):
    def new_f(*args,**kwargs):
        with t.no_grad():
           return f(*args,**kwargs)
    return new_f
# 데코레이터를 통해 함수에서 기본적으로 실행되어야 하는 코드를 만든다.

class FasterRCNN(nn.Module):
    """Base class for Faster R-CNN.

    This is a base class for Faster R-CNN links supporting object detection
    API [#]_. The following three stages constitute Faster R-CNN.

    1. **Feature extraction**: Images are taken and their \
        feature maps are calculated.
    2. **Region Proposal Networks**: Given the feature maps calculated in \
        the previous stage, produce set of RoIs around objects.
    3. **Localization and Classification Heads**: Using feature maps that \
        belong to the proposed RoIs, classify the categories of the objects \
        in the RoIs and improve localizations.

    Each stage is carried out by one of the callable
    :class:`torch.nn.Module` objects :obj:`feature`, :obj:`rpn` and :obj:`head`.

    There are two functions :meth:`predict` and :meth:`__call__` to conduct
    object detection.
    :meth:`predict` takes images and returns bounding boxes that are converted
    to image coordinates. This will be useful for a scenario when
    Faster R-CNN is treated as a black box function, for instance.
    :meth:`__call__` is provided for a scnerario when intermediate outputs
    are needed, for instance, for training and debugging.

    Links that support obejct detection API have method :meth:`predict` with
    the same interface. Please refer to :meth:`predict` for
    further details.

    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        extractor (nn.Module): A module that takes a BCHW image
            array and returns feature maps.
        rpn (nn.Module): A module that has the same interface as
            :class:`model.region_proposal_network.RegionProposalNetwork`.
            Please refer to the documentation found there.
        head (nn.Module): A module that takes
            a BCHW variable, RoIs and batch indices for RoIs. This returns class
            dependent localization paramters and class scores.
        loc_normalize_mean (tuple of four floats): Mean values of
            localization estimates.
        loc_normalize_std (tupler of four floats): Standard deviation
            of localization estimates.

    """

    def __init__(self, extractor, rpn, head,
                loc_normalize_mean = (0., 0., 0., 0.),
                loc_normalize_std = (0.1, 0.1, 0.2, 0.2)
    ): # extractor(특징맵), rpn, roi를 받아 classifier, regressor로 가는 head, localization의 평균과 분산을 받아 생성함수 구성.
        super(FasterRCNN, self).__init__()
        self.extractor = extractor
        self.rpn = rpn
        self.head = head
        # extractor, rpn, head를 각각의 멤버 변수로 가져옴.
       
        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std
        self.use_preset('evaluate')
        # preset을 evaluate로 두어 nms할 때에 hyperparameter를 결정한다.

    @property
    def n_class(self):
        # Total number of classes including the background.
        return self.head.n_class
        # head에서 classifier로 들어가서 class를 구성하는 갯수를 반환.
        
    def forward(self, x, scale=1.):
        """Forward Faster R-CNN.

        Scaling paramter :obj:`scale` is used by RPN to determine the
        threshold to select small objects, which are going to be
        rejected irrespective of their confidence scores.

        Here are notations used.

        * :math:`N` is the number of batch size
        * :math:`R'` is the total number of RoIs produced across batches. \
            Given :math:`R_i` proposed RoIs from the :math:`i` th image, \
            :math:`R' = \\sum _{i=1} ^ N R_i`.
        * :math:`L` is the number of classes excluding the background.

        Classes are ordered by the background, the first class, ..., and
        the :math:`L` th class.

        Args:
            x (autograd.Variable): 4D image variable.
            scale (float): Amount of scaling applied to the raw image
                during preprocessing.

        Returns:
            Variable, Variable, array, array:
            Returns tuple of four values listed below.

            * **roi_cls_locs**: Offsets and scalings for the proposed RoIs. \
                Its shape is :math:`(R', (L + 1) \\times 4)`.
            * **roi_scores**: Class predictions for the proposed RoIs. \
                Its shape is :math:`(R', L + 1)`.
            * **rois**: RoIs proposed by RPN. Its shape is \
                :math:`(R', 4)`.
            * **roi_indices**: Batch indices of RoIs. Its shape is \
                :math:`(R',)`.

        """
        img_size = x.shape[2:]
        # forward를 구성하기 위해 실제 input imaged에 들어갈 x shape의 일부분만 선택.
        h = self.extractor(x) # input image를 입력해서 extractor로 특징맵을 받아온다.
        rpn_locs, rpn_scores, rois, roi_indices, anchor = \
            self.rpn(h, img_size, scale) # Region proposal network에 특징맵과 image size와 scale을 넣어준다.
        roi_cls_locs, roi_scores = self.head(
            h, rois, roi_indices) # rpn에 넣어서 roi를 구했다면 특징맵 h와 같이 roi를 head에 넣어 classifier와 regressor로 전달한다.
        return roi_cls_locs, roi_scores, rois, roi_indices
        # 최종적으로 classification과 localization을 통해 객체가 존재하는 여부에 대한 score와 위치를 얻을 수 있다.
        
    def use_preset(self, preset):
        """Use the given preset during prediction.

        This method changes values of :obj:`self.nms_thresh` and
        :obj:`self.score_thresh`. These values are a threshold value
        used for non maximum suppression and a threshold value
        to discard low confidence proposals in :meth:`predict`,
        respectively.

        If the attributes need to be changed to something
        other than the values provided in the presets, please modify
        them by directly accessing the public attributes.

        Args:
            preset ({'visualize', 'evaluate'): A string to determine the
                preset to use.

        """
        if preset == 'visualize':
            self.nms_thresh = 0.3
            self.score_thresh = 0.7
        elif preset == 'evaluate':
            self.nms_thresh = 0.3
            self.score_thresh = 0.05
        else:
            raise ValueError('preset must be visualize or evaluate')
        # preset으로 str형식의 값을 받아서 원하는 것에 따라서 
        # visualize, evaluate를 위한 non max suppression, score의 threshold를 정한다.
       
    def _suppress(self, raw_cls_bbox, raw_prob):
        bbox = list()
        label = list()
        score = list()
        # bounding box, label, score에 대한 list 생성.
        
        # for문이 0에서 시작하는 것이 아닌 이유는 n_class자체가 background class도 포함되어 있으므로
        # 기존의 class 개수안에서만 for반복문이 돌도록 한다.
        for l in range(1, self.n_class):
            cls_bbox_l = raw_cls_bbox.reshape((-1, self.n_class, 4))[:, l, :]
            # nms이 적용되지 않은 bounding box인 raw_cls_bbox을 가져와서 reshape를 통해 dimension 변경.
            prob_l = raw_prob[:, l] # input으로 들어온 prob의 두번째 행에 있는 확률값만을 가져온다.
            mask = prob_l > self.score_thresh # prob의 요소가 score threshold보다 큰 값을 가지면 1 아니면 0을 가지는 배열 생성.
            cls_bbox_l = cls_bbox_l[mask] # classification된 bounding box중에서 mask에 따른 객체가 있을 확률이 threshold값보다 높은 것만 살린다.
            prob_l = prob_l[mask] # 마찬가지로 prob이 score보다 큰 부분에서만 score를 살리고 나머지는 죽인다.
            keep = nms(cls_bbox_l, prob_l,self.nms_thresh)
            # 추가적으로 non max suppression을 통해 ground-truth bounding box와 가장 큰 roi를 가지는 prediction bounding box와
            # 일정한 threshold이상으로 roi가 구성되면 같은 객체에 대해서 bounding box가 중복적으로 있는 것으로 판단하여
            # 그 bounding box는 없애버리도록 한다.
            
            bbox.append(cls_bbox_l[keep].cpu().numpy())
            # cpu로 numpy형식으로 올려서 bbox list에 nms까지 모두 거친 bounding box를 넣는다.
            label.append((l - 1) * np.ones((len(keep),)))
            # 살아남은 boundingbox의 길이 만큼 lxkeep길이 1짜리 배열로 순차적으로 label list에 넣는다.
            score.append(prob_l[keep].cpu().numpy())
            # bounding box와 마찬가지로 살아남은 bounding box에 해당하는 score들을 numpy형으로 cpu로 올리고 score list에 넣는다.
        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)
        # 각각의 list들을 열방향으로 모두 나열한다.
        
        return bbox, label, score
        
        
    @nograd
    def predict(self, imgs,sizes=None,visualize=False):
        """Detect objects from images.

        This method predicts objects for each image.

        Args:
            imgs (iterable of numpy.ndarray): Arrays holding images.
                All images are in CHW and RGB format
                and the range of their value is :math:`[0, 255]`.

        Returns:
           tuple of lists:
           This method returns a tuple of three lists,
           :obj:`(bboxes, labels, scores)`.

           * **bboxes**: A list of float arrays of shape :math:`(R, 4)`, \
               where :math:`R` is the number of bounding boxes in a image. \
               Each bouding box is organized by \
               :math:`(y_{min}, x_{min}, y_{max}, x_{max})` \
               in the second axis.
           * **labels** : A list of integer arrays of shape :math:`(R,)`. \
               Each value indicates the class of the bounding box. \
               Values are in range :math:`[0, L - 1]`, where :math:`L` is the \
               number of the foreground classes.
           * **scores** : A list of float arrays of shape :math:`(R,)`. \
               Each value indicates how confident the prediction is.

        """
        self.eval()
        # training한 모델을 검증.
        if visualize:
            self.use_preset('visualize') # preset을 visualize를 통해 hyperparameter 결정
            prepared_imgs = list() # img를 넣을 list 생성.
            sizes = list() # size를 넣을 list 생성.
            for img in imgs:
                size = img.shape[1:]
                img = preprocess(at.tonumpy(img))
                prepared_imgs.append(img)
                sizes.append(size)
            # input으로 들어오는 image들의 scale을 계산하고 resize와 normalization을 거친 image로 변환한다.
        else:
             prepared_imgs = imgs
             # visualize가 아니라면 imgs를 그대로 list에 넣는다.
            
        bboxes = list()
        labels = list()
        scores = list()
        for img, size in zip(prepared_imgs, sizes):
            img = at.totensor(img[None]).float()
            scale = img.shape[3] / size[1]
            roi_cls_loc, roi_scores, rois, _ = self(img, scale=scale)
            # We are assuming that batch size is 1.
            roi_score = roi_scores.data
            roi_cls_loc = roi_cls_loc.data
            roi = at.totensor(rois) / scale
            # 전처리된 image들과 size들을 넣어서 img의 scale을 계산하고 forward propogation을 진행하면서
            # roi의 classification, localization 결과를 가져온다.
            
            mean = t.Tensor(self.loc_normalize_mean).cuda(). \
                repeat(self.n_class)[None]
            # class의 개수만큼 localization으로 측정되어 계산된 평균을 복사한다.
            std = t.Tensor(self.loc_normalize_std).cuda(). \
                repeat(self.n_class)[None]
            # class의 개수만큼 localization으로 측정되어 계산된 분산을 복사한다.

            roi_cls_loc = (roi_cls_loc * std + mean) # localization과 classification을 통해 나온 roi를 분산에 곱, 평균과 합을 한다.
            roi_cls_loc = roi_cls_loc.view(-1, self.n_class, 4)
            roi = roi.view(-1, 1, 4).expand_as(roi_cls_loc)
            cls_bbox = loc2bbox(at.tonumpy(roi).reshape((-1, 4)),
                                at.tonumpy(roi_cls_loc).reshape((-1, 4)))
            cls_bbox = at.totensor(cls_bbox)
            cls_bbox = cls_bbox.view(-1, self.n_class * 4)
            # roi를 bounding box를 만드는 작업.
            cls_bbox[:, 0::2] = (cls_bbox[:, 0::2]).clamp(min=0, max=size[0])
            cls_bbox[:, 1::2] = (cls_bbox[:, 1::2]).clamp(min=0, max=size[1])
            # bounding box를 bounded하게 만들기 위해 clamp 사용.
            # clamp를 이용하면 min보다 작은 값은 min max보다 큰 값은 max값으로 변환되므로 모든 값은
            # min과 max값 사이에 있게 된다.

            prob = (F.softmax(at.totensor(roi_score), dim=1))
            # roi score를 softmax function에 적용하면서 0부터 1사이의 확률값이 되도록 한다.
            bbox, label, score = self._suppress(cls_bbox, prob)
            # 이전에 만든 _suppress함수로 nms을 실행하고 살아남은 bounding box, 그 box에 대한 label과 score를 받는다.
            bboxes.append(bbox)
            labels.append(label)
            scores.append(score)

        self.use_preset('evaluate')
        # visulize로 모든 forward를 진행했으면 이 후 과정인 evaluate로 검증.
        self.train()
        return bboxes, labels, scores

    def get_optimizer(self):
        """
        return optimizer, It could be overwriten if you want to specify 
        special optimizer
        """
        lr = opt.lr
        # config파일에 있는 이 코드의 개발자가 미리 설정한 learning rate를 가져온다.
        params = []
        for key, value in dict(self.named_parameters()).items(): # parameter name을 dictionary 형태에서 key와 value로 가져온다.
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': opt.weight_decay}]
                    # 가중치 감쇠 적용.
        if opt.use_adam:
            self.optimizer = t.optim.Adam(params)
        else:
            self.optimizer = t.optim.SGD(params, momentum=0.9)
        # optimizer로 Adam 또는 SGD를 선택해서 사용가능하다.
        return self.optimizer

    def scale_lr(self, decay=0.1):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= decay
        return self.optimizer
    # scale_lr을 호출하면 decay를 통해 learning rate를 감쇠시킬 수 있다.
    # 최적화에 해당하는 위치 근처에서 좀 더 정밀하게 최적점을 찾아갈 수 있도록 한다.




