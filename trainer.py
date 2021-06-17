from __future__ import  absolute_import
import os
from collections import namedtuple
import time
from torch.nn import functional as F
from model.utils.creator_tool import AnchorTargetCreator, ProposalTargetCreator

from torch import nn
import torch as t
from utils import array_tool as at
from utils.vis_tool import Visualizer

from utils.config import opt
from torchnet.meter import ConfusionMeter, AverageValueMeter

LossTuple = namedtuple('LossTuple',
                       ['rpn_loc_loss',
                        'rpn_cls_loss',
                        'roi_loc_loss',
                        'roi_cls_loss',
                        'total_loss'
                        ])
# namedtuple을 이용해 dictionary와 같은 형태로 key 값들을 선언한다.

# nn.Module 상속하는 FasterRCNNTRainer 클래스 = 학습과정을 wrapping하며 losses를 리턴한다.
class FasterRCNNTrainer(nn.Module):
    

    The losses include:

    * :obj:`rpn_loc_loss`: RPN bounding box의 regression loss(bounding box의 위치에 관한 손실함수)
    * :obj:`rpn_cls_loss`: RPN classification loss(물체의 존재 유무).
    * :obj:`roi_loc_loss`: head module(ROI)의 classification loss(class probability).
    * :obj:`roi_cls_loss`: head module(ROI)의 regression loss.
    * :obj:`total_loss`: The sum of 4 loss above.

    Args:
        faster_rcnn (model.FasterRCNN):
            A Faster R-CNN model that is going to be trained.
   

    def __init__(self, faster_rcnn):
        super(FasterRCNNTrainer, self).__init__()
        # pytorch의 nn.Module을 상속받아 생성자 호출.
        self.faster_rcnn = faster_rcnn
        self.rpn_sigma = opt.rpn_sigma
        self.roi_sigma = opt.roi_sigma
        # L1 smooth loss를 위해서 hyperparameter sigma 설정.
        # target creator create gt_bbox gt_label etc as training targets. 
        self.anchor_target_creator = AnchorTargetCreator() #Anchor box를  ground truth 박스 주변에 생성한다.
        self.proposal_target_creator = ProposalTargetCreator() #ground truth box를 roi에 할당한다.

        self.loc_normalize_mean = faster_rcnn.loc_normalize_mean
        self.loc_normalize_std = faster_rcnn.loc_normalize_std
        #faster_rcnn 객체의 get_optimizer 메쏘드를 호출,learning rate, weight_decay 설정, optimizer 설정
        self.optimizer = self.faster_rcnn.get_optimizer() 
        # visdom wrapper
        self.vis = Visualizer(env=opt.env)

        # indicators for training status
        self.rpn_cm = ConfusionMeter(2) #물체의 존재 유무 2개에관한 confusion matrix 객체 생성
        self.roi_cm = ConfusionMeter(21)#물체의 종류 20개+1(background)에관한 confusion matrix 객체 생성
        self.meters = {k: AverageValueMeter() for k in LossTuple._fields}
        # visdom을 위해 confusionmeter를 사용해 confusion matrix를 만든다.
        # AverageValueMeter를 이용해 loss에 대한 평균을 구한다. (내부적으로 분산도 구함.)
    def forward(self, imgs, bboxes, labels, scale):
        """ Faster- RCNN forward propagation 수행하고 loss 값을 계산한다

        * :math:`N` 배치사이즈 크기를 의미
        * :math:`R` 이미지 당 bounding box의 갯수


        Args:
            imgs (~torch.autograd.Variable): 배치사이즈 만큼의 이미지
            bboxes (~torch.autograd.Variable): 배치의 bounding , shape :math:`(N, R, 4)`.
            labels (~torch.autograd..Variable): 배치의 라벨, shape=(N,R), background는 제외한다
            scale (float): 전처리 동안 raw image에 적용되는 scale

        Returns:
            namedtuple of 5 losses #namedtuple의 5개 손실함수를 반환
        """
        n = bboxes.shape[0] # bounding box의 batch size를 가져온다.
        if n != 1:
            raise ValueError('Currently only batch size 1 is supported.')
        # batch size가 1이 아니면 에러가 발생되도록 한다.
        
        _, _, H, W = imgs.shape
        img_size = (H, W)
        
        features = self.faster_rcnn.extractor(imgs) #imgs의 특징을 추출한 특징맵을 생성한다
        
        #rpn에 특징맵, image 크기, scale 값을 집어넣어서 rpn의 loss값, score 값, roi 객체, index, anchor박스를 얻는다. 등을 얻는다.
        rpn_locs, rpn_scores, rois, roi_indices, anchor = \
            self.faster_rcnn.rpn(features, img_size, scale)

        # 배치사이즈가 1이기 때문에 0으로 인덱싱
        bbox = bboxes[0]
        label = labels[0]
        rpn_score = rpn_scores[0]
        rpn_loc = rpn_locs[0]
        roi = rois

        
        #roi 내부에서 forward propagation을 진행
        # 
        sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
            roi,
            at.tonumpy(bbox),
            at.tonumpy(label),
            self.loc_normalize_mean,
            self.loc_normalize_std) # ground truth box를 sampled 된 proposal에 할당한다. 
        sample_roi_index = t.zeros(len(sample_roi)) #batchsize 크기가 1이므로 인덱스 0을 할당하여 sample_roi_index를 얻어낸다.
        roi_cls_loc, roi_score = self.faster_rcnn.head(
            features,
            sample_roi,
            sample_roi_index)

        # ------------------ RPN losses 계산 -------------------#
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
            at.tonumpy(bbox),
            anchor,
            img_size)
        gt_rpn_label = at.totensor(gt_rpn_label).long() #ground truth label, ground truth location 값을 tensor 형태로 변환
        gt_rpn_loc = at.totensor(gt_rpn_loc)
        rpn_loc_loss = _fast_rcnn_loc_loss(
            rpn_loc,
            gt_rpn_loc,
            gt_rpn_label.data,
            self.rpn_sigma) #rpn localization loss 계산

        rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label.cuda(), ignore_index=-1) #cross entropy방식 사용하여 rpn classification 계산
        _gt_rpn_label = gt_rpn_label[gt_rpn_label > -1]
        _rpn_score = at.tonumpy(rpn_score)[at.tonumpy(gt_rpn_label) > -1]
        self.rpn_cm.add(at.totensor(_rpn_score, False), _gt_rpn_label.data.long())

        # ------------------ ROI losses 계산 -------------------#
        n_sample = roi_cls_loc.shape[0]
        roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
        roi_loc = roi_cls_loc[t.arange(0, n_sample).long().cuda(), \
                              at.totensor(gt_roi_label).long()]
        gt_roi_label = at.totensor(gt_roi_label).long()
        gt_roi_loc = at.totensor(gt_roi_loc)
        # ground-truth와 비교 위해 gt_roi_label과 loc 정리.
        roi_loc_loss = _fast_rcnn_loc_loss(
            roi_loc.contiguous(),
            gt_roi_loc,
            gt_roi_label.data,
            self.roi_sigma)

        roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label.cuda())
        # roi에 대한 loss 계산.
        # crossentropy loss 함수 적용 하여, roi score와 roi ground truth를 넣는다.

        self.roi_cm.add(at.totensor(roi_score, False), gt_roi_label.data.long())

        losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]
        losses = losses + [sum(losses)] # 모든 loss더해서 최종 loss 구하기.

        return LossTuple(*losses) # loss tuple 형식 반환.

    def train_step(self, imgs, bboxes, labels, scale):
        self.optimizer.zero_grad() # gradient zero 로 초기화.
        losses = self.forward(imgs, bboxes, labels, scale) # foward propogation 진행. / loss 반환
        losses.total_loss.backward() # loss로 backward propogation 진행.
        self.optimizer.step() # gradient로 optimization 진행
        self.update_meters(losses)  # weight, bias 업데이트.
        return losses

    def save(self, save_optimizer=False, save_path=None, **kwargs):
        """serialize models include optimizer and other info
        return path where the model-file is stored.

        Args:
            save_optimizer (bool): whether save optimizer.state_dict().
            save_path (string): where to save model, if it's None, save_path
                is generate using time str and info from kwargs.
        
        Returns:
            save_path(str): the path to save models.
        """
        save_dict = dict() # 여러 정보담을 dictionary 생성.

        save_dict['model'] = self.faster_rcnn.state_dict() # model의 architecture 저장.
        save_dict['config'] = opt._state_dict() # config file, 즉 hyperparameter 저장.
        save_dict['other_info'] = kwargs 
        save_dict['vis_info'] = self.vis.state_dict() # 상태에 관한 정보들이 담긴 dictionary를 visdom으로 올리는 작업 저장.

        if save_optimizer:
            save_dict['optimizer'] = self.optimizer.state_dict()
            # optimization 저장.

        if save_path is None:
            timestr = time.strftime('%m%d%H%M')
            save_path = 'checkpoints/fasterrcnn_%s' % timestr
            for k_, v_ in kwargs.items():
                save_path += '_%s' % v_

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            # checkpoint를 저장할 경로를 만들고 그 디렉토리에 저장.
            # 디렉토리가 없으면 만들고 있으면 만들지 않음.

        t.save(save_dict, save_path)
        self.vis.save([self.vis.env])
        # visdom 정보도 저장.
        return save_path

    def load(self, path, load_optimizer=True, parse_opt=False, ):
        state_dict = t.load(path)
        if 'model' in state_dict:
            self.faster_rcnn.load_state_dict(state_dict['model'])
            # 모델 architecture 불러오기.
        else: 
            self.faster_rcnn.load_state_dict(state_dict)
            return self
        if parse_opt:
            opt._parse(state_dict['config'])
            # hyperparameter 가져오기.
        if 'optimizer' in state_dict and load_optimizer:
            self.optimizer.load_state_dict(state_dict['optimizer'])
            # optimizer 정보 가져오기.
        return self

    def update_meters(self, losses):
        loss_d = {k: at.scalar(v) for k, v in losses._asdict().items()}
        for key, meter in self.meters.items():
            meter.add(loss_d[key])
            # parameter 업데이트를 위한 함수.
            # loss를 통해 계산한 gradient로 업데이트.

    def reset_meters(self):
        for key, meter in self.meters.items():
            meter.reset()
        self.roi_cm.reset()
        self.rpn_cm.reset()
        # roi, rpn의 confusion matrix 초기화.
        # confusion matrix를 초기화함으로써 mAP를 구하는 matrix가 초기화되는 것.
    def get_meter_data(self):
        return {k: v.value()[0] for k, v in self.meters.items()}
        

def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()
    # Legularization 1 Loss 사용.


def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    in_weight = t.zeros(gt_loc.shape).cuda()
    # Localization의 loss는 positive roi로만 계산됨.
    
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
    # 예측값 localization과 ground-truth localization을 L1 loss적용해서 loss 계산. / sigma parameter 적용.
    
    loc_loss /= ((gt_label >= 0).sum().float())
    return loc_loss
