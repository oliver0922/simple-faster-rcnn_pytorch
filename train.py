from __future__ import  absolute_import
import os

import ipdb
import matplotlib
from tqdm import tqdm

from utils.config import opt
from data.dataset import Dataset, TestDataset, inverse_normalize
from model import FasterRCNNVGG16
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc

# fix for ulimit
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE) # 생성가능한 프로세스의 갯수를 가져온다.
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1])) # 이 프로세스에서 생성가능한 프로세스 개수를 변경.

matplotlib.use('agg') # agg 랜더링 기반 사용.


def eval(dataloader, faster_rcnn, test_num=10000): # model 검증.
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    ## 각각의 ground-truth bounding box와 prediction bounding box, scores를 list형식으로 선언.
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader)):
        # dataset에서 groundtruth bounding box와 img에 대한 정보를 가져온다.
        # 이때 tqdm으로 진행되는 부분을 시각적으로 볼 수 있게 한다.
        sizes = [sizes[0][0].item(), sizes[1][0].item()] # size에 있는 key와 value를 가져온다.
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        # faster_cnn 객체를 통해 train을 시킨 모델에 img, size를 넣어 예측값을 얻는다.
        gt_bboxes += list(gt_bboxes_.numpy()) 
        # tensor형태의 gt_bboxes_를 numpy형태의 array로 변환 후 gt_bboxes list에 넣는다.
        # gt_bboxes_와 gt_bboxes 다른 변수이다.
        gt_labels += list(gt_labels_.numpy())
        # tensor형태의 gt_labels_ numpy형태의 array로 변환 후 gt_labels list에 넣는다.
        gt_difficults += list(gt_difficults_.numpy())
        # tensor형태의 gt_difficults_ numpy형태의 array로 변환 후 gt_difficults list에 넣는다.
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        # 예측되는 값이 나올 때마다 각각의 list에 넣어준다.
        if ii == test_num: break # ii가 test num에 도달하면 반복문을 멈춘다.

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    # eval_detection_voc를 호출함으로써, calc_detection_voc_prec_rec함수와 calc_detection_voc_ap함수를 호출.
    # precision과 recall -> calc_detection_voc_prec_rec함수로 계산.
    # mAP -> calc_detection_voc_ap함수로 계산.
    # use_07_metric은 pascal voc 2007 dataset의 평가지표를 사용할 것인지에 관한 boolean 값이다.
    return result


def train(**kwargs):
    opt._parse(kwargs)
    # kwargs를 통해서 argument parse로 들어왔던 인자를 전달받음.

    dataset = Dataset(opt)
    print('load data')
    dataloader = data_.DataLoader(dataset, \
                                  batch_size=1, \
                                  shuffle=True, \
                                  # pin_memory=True,
                                  num_workers=opt.num_workers)
    # data를 불러오는 작업.
    # hyperparameter설정 -> batch_size, shuffle 유무, num_workers로 cpu나 gpu에서 몇개의 코어를 사용할 지 결정.
    testset = TestDataset(opt)
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False, \
                                       pin_memory=True
                                       )
    # test dataset load 후 hyperparameter 설정.
    # hyperparameter설정 -> batch_size, shuffle 유무, pin_memory로 batch들을 생성하는 dataloader를 pinned memory에 위치시킨다.
    
    faster_rcnn = FasterRCNNVGG16()
    # simple-faster-rcnn-pytorch/model/faster_rcnn_vgg16.py의 class를 받아온다.
    # FasterRCNNVGG16를 선언하면서 생성자함수에서 나온 rpn, head를 더불어 extractor 등을 부모 class인 FasterRCNN class에 넣는다.
    # 그렇게 되면 forward propogation이 진행되면서 predict 값을 얻을 수 있다.
    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    # FasterRCNN을 학습시키기 위해 faster_rcnn 구조를 FasterRCNNTrainer class에 넣어 GPU에 올리고 선언.
    if opt.load_path:
        trainer.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)
    # pretrained model을 받아오려고 하는 경우에 실행되는 문구.
    trainer.vis.text(dataset.db.label_names, win='labels')
    # visdom을 통해 training 도중의 결과를 얻을 수 있도록 한다.
    best_map = 0
    lr_ = opt.lr
    # learning rate -> 기존에 opt에서 사용자가 설정한 값으로 설정.
    for epoch in range(opt.epoch):
        trainer.reset_meters()
        # 평균 loss에 대한 key, value를 가져와서 reset하고 roi, rpn에 대한 클래스개수 만큼 confusion meter를 적용해서 matrix를 만듬.
        for ii, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):
            scale = at.scalar(scale) # scale이 nummpy의 array형식이라면 reshape를 통해 dimension을 변경하고 tensor라면 key,value를 받아옴.
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            # img, bbox, label을 GPU에 올린다.
            trainer.train_step(img, bbox, label, scale)
            # trainer의 train_step함수를 통해 training.

            if (ii + 1) % opt.plot_every == 0:
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()
                # plot_every는 사용자설정값 40으로 설정되어있는데, training 40번마다 debug_file을 보고할 내용이 있으면
                # set_trace를 통해 training중에 사용자가 설정을 변경하거나 다른 실행을 하도록 할 수 있게 한다.
             
                trainer.vis.plot_many(trainer.get_meter_data())
                # visdom을 이용해 trainer가 training되는 동안 loss를 plot하도록 한다.
                
                ori_img_ = inverse_normalize(at.tonumpy(img[0]))
                # ground-truth image를 가져온다.
                gt_img = visdom_bbox(ori_img_,
                                     at.tonumpy(bbox_[0]),
                                     at.tonumpy(label_[0]))
                trainer.vis.img('gt_img', gt_img)
                # visdom으로 나타내기 위해 visdom_bbox 함수에 image와 bbox, label을 numpy형태로 넣어
                # visdom으로 나타낼 수 있는 형태의 image를 얻는다.
                # 결과적으로 visdom으로 image가 나올땐 ground-truth값의 boundingbox가 image에 투영되어 
                # 시각적으로 나타내게 된다.
                
                _bboxes, _labels, _scores = trainer.faster_rcnn.predict([ori_img_], visualize=True)
                # 이번엔 trainer의 faster_rcnn.predict를 통해 기존 이미지를 넣어 예측값을 받아온다. 
                pred_img = visdom_bbox(ori_img_,
                                       at.tonumpy(_bboxes[0]),
                                       at.tonumpy(_labels[0]).reshape(-1),
                                       at.tonumpy(_scores[0]))
                # ground-truth때와 마찬가지로 각 prediction boundingbox, label, score를 visdom에 넣어서
                # boundingbox와 score, label이 기존 이미지에 나타난 이미지를 얻게 된다.
                trainer.vis.img('pred_img', pred_img)
                
                trainer.vis.text(str(trainer.rpn_cm.value().tolist()), win='rpn_cm')
                # confusion meter를 통해 rpn의 confusion matrix를 얻는다.
               
                trainer.vis.img('roi_cm', at.totensor(trainer.roi_cm.conf, False).float())
                # confusion meter를 통해 roi의 confusion matrix를 얻는다.
        eval_result = eval(test_dataloader, faster_rcnn, test_num=opt.test_num)
        # test dataset과 faster_rcnn 모델을 통해 검증.
        trainer.vis.plot('test_map', eval_result['map'])
        # trainer에서 mAP를 계산하여 visdom에서 시각화.
        lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']
        # trainer에서 learning rate를 변경.
        log_info = 'lr:{}, map:{},loss:{}'.format(str(lr_),
                                                  str(eval_result['map']),
                                                  str(trainer.get_meter_data()))
        trainer.vis.log(log_info)
        # visdom으로 learning rate, 검증결과, confusion matrix에 대한 정보를 시각화한다.
        
        if eval_result['map'] > best_map:
            best_map = eval_result['map']
            best_path = trainer.save(best_map=best_map)
        # 기존의 best_map는 0이었는데 계속해서 높은 map로 갱신해가면서 가장 높은 map를 찾아 저장.
        if epoch == 9:
            trainer.load(best_path)
            trainer.faster_rcnn.scale_lr(opt.lr_decay)
            lr_ = lr_ * opt.lr_decay
        # best_map일 때의 best_path를 load해서 learning rate를 감쇠하여 좀 더 세밀하게 optimization 지점을 찾도록 한다.

        if epoch == 13: 
            break
        # epoch이 13일 때 중지.

if __name__ == '__main__':
    import fire

    fire.Fire()
    # fire의 fire함수를 통해 모든 함수를 command line interface로 만들어주며, 만든 현재의 모듈을 모두 실행한다.
    # 즉 위의 함수를 모두 실행.
