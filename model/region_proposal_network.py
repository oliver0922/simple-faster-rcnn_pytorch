import numpy as np
from torch.nn import functional as F
import torch as t
from torch import nn

## numpy, pytorch에서 필요한 모듈을 import한다.

from model.utils.bbox_tools import generate_anchor_base
from model.utils.creator_tool import ProposalCreator

## Rpn 구성에 필요한 bounding box tool, creator tool에서 각각의 함수를 import한다.


class RegionProposalNetwork(nn.Module):
    """Region Proposal Network introduced in Faster R-CNN.

    This is Region Proposal Network introduced in Faster R-CNN [#]_.
    This takes features extracted from images and propose
    class agnostic bounding boxes around "objects".

    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        in_channels (int): The channel size of input.
        mid_channels (int): The channel size of the intermediate tensor.
        ratios (list of floats): This is ratios of width to height of
            the anchors.
        anchor_scales (list of numbers): This is areas of anchors.
            Those areas will be the product of the square of an element in
            :obj:`anchor_scales` and the original area of the reference
            window.
        feat_stride (int): Stride size after extracting features from an
            image.
        initialW (callable): Initial weight value. If :obj:`None` then this
            function uses Gaussian distribution scaled by 0.1 to
            initialize weight.
            May also be a callable that takes an array and edits its values.
        proposal_creator_params (dict): Key valued paramters for
            :class:`model.utils.creator_tools.ProposalCreator`.

    .. seealso::
        :class:`~model.utils.creator_tools.ProposalCreator`

    """

    def __init__(
            self, in_channels=512, mid_channels=512, ratios=[0.5, 1, 2],
            anchor_scales=[8, 16, 32], feat_stride=16,
            proposal_creator_params=dict(),
    ):  # 생성함수 정의 : in, output channel, anchor의 ratio, scale, feat stride 정의
        # feat stride는 CNN이 한번 실행된 특징맵에서 RPN으로 이전될 때의 stride이다.
        super(RegionProposalNetwork, self).__init__()
        self.anchor_base = generate_anchor_base(
            anchor_scales=anchor_scales, ratios=ratios)  
        ## model/utils/bbox_tools.py 안에 있는 generate_anchor_base라는 함수를 이용해 
        ## anchor scale 과 ratio를 이용해 anchor box를 만드는데 이 때 반환하는 값은
        ## anchor box의 위, 아래, 오른쪽, 왼쪽 각 꼭짓점의 좌표를 가진 9x4 배열이다.
        self.feat_stride = feat_stride
        self.proposal_layer = ProposalCreator(self, **proposal_creator_params)
        ## RPN에서의 Region Proposal을 찾기 위해 model/utils/creator_tool.py의 ProposalCreator class를 이용.
        ## * Proposal Creator class의 각 key, value값들을 가져오기 위해 딕셔너리로 선언한 proposal_creator_params를 넣어준다.
        ## Non-max suppression을 통해 threshold보다 작은 anchor box는 제거하면서 최종적으로 roi가 나오게 된다.
        n_anchor = self.anchor_base.shape[0] # anchor box의 개수
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1) 
        # 특징맵에서 intermediate 층으로 가기 위한 연산
        # Convolution 연산, inputchannel, outputchannel 값 대입, stride = 3, padding = 1, dilation = 1
        # 여기서 dilation은 convoultion 연산할 때의 input에서 돌아다니는 window를 변형한 것인데 
        # 잘 이해하기 위해서는 visualization으로 dilation이 무엇을 하는지 알아봐야함.
        # dilation 참고 : https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
        self.score = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)
        # classifier로 들어가기 위한 연산
        # intermediate 층을 convolution 연산 진행.
        # 객체의 유무를 판단을 위한 지표로 각 box마다 2개가 필요하므로 anchor box의 개수에 곱하기 2를 해준다.
        # stride = 1, padding = 1, dilation = 0 적용.
        self.loc = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)
        # Regressor로 들어가기 위한 연산
        # intermediate 층을 convoltution 연산 잔행.
        # bounding box의 위치 정보를 담아야 하므로 anchor box 개수 곱하기 4 -> x, y, w, h.
        # stride = 1, padding = 1, dilation = 0 적용.
        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)
        # 각각의 가중치를 초기화시킨다. 평균이 0, 분산 0.01 적용.
        
    def forward(self, x, img_size, scale=1.):
        """Forward Region Proposal Network.

        Here are notations.

        * :math:`N` is batch size.
        * :math:`C` channel size of the input.
        * :math:`H` and :math:`W` are height and witdh of the input feature.
        * :math:`A` is number of anchors assigned to each pixel.

        Args:
            x (~torch.autograd.Variable): The Features extracted from images.
                Its shape is :math:`(N, C, H, W)`.
            img_size (tuple of ints): A tuple :obj:`height, width`,
                which contains image size after scaling.
            scale (float): The amount of scaling done to the input images after
                reading them from files.

        Returns:
            (~torch.autograd.Variable, ~torch.autograd.Variable, array, array, array):

            This is a tuple of five following values.

            * **rpn_locs**: Predicted bounding box offsets and scales for \
                anchors. Its shape is :math:`(N, H W A, 4)`.
            * **rpn_scores**:  Predicted foreground scores for \
                anchors. Its shape is :math:`(N, H W A, 2)`.
            * **rois**: A bounding box array containing coordinates of \
                proposal boxes.  This is a concatenation of bounding box \
                arrays from multiple images in the batch. \
                Its shape is :math:`(R', 4)`. Given :math:`R_i` predicted \
                bounding boxes from the :math:`i` th image, \
                :math:`R' = \\sum _{i=1} ^ N R_i`.
            * **roi_indices**: An array containing indices of images to \
                which RoIs correspond to. Its shape is :math:`(R',)`.
            * **anchor**: Coordinates of enumerated shifted anchors. \
                Its shape is :math:`(H W A, 4)`.

        """
        n, _, hh, ww = x.shape # 특징맵의 shape.
        anchor = _enumerate_shifted_anchor(
            np.array(self.anchor_base),
            self.feat_stride, hh, ww)
        # anchor box를 feat_stride와 h, w를 이용해 shift를 만든 뒤 적용.
        # 즉 anchor box가 feature map에 투영되도록 함.

        n_anchor = anchor.shape[0] // (hh * ww)
        # anchor box 개수를 불러옴.
        h = F.relu(self.conv1(x))
        # 특징맵을 이전에 만든 conv1 함수를 적용한 뒤 relu 활성화 함수에 넣는다.
        rpn_locs = self.loc(h)
        # 이전에 만든 loc함수에 h를 넣어 localization 진행.
        # UNNOTE: check whether need contiguous
        # A: Yes
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
        # rpn_locs를 permute를 통해 각 index가 위치하는 곳으로 dimension을 변경 후 contiguous를 통해 연속성있게 바꿔준 뒤
        # reshape와 같은 기능을 가진 view로 dimension을 다시 변경한다.
        rpn_scores = self.score(h)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous()
        rpn_softmax_scores = F.softmax(rpn_scores.view(n, hh, ww, n_anchor, 2), dim=4)
        # softmax함수에 넣기 전 dimension변경.
        # softmax함수에 넣어서 0부터 1사이의 확률적인 값으로 각 객체의 유무를 얻도록 한다.
        rpn_fg_scores = rpn_softmax_scores[:, :, :, :, 1].contiguous()
        rpn_fg_scores = rpn_fg_scores.view(n, -1)
        rpn_scores = rpn_scores.view(n, -1, 2)
        # dimension 변경.
        
        rois = list()
        roi_indices = list()
        # list 선언.
        for i in range(n):
            roi = self.proposal_layer(
                rpn_locs[i].cpu().data.numpy(),
                rpn_fg_scores[i].cpu().data.numpy(),
                anchor, img_size,
                scale=scale)
            # cpu에서 처리할 수 있도록 cpu에 변수올림.
            batch_index = i * np.ones((len(roi),), dtype=np.int32)
            rois.append(roi)
            roi_indices.append(batch_index)
            # 각각의 list에 해당하는 값 넣어줌.

        rois = np.concatenate(rois, axis=0) 
        # rois내부 요소들을 행 방향으로 array들을 합친다.
        roi_indices = np.concatenate(roi_indices, axis=0)
        # roi_indices내부 요소들을 행 방향으로 batch_index들을 합친다.
        return rpn_locs, rpn_scores, rois, roi_indices, anchor


def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    # Enumerate all shifted anchors:
    #
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    # return (K*A, 4)

    # !TODO: add support for torch.CudaTensor
    # xp = cuda.get_array_module(anchor_base)
    # it seems that it can't be boosed using GPU
    import numpy as xp
    shift_y = xp.arange(0, height * feat_stride, feat_stride) 
    # arange 함수로 0 부터 height x feat_stride까지 feat_stide만큼 점들을 찍는다.
    shift_x = xp.arange(0, width * feat_stride, feat_stride)
    # arange 함수로 0 부터 width x feat_stride까지 feat_stide만큼 점들을 찍는다.
    shift_x, shift_y = xp.meshgrid(shift_x, shift_y) # 그렇게 만든 점들을 가지고 meshgrid를 통해 격자점을 생성
    shift = xp.stack((shift_y.ravel(), shift_x.ravel(),
                      shift_y.ravel(), shift_x.ravel()), axis=1)
    # shift_x, y를 1차원으로 풀어서 열방향으로 쌓는다.
    A = anchor_base.shape[0] # anchor box 개수
    K = shift.shape[0] # shift_x, y내부의 점 개수
    anchor = anchor_base.reshape((1, A, 4)) + \
             shift.reshape((1, K, 4)).transpose((1, 0, 2))
    # anchor_base의 첫번째 요소에 shift 정보를 넣어준다.
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor


def _enumerate_shifted_anchor_torch(anchor_base, feat_stride, height, width):
    # Enumerate all shifted anchors:
    #
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    # return (K*A, 4)

    # !TODO: add support for torch.CudaTensor
    # xp = cuda.get_array_module(anchor_base)
    import torch as t
    shift_y = t.arange(0, height * feat_stride, feat_stride)
    shift_x = t.arange(0, width * feat_stride, feat_stride)
    shift_x, shift_y = xp.meshgrid(shift_x, shift_y)
    shift = xp.stack((shift_y.ravel(), shift_x.ravel(),
                      shift_y.ravel(), shift_x.ravel()), axis=1)

    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + \
             shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor
    # _enumerate_shifted_anchor 함수와 거의 일치하는데 arange 함수를 사용할 때 numpy, pytorch 사용 차이.

def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
    # 들어오는 평균과 분산에 따라서 각 가중치에 대한 초기화를 진행해주는 함수.
