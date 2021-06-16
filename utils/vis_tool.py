import time

import numpy as np
import matplotlib
import torch as t
import visdom

matplotlib.use('Agg')
from matplotlib import pyplot as plot

# from data.voc_dataset import VOC_BBOX_LABEL_NAMES

# VOC dataset의 class label명.
# 필자는 kitti dataset을 customize하여 적용했으므로 변경했었음.
VOC_BBOX_LABEL_NAMES = (
    'fly',
    'bike',
    'bird',
    'boat',
    'pin',
    'bus',
    'c',
    'cat',
    'chair',
    'cow',
    'table',
    'dog',
    'horse',
    'moto',
    'p',
    'plant',
    'shep',
    'sofa',
    'train',
    'tv',
)


def vis_image(img, ax=None):
    """Visualize a color image.

    Args:
        img (~numpy.ndarray): An array of shape :math:`(3, height, width)`.
            This is in RGB format and the range of its value is
            :math:`[0, 255]`.
        ax (matplotlib.axes.Axis): The visualization is displayed on this
            axis. If this is :obj:`None` (default), a new axis is created.

    Returns:
        ~matploblib.axes.Axes:
        Returns the Axes object with the plot for further tweaking.

    """

    if ax is None:
        fig = plot.figure()
        ax = fig.add_subplot(1, 1, 1)
    # CHW -> HWC
    img = img.transpose((1, 2, 0))
    ax.imshow(img.astype(np.uint8))
    return ax
# visdom과 matplot library를 이용하여 image를 시각화하는 함수.

def vis_bbox(img, bbox, label=None, score=None, ax=None):
    """Visualize bounding boxes inside image.

    Args:
        img (~numpy.ndarray): An array of shape :math:`(3, height, width)`.
            This is in RGB format and the range of its value is
            :math:`[0, 255]`.
        bbox (~numpy.ndarray): An array of shape :math:`(R, 4)`, where
            :math:`R` is the number of bounding boxes in the image.
            Each element is organized
            by :math:`(y_{min}, x_{min}, y_{max}, x_{max})` in the second axis.
        label (~numpy.ndarray): An integer array of shape :math:`(R,)`.
            The values correspond to id for label names stored in
            :obj:`label_names`. This is optional.
        score (~numpy.ndarray): A float array of shape :math:`(R,)`.
             Each value indicates how confident the prediction is.
             This is optional.
        label_names (iterable of strings): Name of labels ordered according
            to label ids. If this is :obj:`None`, labels will be skipped.
        ax (matplotlib.axes.Axis): The visualization is displayed on this
            axis. If this is :obj:`None` (default), a new axis is created.

    Returns:
        ~matploblib.axes.Axes:
        Returns the Axes object with the plot for further tweaking.

    """
    # 각 boundingbox가 이미지에 투영되어 나타나도록 하는 visdom 함수.
    label_names = list(VOC_BBOX_LABEL_NAMES) + ['bg']
    # add for index `-1`
    if label is not None and not len(bbox) == len(label):
        raise ValueError('The length of label must be same as that of bbox')
    if score is not None and not len(bbox) == len(score):
        raise ValueError('The length of score must be same as that of bbox')
    # 오류 설정.
    
    # Returns newly instantiated matplotlib.axes.Axes object if ax is None
    ax = vis_image(img, ax=ax)
    # 위에서 만든 visdom으로 나타내려는 imaeg를 가져온다.
    # If there is no bounding box to display, visualize the image and exit.
    if len(bbox) == 0:
        return ax
    # bounding box가 없으면 그냥 image만 출력하도록 한다.
    for i, bb in enumerate(bbox):
        xy = (bb[1], bb[0])
        height = bb[2] - bb[0]
        width = bb[3] - bb[1]
        ax.add_patch(plot.Rectangle(
            xy, width, height, fill=False, edgecolor='red', linewidth=2))
    # bounding box가 존재하면 bbox의 정보를 받아와서 색깔과 선의 굵기를 결정해서 나타낼 수 있도록 한다.
    # ax(visdom형식의 image)에 patch를 넣는 함수를 이용.
        caption = list()
    
        if label is not None and label_names is not None:
            lb = label[i]
            if not (-1 <= lb < len(label_names)):  # modfy here to add backgroud
                raise ValueError('No corresponding name is given')
            caption.append(label_names[lb])
        # label이 존재할 때 caption을 이용해서 visdom에 이미지와 boundingbox와 같이 출력할 수 있도록 한다.
        if score is not None:
            sc = score[i]
            caption.append('{:.2f}'.format(sc))
        # 마찬가지로 score가 존재하면 bounding box와 label과 함꼐 이미지에 시각화한다.

        if len(caption) > 0:
            ax.text(bb[1], bb[0],
                    ': '.join(caption),
                    style='italic',
                    bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 0})
        # caption이 존재한다면 image에 나타날 수 있도록 한다.
        # 부가적인 설정 추가.
    return ax
    # bbox, label name, score의 추가적인 설정이 완료된 image가 반환됨.


def fig2data(fig):
    """
    brief Convert a Matplotlib figure to a 4D numpy array with RGBA 
    channels and return it
    
    @param fig: a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()
    # fig를 그림.
    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)   
    # fig를 canvas를 이용해 그리고 numpy의 fromstring을 통해 canvas내에 그려진 fig의 정보를 가져온다.

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf.reshape(h, w, 4)


def fig4vis(fig):
    """
    convert figure to ndarray
    """
    ax = fig.get_figure()
    img_data = fig2data(ax).astype(np.int32)
    plot.close()
    # HWC->CHW
    return img_data[:, :, :3].transpose((2, 0, 1)) / 255.
# fig를 visdom으로 visualization하기 위한 전처리 함수.

def visdom_bbox(*args, **kwargs):
    fig = vis_bbox(*args, **kwargs)
    data = fig4vis(fig)
    return data
# args list와 kwargs dictionary를 받아와서 bounding box를 visdom형태로 만든다.

class Visualizer(object):
    """
    wrapper for visdom
    you can still access naive visdom function by 
    self.line, self.scater,self._send,etc.
    due to the implementation of `__getattr__`
    """
# visualizaion을 위한 class
    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom('localhost',env=env, use_incoming_socket=False, **kwargs)
        self._vis_kw = kwargs
    # visdom 초기설정.
    # localhost로 설정되어 host의 local환경에서 visdom을 돌리도록함.
    # 원한다면 다른 ip주소를 입력하여 다른 환경에서 visdom을 실행할 수 있다.
        # e.g.('loss',23) the 23th value of loss
        self.index = {}
        self.log_text = ''

    def reinit(self, env='default', **kwargs):
        """
        change the config of visdom
        """
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self
    # self.vis를 다시 선언.
    # 재생성함수.
    
    def plot_many(self, d):
        """
        plot multi values
        @params d: dict (name,value) i.e. ('loss',0.11)
        """
        for k, v in d.items():
            if v is not None:
                self.plot(k, v)
    # 여러개를 plot하려고 할때.
                
    def img_many(self, d):
        for k, v in d.items():
            self.img(k, v)
    # 여러 image를 나타내기 위함.

    def plot(self, name, y, **kwargs):
        """
        self.plot('loss',1.00)
        """
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=name,
                      opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + 1
    # 
    
    def img(self, name, img_, **kwargs):
        """
        self.img('input_img',t.Tensor(64,64))
        self.img('input_imgs',t.Tensor(3,64,64))
        self.img('input_imgs',t.Tensor(100,1,64,64))
        self.img('input_imgs',t.Tensor(100,3,64,64),nrows=10)
        !!don't ~~self.img('input_imgs',t.Tensor(100,64,64),nrows=10)~~!!
        """
        self.vis.images(t.Tensor(img_).cpu().numpy(),
                        win=name,
                        opts=dict(title=name),
                        **kwargs
                        )

    def log(self, info, win='log_text'):
        """
        self.log({'loss':1,'lr':0.0001})
        """
        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m%d_%H%M%S'), \
            info=info))
        self.vis.text(self.log_text, win)

    def __getattr__(self, name):
        return getattr(self.vis, name)

    def state_dict(self):
        return {
            'index': self.index,
            'vis_kw': self._vis_kw,
            'log_text': self.log_text,
            'env': self.vis.env
        }

    def load_state_dict(self, d):
        self.vis = visdom.Visdom(env=d.get('env', self.vis.env), **(self.d.get('vis_kw')))
        self.log_text = d.get('log_text', '')
        self.index = d.get('index', dict())
        return self
