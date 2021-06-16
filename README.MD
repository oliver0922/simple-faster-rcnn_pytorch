# KITT Dataset을 이용해 Faster R-CNN 학습하기  (Google Coab 환경)

## 1. Data Preparation and processing 

(1) Download KITTI Dataset and Upload on Gdirve  

link= http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d  

1. VOC2007 폴더를 생성한 후 내부에 JPEGImages, ImageSets, Annotations 3개의 폴더를 생성한다.  

![image](https://user-images.githubusercontent.com/69920975/122263210-2362c000-cf11-11eb-9ae1-d26932e05be6.png)

2. ImageSets 폴더 내부에 Main 폴더를 생성한다.

3. 위 링크에 접속하여 left color images of objec data set(12GB)와 training labels of object data set(5MB)를 다운로드한다.  


4. train 폴더 내부의 7481개의 image의 확장자를 png에서 jpg로 변경해준다. 

5. 다시 압축을 한 후 gdrive에 업로드 해준다.  

6. JPEGImages 폴더 내부에 image 압축파일을 넣고 해제한 후 .

(2) Data Preprocessing  

1. (Optional) KITTI Dataset의 class ('Car', 'Van', 'Truck','Pedestrian', 'Person_sitting', 'Cyclist', 'Tram','Misc' or 'DontCare') 의 개수는 총 9개이다.  
class의 개수를 줄이고 싶다면 Fine tuning을 위해 classnamechange.py 파일 내부의 class를 변경해주고 실행해준다. 

2. Faster R-CNN에 사용될 dataset은 Pascal VOC dataset 형식(xml 확장자파일)이므로 KITTI Dataset label(txt 확장자파일)을 변경해줄 필요가 있다.
따라서 txt_to_xml.py 파일을 실행해준다. 

3. trainval, test dataset으로 나누기 위해 ceat_train_test_txt.py 파일을 실행시켜준다. 

## 2. Git clone and import packages

git repository를 clone 하고 requirements.txt 파일을 실행시켜 내부의 package들을 import 한다.

## 3. Setting IP and Port for Using visdom  

visdom library(훈련 과정을 관찰하게 해주는 library)를 사용하기 위해 visdom을 설치해준다.
utils/config.py 내부에서 port 번호를 변경해준다. 
utils/vis_tool.py 내부에서  Visualizer class 내부의 생성자 함수에서 visdom.Visdom('자신의 IP주소', port 번호)로 변경해준다.

## 4. Train  

train.py 파일을 실행시킨다. (train.py 파일 내부에서 epoch를 변경할 수 있다.)  
다른 hyperparameter를 변경하고자 하는 경우는 utils/config.py 내부에서 값을 변경해준다. 
본인이 설정한 ip주소:port 번호로 접속을 하면 아래와 같이 훈련과정을 살펴보고 그 때의 mAP값을 측정할 수 있다.   

#### Training Loss  
(roi_loc_loss = roi pooling을 거친 후의 localization loss, roi_cls_loss = roi pooling을 거친 후의 classification(물체의 종류) loss, rpn_loc_loss = rpn network에서의 localization loss, rpn_cls_loss= rpn network에서의 bounding box의 classification(물체의 유무만 판단) loss)
![image](https://user-images.githubusercontent.com/69920975/122257685-35416480-cf0b-11eb-9690-a9dace1d2373.png)  
  
#### Total Loss

![image](https://user-images.githubusercontent.com/69920975/122258688-4a6ac300-cf0c-11eb-8b21-91699b0e7dd7.png)

  
#### Ground Truth Image VS Predicted Image  

![image](https://user-images.githubusercontent.com/69920975/122258459-04adfa80-cf0c-11eb-9cc2-e76eb5038bd8.png)


#### mAP(meana Average Precision)  

![image](https://user-images.githubusercontent.com/69920975/122258755-5eaec000-cf0c-11eb-89eb-3fc429bfbfd0.png)

  




Licensed under MIT, see the LICENSE for more detail.


