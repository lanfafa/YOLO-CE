import warnings

import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'E:\YOLO11-main\ultralytics\cfg\models\11\yolo-CE.yaml')
    model.train(data=r'E:\YOLO11-main\ultralytics\cfg\datasets\UTDAC2020.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                single_cls=False,  # 是否是单类别检测
                batch=64,
                close_mosaic=10,
                workers=0,
                device='0',
                optimizer='SGD',
                amp=True,
                project='runs/train',
                name='exp',
                )