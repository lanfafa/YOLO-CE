# YOLO-CE 

------  

This repository is the official PyTorch implementation of **YOLO-CE**: An Underwater Low-Visibility Environment Target Detection Algorithm Based on YOLOv11.  

**Abstract**  
In challenging underwater environments with limited visibility due to poor lighting and murky water, the ability to detect and identify targets is essential for various marine operations. We present an improved YOLOv11 network, called **YOLO-CE**, to increase the accuracy of underwater target recognition under such circumstances:  
1. **CRAConv Module**: A new convolutional module integrating **Coordinate Attention (CA)** and **Receptive Field Attention (RFA)** into the backbone network.  
2. **Edge Spatial Fusion Module (ESFM)**: Integrated into C3k2 to form **C3k2-ESFM**, enabling deeper learning of multi-scale image features.  
3. **Content-guided Attention (CGA)**: Embedded into the Feature Pyramid Network (FPN) to enforce feature fusion consistency across the backbone and neck.  
4. **Wise IoU v3**: Replaces traditional CIoU to improve target localization precision and stability.  

**Experimental Results**  
On the **UTDAC2020** and **URPC2021** datasets, YOLO-CE achieves **mAP50 scores of 85%** and **82.7%**, respectively. Compared with mainstream object detection algorithms, our approach demonstrates significant improvements in **small-target detection** and overall precision.  

------  

### Training  

If you need to train YOLO-CE from scratch, follow these steps:  

#### 1. Dataset Preparation  
Download the **UTDAC dataset** .  

#### 2. Environmental Requirements  
- Python == 3.9  
- PyTorch == 2.0 (with CUDA support)  
- Other dependencies: `einops`, `numpy`, `opencv-python`, `tensorboardX` (see `requirements.txt` for details)  

#### 3. Training Command  
```bash  
python train.py --data udtac.yaml --cfg yoloce.yaml --weights '' --batch-size 32 --epochs 300 
