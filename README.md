# GranQ: Efficient Channel-wise Quantization via Vectorized Pre-Scaling for Zero-Shot QAT
This repository is the official code for the paper "GranQ: Efficient Channel-wise Quantization via Vectorized Pre-Scaling for Zero-Shot QAT" by Inpyo Hong, Youngwan Jo, Hyojeong Lee, Sunghyun Ahn, Kijung Lee and Sanghyun Park.

## Abstract
Zero-shot quantization (ZSQ) enables neural network compression
without original training data, making it a promising solution for
restricted data access scenarios. To compensate for the lack of data,
recent ZSQ methods typically rely on synthetic inputs generated
from the full-precision model. However, these synthetic inputs
often lead to activation distortion, especially under low-bit settings.
To mitigate this, existing methods typically employ per-channel
scaling, but they still struggle due to the severe computational
overhead during the accumulation process. To overcome this critical
bottleneck, we propose GranQ, a novel activation quantization
framework that introduces an efficient pre-scaling strategy. Unlike
conventional channel-wise methods that repeatedly perform scaling
operations during accumulation, GranQ applies scaling factors in a
pre-scaling step through fully vectorized computation, eliminating
runtime scaling overhead. This design enables GranQ to maintain
fine-grained quantization accuracy while significantly reducing
computational burden, particularly in low-bit quantization settings.
Extensive experiments under quantization-aware training (QAT)
settings demonstrate that GranQ consistently outperforms state-ofthe-
art ZSQ methods across CIFAR and ImageNet. In particular, our
method achieves up to 5.45% higher accuracy in the 3-bit setting on
CIFAR-100 and even surpasses the full-precision baseline on CIFAR-
10. The official code is available at https://github.com/Inpyo-Hong/GranQ.

<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/2ff95ead-350d-42ca-84cb-4f141dce2643" alt="Figure1-1"></td>
    <td><img src="https://github.com/user-attachments/assets/c5d92791-62c0-4b9e-b78c-67caaa08fd99" alt="Figure1-2"></td>
  </tr>
</table>

![Figure3](https://github.com/user-attachments/assets/453a52a7-fe9e-4241-b2c2-d6fab2f493c6)


## Requirements
python==3.8.0<br/>
numpy==1.16.4<br/>
requests==2.21.0<br/>
pyhocon==0.3.51<br/>
torchvision==0.4.0<br/>
torch==1.2.0+cu92<br/>
Pillow==7.2.0<br/>
termcolor==1.1.0


## Usage
Example of 3-bit Quantization on GranQ.
```
conda create -n GranQ python=3.8
conda activate GranQ

cd CIFAR
pip install -r requirements.txt
bash run_3bit.sh
bash run_3bit_cifar100.sh
```

## Experimental Results
Accuracy comparison of GranQ with zero-shot quantization methods.

<img width="1355" height="686" alt="image" src="https://github.com/user-attachments/assets/04ba7ebe-b2ad-4146-bb2e-fedd78744d0d" />
<img width="1376" height="274" alt="image" src="https://github.com/user-attachments/assets/75769a2b-c0ad-409d-8973-515665077925" />



<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/300da185-2ff2-42fd-a062-c90c6fd685ed" alt="Figure2-1"></td>
    <td><img src="https://github.com/user-attachments/assets/84a3a131-7c6c-4e53-bbec-8b2426092052" alt="Figure2-2"></td>


  </tr>
</table>




