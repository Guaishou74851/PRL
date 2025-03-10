# (IJCV 2023) Deep Physics-Guided Unrolling Generalization for Compressed Sensing [PyTorch]

[![Springer](https://img.shields.io/badge/Springer-Paper-<COLOR>.svg)](https://link.springer.com/article/10.1007/s11263-023-01814-w) [![ArXiv](https://img.shields.io/badge/ArXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2307.08950) ![visitors](https://visitor-badge.laobi.icu/badge?page_id=Guaishou74851.PRL)

[Bin Chen](https://scholar.google.com/citations?hl=en&user=aZDNm98AAAAJ), [Jiechong Song](https://scholar.google.com/citations?user=EBOtupAAAAAJ), [Jingfen Xie](https://scholar.google.com/citations?user=FKYnbiMAAAAJ), and [Jian Zhang](https://jianzhang.tech/)†

*School of Electronic and Computer Engineering, Peking University Shenzhen Graduate School, Shenzhen, China.*

† Corresponding author

Accepted for publication in International Journal of Computer Vision (IJCV).

⭐ If PRL is helpful to you, please star this repo. Thanks! 🤗

## 📝 Abstract

By absorbing the merits of both the model- and data-driven methods, deep physics-engaged learning scheme achieves high-accuracy and interpretable image reconstruction. It has attracted growing attention and become the mainstream for inverse imaging tasks. Focusing on the image compressed sensing (CS) problem, we find the intrinsic defect of this emerging paradigm, widely implemented by deep algorithm-unrolled networks, in which more plain iterations involving real physics will bring enormous computation cost and long inference time, hindering their practical application. A novel deep Physics-guided unRolled recovery Learning (PRL) framework is proposed by generalizing the traditional iterative recovery model from image domain (ID) to the high-dimensional feature domain (FD). A compact multiscale unrolling architecture is then developed to enhance the network capacity and keep real-time inference speeds. Taking two different perspectives of optimization and range-nullspace decomposition, instead of building an algorithm-specific unrolled network, we provide two implementations: PRL-PGD and PRL-RND. Experiments exhibit the significant performance and efficiency leading of PRL networks over other state-of-the-art methods with a large potential for further improvement and real application to other inverse imaging problems or optimization models.

## 🍭 Overview

Our poster of this work ([high-resolution PDF version](https://drive.google.com/file/d/1FhE6DhD4-yP04GZUc59uys79jT_OVcxx/view?usp=drive_link)):

![poster](figs/PRL-poster.png)

## ⚙ Environment

```shell
torch.__version__ == "1.11.0+cu113"
numpy.__version__ == "1.22.4"
skimage.__version__ == "0.19.2"
```

## ⚡ Test

Download the packaged file of model checkpoints [model.zip](https://drive.google.com/file/d/1C9hFf4qFaqROy0F8pS-t64x3JOxe8wmo/view?usp=drive_link) and put it into `./`, then run:

```shell
unzip model
python test.py --testset_name=Set11 --cs_ratio=0.1/0.2/0.3/0.4/0.5
```

The test sets are in `./data`.

The test sets CBSD68, Urban100, and DIV2K are available at https://github.com/Guaishou74851/SCNet/tree/main/data.

For easy comparison, test results of various existing image CS methods are available on [Google Drive](https://drive.google.com/drive/folders/1Lif_7N_bCyILFLac5JcOtJ9cWpGBNVCd) and [PKU Disk](https://disk.pku.edu.cn/link/AA1C2D8A08050744449CBFCAB51A846B2D).

## 🔥 Train

Download the dataset of [Waterloo Exploration Database](https://kedema.org/project/exploration/index.html) and put the `pristine_images` directory (containing 4744 `.bmp` image files) into `./data`, then run:

```
python train.py --cs_ratio=0.1/0.2/0.3/0.4/0.5
```

The log and model files will be in `./log` and `./model`, respectively.

Note: The `num_feature` and `ID_num_feature` arguments should keep same (e.g. 8 or 16) by default.

## 🎓 Citation

If you find the code helpful in your research or work, please cite the following paper:

```
@article{chen2023deep,
  title={Deep physics-guided unrolling generalization for compressed sensing},
  author={Chen, Bin and Song, Jiechong and Xie, Jingfen and Zhang, Jian},
  journal={International Journal of Computer Vision},
  volume={131},
  number={11},
  pages={2864--2887},
  year={2023},
  publisher={Springer}
}
```
