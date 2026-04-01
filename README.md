# Photo2Anime (AnimeGAN)

TensorFlow implementation of **AnimeGAN** for fast photo-to-anime style transfer. This project trains and runs models that map real photographs to anime-style images, with support for inference, video stylization, and training pipelines.

A successor experiment, **photo2anime-gan2**, is available separately: [photo2anime-gan2](https://github.com/codeliveyou/photo2anime-gan2).

## AnimeGANv2 (reference) improvements

Compared to earlier AnimeGAN variants, typical v2-style goals include:

1. Reduced high-frequency artifacts in generated images  
2. More stable training and closer reproduction of published results  
3. Fewer parameters in the generator  
4. Higher-quality style data (e.g. from Blu-ray sources)

## Training notes

- The default training set emphasizes **landscape** photos. For people-centric stylization, consider augmenting with a large set of portrait images (on the order of thousands) and retraining.
- When pairing photo and anime training images, **matching gender** between faces often improves facial animation quality.
- Generated tone follows the **style dataset**; avoid predominantly dark anime references unless you compensate with exposure so brightness is consistent across style images.

## Requirements

- Python 3.7  
- TensorFlow GPU 1.15.0 (example: Ubuntu, RTX 2080 Ti, CUDA 10.0.130, cuDNN 7.6.0)  
- OpenCV, tqdm, numpy  
- Standard library: `glob`, `argparse`

Install Python dependencies as appropriate for your CUDA stack.

## Usage

### 1. Inference

```bash
python test.py --checkpoint_dir checkpoint/generator_Hayao_weight --test_dir dataset/test/real --style_name H
```

### 2. Video to anime

```bash
python video2anime.py --video video/input/a.mp4 --checkpoint_dir ./checkpoint/generator_Hayao_weight
```

### 3. Edge smoothing

```bash
python edge_smooth.py --dataset Hayao --img_size 256
```

### 4. Train

```bash
python train.py --dataset Hayao --epoch 101 --init_epoch 5
```

### 5. Export generator weights

```bash
python get_generator_ckpt.py --checkpoint_dir ../checkpoint/AnimeGAN_Hayao_lsgan_300_300_1_1_10 --style_name Hayao
```

## Acknowledgments

Based on ideas and code from [CartoonGAN-Tensorflow](https://github.com/taki0112/CartoonGAN-Tensorflow/blob/master/CartoonGAN.py) and [Anime-Sketch-Coloring-with-Swish-Gated-Residual-UNet](https://github.com/pradeeplam/Anime-Sketch-Coloring-with-Swish-Gated-Residual-UNet). Thanks to the original authors.

## License

See the repository license file if present.
