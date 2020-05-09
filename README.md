# Tacotron2 on TPU

A clone of Nvidia/Tacotron2 model using Deepmind Haiku library.
Haiku is based on Jax library which supports CPU/GPUs/TPUs all together!

**Here is a [google colab notebook](https://colab.research.google.com/drive/12VLXy3tim2rBYNewdd0235QlGAcDU2-U?usp=sharing) showing how to train the network on LJ dataset.**


### Advantages:

 - Run on TPU
 - x2 faster than pytorch model with float32 computation on Google Colab.
 - JIT compiled models on CPU/GPUs/TPUs.

### Disadvantages:
 
 - float16 computation is NOT supported.


## Features:

 - Most of features from the original repository. Please use files with `hk_` prefix  (e.g., `hk_train.py` and `hk_inference.ipynb`) for training/inference.
 - Converting nvidia/tacotron2 pretrained model to haiku format. 
            
            $ python torch2haiku.py /path/to/pytorch/model.pt /path/to/output.hk

    For example:
            
            $ gdown --id 1c5ZTuT7J08wLUoVZ2KkUs_VdZuJ86ZqA  # download published nvidia/tactron2 pretrained model: tacotron2_statedict.pt
            $ python torch2haiku.py tacotron2_statedict.pt tacotron2.hk



## Setup:

            ## Install jax jaxlib with GPU support
            $ pip install https://storage.googleapis.com/jax-releases/`nvidia-smi | \
                sed -En "s/.* CUDA Version: ([0-9]*)\.([0-9]*).*/cuda\1\2/p"`/jaxlib-0.1.46-`python3 -V | \
                sed -En "s/Python ([0-9]*)\.([0-9]*).*/cp\1\2/p"`-none-linux_x86_64.whl jax
            $ pip install wandb gdown fire dm-haiku
            $ pip install -r requirements.txt



## Train Tacotron2 text-to-speech model from scratch on LJ dataset.
1. Download the LJ dataset at https://keithito.com/LJ-Speech-Dataset/

            $ cd /tmp
            $ wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
            $ tar -xjf LJSpeech-1.1.tar.bz2
2. Setup paths to *.wav files in the LJ dataset

            $ sed -i -- 's,DUMMY,/tmp/LJSpeech-1.1/wavs,g' filelists/*.txt
3. Train the model:

            $ python hk_train.py --output_directory=outdir/1 --log_directory=logdir --hparams=batch_size=32
4. To restart the training from a checkpoint:

            $ python hk_train.py --output_directory=outdir/1 --log_directory=logdir --hparams=batch_size=32 -c=./outdir/1/checkpoint_1000.hk


## Generate speech from text. 
1. You need a trained tacotron2 model (for example `tacotron2.hk`) and a trained waveglow model. To download the nvidia waveglow pretrained model:

            $ gdown --id 1WsibBTsuRg_SF2Z6L6NFRTT-NjEy1oTx ## > waveglow_256channels_ljs_v2.pt

2. Follow step-by-step instructions in the jupyter notebook `hk_inference.ipynb`.


## Train Tacotron2 on a new dataset from a pretrained model.
1. Prepare your dataset by changing files in `filelists/` directory:
    - `ljs_audio_text_train_filelist.txt`: for training the network (training loss).
    - `ljs_audio_text_val_filelist.txt`: for validation (validation loss).
    - `ljs_audio_text_test_filelist.txt`: for testing at the end.
2. Setup your network and dataset configuration by editting the file `hparams.py`. For examples:
    - `sampling_rate`: The sampling rate of your wav files, usually it's 22050 or 44100.
    - `text_cleaners`: If your dataset is not english, you have to create a new cleaner for your text in the file `text/cleaners.py`. **Also,** redefine `symbols` in the file `text/symbols.py` according to the alphabet of your dataset.
3. We intialize our model by copying weights from a pretrained model (for example, `tacotron2.hk` after convertion).
4. Note that: parametters `hk_ignore_layers` in `hparams.py` lists layers whose weights will be initialized randomly.
5. Train your model:
            $ python hk_train.py --output_directory=outdir --log_directory=logdir --hparams=batch_size=32 --warm_start -c=/path/to/your/pretrain/model.hk


**Below is the original README.md from Nvidia/Tacotron2 repository**

-------

# Tacotron 2 (without wavenet)

PyTorch implementation of [Natural TTS Synthesis By Conditioning
Wavenet On Mel Spectrogram Predictions](https://arxiv.org/pdf/1712.05884.pdf). 

This implementation includes **distributed** and **automatic mixed precision** support
and uses the [LJSpeech dataset](https://keithito.com/LJ-Speech-Dataset/).

Distributed and Automatic Mixed Precision support relies on NVIDIA's [Apex] and [AMP].

Visit our [website] for audio samples using our published [Tacotron 2] and
[WaveGlow] models.

![Alignment, Predicted Mel Spectrogram, Target Mel Spectrogram](tensorboard.png)


## Pre-requisites
1. NVIDIA GPU + CUDA cuDNN

## Setup
1. Download and extract the [LJ Speech dataset](https://keithito.com/LJ-Speech-Dataset/)
2. Clone this repo: `git clone https://github.com/NVIDIA/tacotron2.git`
3. CD into this repo: `cd tacotron2`
4. Initialize submodule: `git submodule init; git submodule update`
5. Update .wav paths: `sed -i -- 's,DUMMY,ljs_dataset_folder/wavs,g' filelists/*.txt`
    - Alternatively, set `load_mel_from_disk=True` in `hparams.py` and update mel-spectrogram paths 
6. Install [PyTorch 1.0]
7. Install [Apex]
8. Install python requirements or build docker image 
    - Install python requirements: `pip install -r requirements.txt`

## Training
1. `python train.py --output_directory=outdir --log_directory=logdir`
2. (OPTIONAL) `tensorboard --logdir=outdir/logdir`

## Training using a pre-trained model
Training using a pre-trained model can lead to faster convergence  
By default, the dataset dependent text embedding layers are [ignored]

1. Download our published [Tacotron 2] model
2. `python train.py --output_directory=outdir --log_directory=logdir -c tacotron2_statedict.pt --warm_start`

## Multi-GPU (distributed) and Automatic Mixed Precision Training
1. `python -m multiproc train.py --output_directory=outdir --log_directory=logdir --hparams=distributed_run=True,fp16_run=True`

## Inference demo
1. Download our published [Tacotron 2] model
2. Download our published [WaveGlow] model
3. `jupyter notebook --ip=127.0.0.1 --port=31337`
4. Load inference.ipynb 

N.b.  When performing Mel-Spectrogram to Audio synthesis, make sure Tacotron 2
and the Mel decoder were trained on the same mel-spectrogram representation. 


## Related repos
[WaveGlow](https://github.com/NVIDIA/WaveGlow) Faster than real time Flow-based
Generative Network for Speech Synthesis

[nv-wavenet](https://github.com/NVIDIA/nv-wavenet/) Faster than real time
WaveNet.

## Acknowledgements
This implementation uses code from the following repos: [Keith
Ito](https://github.com/keithito/tacotron/), [Prem
Seetharaman](https://github.com/pseeth/pytorch-stft) as described in our code.

We are inspired by [Ryuchi Yamamoto's](https://github.com/r9y9/tacotron_pytorch)
Tacotron PyTorch implementation.

We are thankful to the Tacotron 2 paper authors, specially Jonathan Shen, Yuxuan
Wang and Zongheng Yang.


[WaveGlow]: https://drive.google.com/file/d/1WsibBTsuRg_SF2Z6L6NFRTT-NjEy1oTx/view?usp=sharing
[Tacotron 2]: https://drive.google.com/file/d/1c5ZTuT7J08wLUoVZ2KkUs_VdZuJ86ZqA/view?usp=sharing
[pytorch 1.0]: https://github.com/pytorch/pytorch#installation
[website]: https://nv-adlr.github.io/WaveGlow
[ignored]: https://github.com/NVIDIA/tacotron2/blob/master/hparams.py#L22
[Apex]: https://github.com/nvidia/apex
[AMP]: https://github.com/NVIDIA/apex/tree/master/apex/amp
