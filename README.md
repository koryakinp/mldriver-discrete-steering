# Implementation of Advantage Actor-Critic Algorithm (A2C) with visual observations and discrete action space

## Environment

Custom build [MLDriver](https://github.com/koryakinp/MLDriver) <b>Unity Environment</b>

#### Links

 - [Linux Build](https://github.com/koryakinp/MLDriver/releases/download/5.1/MLDriver_Linux_x86_64.zip)
 - [MAC OS build](https://github.com/koryakinp/MLDriver/releases/download/5.1/MLDriver_MAC_OS_X.zip)
 - [Windows Build](https://github.com/koryakinp/MLDriver/releases/download/5.1/MLDriver_Windows_x86_64.zip)
 - [Web Build](https://koryakinp.github.io/MLDriver)

#### Environment Parameters:

 - Observation Space: `[64, 64, 1]`
 - Action Space: `[3]`

## Model

<p align="center">
  <b>Neural Network Architecture</b><br>
  <img src="docs/diagram.png">
</p>

 - Input Tensor with Dimensions `[64,64,5]`
 - Convolutional Layer with filter size `[8,8]` and strides `[4,4]` and ReLU activation
 - Convolutional Layer with filter size `[4,4]` and strides `[2,2]` and ReLU activation
 - Convolutional Layer with filter size `[3,3]` and strides `[1,1]` and ReLU activation
 - Fully-Connected Layer with 1024 neurons and ReLU activation
 - Fully-Connected Layer with 512 neurons and ReLU activation
 - Fully-Connected Layer with 256 neurons and ReLU activation
 - Policy Head with 3 output neurons and Value head with 1 output neurons

 ## Results

<p align="center">
  <b>Smothed average episode reward vs number of training steps</b><br>
  <img src="docs/reward-chart.png">
</p>

<p align="center">
  <b>Sample Run</b><br>
  <img src="docs/run.gif" width="320" height="320">
</p>

## Authors
Pavel Koryakin <koryakinp@koryakinp.com>

## Acknowledgments
- [Chris Yoon, Understanding Actor Critic Methods and A2C](https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f)
- [Daniel Seita, Actor-Critic Methods: A3C and A2C](https://danieltakeshi.github.io/2018/06/28/a2c-a3c/)