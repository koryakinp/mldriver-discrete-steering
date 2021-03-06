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

## Agent

<p align="center">
  <b>Neural Network Architecture</b><br>
  <img src="docs/diagram.png">
</p>

 - Input Tensor with Dimensions `[64,64,5]`
 - Convolutional Layer with 32 kernels of size `[8,8]`, strides `[4,4]` and ReLU activation
 - Convolutional Layer with 64 kernels of size `[4,4]`, strides `[2,2]` and ReLU activation
 - Convolutional Layer with 64 kernels of size `[3,3]`, strides `[1,1]` and ReLU activation
 - Fully-Connected Layer with <b>1024</b> neurons and ReLU activation
 - Fully-Connected Layer with <b>512</b> neurons and ReLU activation
 - Fully-Connected Layer with <b>256</b> neurons and ReLU activation
 - Policy Head (Actor) with <b>3</b> output neurons and Value Head (Critic) with <b>1</b> output neurons

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
- [Mostafa Gamal, A Clearer and Simpler Synchronous Advantage Actor Critic (A2C) Implementation in TensorFlow](https://github.com/MG2033/A2C)