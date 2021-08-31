# ImageDenoising
The images that are captured in the real world come with a lot of noise. So there is a need to remove these noise from images when it comes to low level vision tasks and similar applications.

Over the years many techniques and filters have been introduced for image denoising. They used to work to some extent in denoising the images. But most of these techniques assumed the noise in images to be gaussian noise or impulse noise. But this assumption doesn't completely hold true for real noise in photographs. The real world noise is more sophisticated and diverse. Due to this most of these techniques performed poorly in completely removing real noise from images.

This is where deep learning comes into picture and experiments have proved that training a convolutional blind denoising deep learning network outperforms other techniques in image denoising tasks by a large margin.

In this case study, I have implemented four state-of-the-art CNN based architecture for image denoising tasks as follows
1. Autoencoder (as a baseline model)
2. CBDNet
3. PRIDNet
4. RIDNet

Among these models, RIDNet gave the best performance and using it, I have created a web app using streamit and deployed the same using streamlit sharing.
Check the below link to see how the model denoise an image.
https://share.streamlit.io/sharathsolomon/imagedenoising/main/model.py

Below are the few predcitions of RIDNet model in image denoising on some real noisy images
![image](https://user-images.githubusercontent.com/85414148/131446414-a4dbe8cf-f8c6-4ec5-887a-fc3a4f3deb42.png)
![image](https://user-images.githubusercontent.com/85414148/131446612-ec16213e-fc6b-4a5a-9d52-70dffe22e799.png)

You can check out the below blog for a detailed explanation of this case study.
https://sharathsolomon.medium.com/image-denoising-using-deep-learning-dc2b19a3fd54

This repository contains all the relevant codes and python notebooks in reference to this case study.
