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

Please find the below link to the blog 
