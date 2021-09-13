import streamlit as st
import cv2 
import numpy as np    
import tensorflow as tf
import time
import os
from patchify import patchify, unpatchify
import matplotlib.pyplot as plt
from PIL import Image
import keras

def main():
    selected_box = st.sidebar.selectbox(
        'Choose an option..',
        ('About the App','Denoise Image')
        )
    if selected_box == 'About the App':
        readme=Image.open('readme_app.PNG')
        st.image(readme)
                
    if selected_box == 'Denoise Image':
        models()
    
def models():
    st.title("Image Denoising using Deep Learning")
    st.subheader('You can predict on sample images or you can upload a noisy image and get its denoised output.')
    
    selection=st.selectbox("Choose how to load image",["<Select>","Upload an Image","Predict on sample Images"])
    
    if selection=="Upload an Image":
        image = st.file_uploader('Upload the image below')
        predict_button = st.button('Predict on uploaded image')
        if predict_button:
            if image is not None:
                file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
                nsy_img = cv2.imdecode(file_bytes, 1)
                prediction(nsy_img)
            else:
                st.text('Please upload the image')    
    
    if selection=='Predict on sample Images':
        option = st.selectbox('Select a sample image',('<select>','Toy car','Vegetables','Gadget desk','Scrabble board','Shoes','Door','Chess board','A note'))
        if option=='<select>':
            pass
        else:
            path = os.path.join(os.getcwd(),'NoisyImage/')
            nsy_img = cv2.imread(path+option+'.jpg')
            prediction(nsy_img)
            
def patches(img,patch_size):
  patches = patchify(img, (patch_size, patch_size, 3), step=patch_size)
  return patches

#st.cache
def get_model():
    RIDNet=tf.keras.models.load_model('RIDNet.h5')
    return RIDNet

def prediction(img):
    state = st.text('\n Please wait while the model denoise the image.....')
    progress_bar = st.progress(0)
    start = time.time()
    model = get_model()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    nsy_img = cv2.resize(img,(1024,1024))
    nsy_img = nsy_img.astype("float32") / 255.0

    img_patches = patches(nsy_img,256)
    progress_bar.progress(30)
    nsy=[]
    for i in range(4):
        for j in range(4):
            nsy.append(img_patches[i][j][0])
    nsy = np.array(nsy)
    
    pred_img = model.predict(nsy)
    progress_bar.progress(70)
    pred_img = np.reshape(pred_img,(4,4,1,256,256,3))
    pred_img = unpatchify(pred_img, nsy_img.shape)
    end = time.time()
     
    img = cv2.resize(img,(512,512))
    pred_img = cv2.resize(pred_img,(512,512))
    fig,ax = plt.subplots(1,2,figsize=(10,10))
    ax[0].imshow(img) 
    ax[0].get_xaxis().set_visible(False)
    ax[0].get_yaxis().set_visible(False)
    ax[0].title.set_text("Noisy Image")
    
    ax[1].imshow(pred_img) 
    ax[1].get_xaxis().set_visible(False)
    ax[1].get_yaxis().set_visible(False)
    ax[1].title.set_text("Predicted Image")
    
    st.pyplot(fig)
    progress_bar.progress(100)
    st.write('Time taken for prediction :', str(round(end-start,3))+' seconds')
    progress_bar.empty()
    state.text('\n Completed!')
    
if __name__ == "__main__":
    main()   
