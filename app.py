import streamlit as st 
import numpy as np
from colorthief import ColorThief
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config("image shit", page_icon="📷", layout='wide')
imgs = st.file_uploader('Upload Images',type=['jepg','jpg','png'], accept_multiple_files=True)



if imgs :
    cols = st.columns(3)
    for idx, img in enumerate(imgs):
        
        col = cols[idx % 3]
        col.image(img)
        ct = ColorThief(img)
        
        fig, ax = plt.subplots(figsize=(2, 2))
        ax.imshow([[ct.get_color(quality=1)]])
        ax.axis('off')
        
        col.pyplot(fig)

