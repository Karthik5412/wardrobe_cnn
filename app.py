import streamlit as st 
import numpy as np
from colorthief import ColorThief
from PIL import Image
import matplotlib.pyplot as plt
from rembg import remove
import pandas as pd
import io 

def val_to_df (imgs) :
    res = []
    
    for img in imgs:
        
        raw_pixels = remove(img.read(), force_return_bytes=True)
        raw_data = Image.open(io.BytesIO(raw_pixels))
        rgb_data = Image.new("RGB", raw_data.size,(255,255,255))
        rgb_data.paste(raw_data,mask=raw_data.split()[3])
        img_arr = io.BytesIO()
        rgb_data.save(img_arr,format="jpeg")
        img_arr.seek(0)
        
        ct = ColorThief(img_arr)
        pal = ct.get_color(quality=1)
        res.append(pal)
        
    df = pd.DataFrame(res, columns=["Red", "Green", "Blue"])
    
    return df


st.set_page_config("image shit", page_icon="📷", layout='wide')
imgs = st.file_uploader('Upload Images',type=['jepg','jpg','png'], accept_multiple_files=True)

if imgs :
    cols = st.columns(3) 
    for idx, img in enumerate(imgs) :
        col = cols[idx % 3]
        col.image(img)

result = val_to_df(imgs)

st.dataframe(result)


fig, ax = plt.subplots(figsize=(2,2))
ax.imshow([[(31,137,65)]])
ax.axis('off')
st.pyplot(fig)

