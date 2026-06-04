import streamlit as st 
import numpy as np
from colorthief import ColorThief
from PIL import Image
import matplotlib.pyplot as plt
from rembg import remove

st.set_page_config("image shit", page_icon="📷", layout='wide')
imgs = st.file_uploader('Upload Images',type=['jepg','jpg','png'], accept_multiple_files=True)

img = Image.open(imgs[0])

rem = remove(img)

rem.save("trans.png")
st.image('trans.png','Png Image')

ct = ColorThief('trans.png')
dom = ct.get_color(quality=5)
st.write(dom)

fig,ax = plt.subplots(figsize= (2,2))
ax.imshow([[dom]])
ax.axis('off')
st.pyplot(fig)

st.write(np.array(dom))

pal = ct.get_palette(color_count= 3, quality=5)
ax.imshow([[pal[i] for i in range(3)]])
ax.axis('off')
st.pyplot(fig)


# if imgs :
#     cols = st.columns(3)
#     for idx, img in enumerate(imgs):
        
#         col = cols[idx % 3]
#         col.image(img)
#         ct = ColorThief(img)
        
#         fig, ax = plt.subplots(figsize=(2, 2))
#         ax.imshow([[ct.get_color(quality=1)]])
#         ax.axis('off')
        
#         col.pyplot(fig)

