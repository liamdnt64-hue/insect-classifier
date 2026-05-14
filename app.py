
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

@st.cache_resource
def load_model():
    model = tf.keras.applications.MobileNetV2(weights='imagenet')
    return model

model = load_model()
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions

tap_tinh = {
    "mantis": "Bọ ngựa — săn mồi bằng cách phục kích, có thể quay đầu 180 độ",
    "grasshopper": "Châu chấu — nhảy xa gấp 20 lần thân mình, sống thành đàn lớn",
    "walking_stick": "Bọ que — ngụy trang thành cành cây để tránh kẻ thù",
    "butterfly": "Bướm — hút mật hoa bằng vòi dài, giúp thụ phấn cho cây",
    "dragonfly": "Chuồn chuồn — bay với tốc độ 50km/h, săn mồi với tỉ lệ 95%",
    "bee": "Ong — loài côn trùng có thể tạo mật từ phấn hoa",
    "ant": "Kiến — tổ chức xã hội chặt chẽ",
    "fly": "Ruồi — thích những nơi ô nhiễm"
}

st.title("Nhận dạng côn trùng bằng AI")
st.write("Upload ảnh côn trùng và AI sẽ nhận dạng loài cho bạn!")

uploaded_file = st.file_uploader("Chọn ảnh", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB").resize((224, 224))
    st.image(img, caption="Ảnh bạn upload", width=300)
    
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    predictions = model.predict(img_array)
    results = decode_predictions(predictions, top=3)[0]
    
    ten_ma, ten_loai, do_tin_cay = results[0]
    st.subheader(f"Kết quả: {ten_loai} ({do_tin_cay:.2%})")
    
    if ten_loai in tap_tinh:
        st.info(tap_tinh[ten_loai])
    else:
        st.warning("Chưa có thông tin về loài này")
