
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Insect Classifier",
    page_icon="🦋",
    layout="centered"
)

st.markdown("""
<style>
    .stApp {
        background-color: #0f1a0f;
    }
    .main-title {
        color: #4caf50;
        font-size: 2.5em;
        font-weight: bold;
        text-align: center;
        padding: 20px 0;
        text-shadow: 0 0 20px #4caf5066;
    }
    .subtitle {
        color: #81c784;
        text-align: center;
        font-size: 1.1em;
        margin-bottom: 30px;
    }
    .result-box {
        background: linear-gradient(135deg, #1b2e1b, #2d4a2d);
        border: 1px solid #4caf50;
        border-radius: 12px;
        padding: 20px;
        margin-top: 20px;
    }
    .stFileUploader {
        background-color: #1b2e1b;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return tf.keras.applications.MobileNetV2(weights='imagenet')

model = load_model()
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions

tap_tinh = {
    "mantis": ("Bọ ngựa", "🦗", "Săn mồi bằng cách phục kích, có thể quay đầu 180 độ — loài côn trùng duy nhất có khả năng này"),
    "grasshopper": ("Châu chấu", "🦗", "Nhảy xa gấp 20 lần thân mình, sống thành đàn lớn, có thể phá hoại mùa màng"),
    "walking_stick": ("Bọ que", "🌿", "Ngụy trang thành cành cây để tránh kẻ thù, một trong những sinh vật ngụy trang giỏi nhất tự nhiên"),
    "butterfly": ("Bướm", "🦋", "Hút mật hoa bằng vòi dài, giúp thụ phấn cho cây, cánh có phấn để chống thấm nước"),
    "dragonfly": ("Chuồn chuồn", "✨", "Bay với tốc độ 50km/h, săn mồi với tỉ lệ thành công 95%, chỉ thị cho nguồn nước sạch"),
    "bee": ("Ong", "🐝", "Tạo mật từ phấn hoa, có thể nhớ mặt người, đóng vai trò thiết yếu trong hệ sinh thái"),
    "ant": ("Kiến", "🐜", "Tổ chức xã hội chặt chẽ, có thể nâng vật nặng gấp 50 lần thân mình"),
    "fly": ("Ruồi", "🪰", "Có thể phát hiện thức ăn từ xa 750m, đập cánh 200 lần mỗi giây")
}

st.markdown('<div class="main-title">🌿 Insect Classifier AI 🌿</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload ảnh côn trùng — AI sẽ nhận dạng và kể cho bạn nghe về loài đó</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        img_display = Image.open(uploaded_file).convert("RGB")
        st.image(img_display, caption="Ảnh bạn upload", use_column_width=True)
    
    img = img_display.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    predictions = model.predict(img_array)
    results = decode_predictions(predictions, top=3)[0]
    ten_ma, ten_loai, do_tin_cay = results[0]
    
    with col2:
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        
        if ten_loai in tap_tinh:
            ten_viet, icon, mo_ta = tap_tinh[ten_loai]
            st.markdown(f"## {icon} {ten_viet}")
            st.progress(float(do_tin_cay))
            st.caption(f"Độ chính xác: {do_tin_cay:.1%}")
            st.markdown("---")
            st.markdown(f"📖 **Tập tính:** {mo_ta}")
        else:
            st.markdown(f"## 🔍 {ten_loai}")
            st.progress(float(do_tin_cay))
            st.caption(f"Độ chính xác: {do_tin_cay:.1%}")
            st.warning("Chưa có thông tin về loài này")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("### Top 3 dự đoán")
        for _, loai, tin_cay in results:
            st.caption(f"{loai}: {tin_cay:.1%}")
