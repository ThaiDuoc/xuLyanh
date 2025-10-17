import streamlit as st
import cv2
import numpy as np
from scipy.spatial import distance as dist
from imutils import perspective, contours
import imutils
import io

st.set_page_config(page_title="📏 Đo kích thước vật thể", layout="wide")

# --- Custom giao diện CSS ---
st.markdown("""
    <style>
    /* Tổng thể giao diện */
    [data-testid="stAppViewContainer"] {
        background-color: #f8f9fa;
        padding: 2rem 4rem;
    }

    /* Tiêu đề chính */
    h1 {
        color: #2c3e50;
        text-align: center;
        font-weight: 700;
        font-size: 2.2rem;
    }

    /* Mô tả phụ */
    .stCaption {
        text-align: center;
        color: #6c757d;
        font-size: 1rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #dee2e6;
        box-shadow: 2px 0 6px rgba(0,0,0,0.1);
    }

    /* Các nút */
    div.stButton > button {
        background: linear-gradient(90deg, #00b4d8, #0077b6);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        font-size: 1rem;
        font-weight: bold;
        transition: 0.2s;
    }
    div.stButton > button:hover {
        background: linear-gradient(90deg, #0096c7, #023e8a);
        transform: scale(1.02);
    }

    /* Uploader */
    [data-testid="stFileUploader"] {
        border: 2px dashed #90e0ef;
        background-color: #e3f2fd;
        border-radius: 10px;
        padding: 1rem;
    }

    /* Ảnh hiển thị */
    img {
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

st.title("📏 ỨNG DỤNG ĐO KÍCH THƯỚC VẬT THỂ (OpenCV)")
st.caption("Tải ảnh có **vật tham chiếu** (ví dụ đồng xu 20mm) — ứng dụng sẽ tự đo kích thước (mm).")

# --- Sidebar cấu hình ---
with st.sidebar:
    st.header("⚙️ Thiết lập")
    ref_width = st.number_input("Kích thước vật tham chiếu (mm)", 1.0, 300.0, 20.0, 1.0)
    canny_low = st.slider("Canny Low", 0, 255, 50)
    canny_high = st.slider("Canny High", 0, 255, 100)
    blur_kernel = st.selectbox("Độ mờ Gaussian (odd)", [3,5,7,9,11], 3)
    area_threshold = st.slider("Ngưỡng lọc contour nhỏ", 0, 20000, 3000, 100)

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) / 2, (ptA[1] + ptB[1]) / 2)

def get_distance_in_pixels(orig, c):
    box = cv2.minAreaRect(c)
    box = cv2.boxPoints(box)
    box = np.array(box, dtype="int")
    box = perspective.order_points(box)
    cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

    (tl, tr, br, bl) = box
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

    dc_W = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dc_H = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
    return dc_W, dc_H, tltrX, tltrY, trbrX, trbrY

def process_image(image, ref_width, canny_low, canny_high, blur_kernel, area_threshold):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    edged = cv2.Canny(gray, canny_low, canny_high)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if len(cnts) == 0:
        return image, edged, None

    (cnts, _) = contours.sort_contours(cnts)
    P = None
    results = []

    for c in cnts:
        if cv2.contourArea(c) < area_threshold:
            continue

        dc_W, dc_H, tltrX, tltrY, trbrX, trbrY = get_distance_in_pixels(image, c)

        if P is None:
            P = ref_width / dc_H
            dr_W = ref_width
            dr_H = ref_width
        else:
            dr_W = dc_W * P
            dr_H = dc_H * P

        cv2.putText(image, f"{dr_H:.1f} mm", (int(tltrX - 15), int(tltrY - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(image, f"{dr_W:.1f} mm", (int(trbrX + 10), int(trbrY)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        results.append((dr_W, dr_H))

    return image, edged, results

uploaded_file = st.file_uploader("⬆️ Tải ảnh JPG/PNG có vật tham chiếu", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Ảnh gốc", use_container_width=True)
    if st.button("🚀 Đo kích thước"):
        result, edged, measures = process_image(image.copy(), ref_width, canny_low, canny_high, blur_kernel, area_threshold)
        with col2:
            st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="Kết quả", use_container_width=True)
