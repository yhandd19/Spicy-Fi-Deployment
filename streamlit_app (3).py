import streamlit as st
from PIL import Image
import pandas as pd
import os
import subprocess
import glob
import yaml
from datetime import datetime

# Konfigurasi path model YOLOv5
cfg_model_path = "/content/YOLOv5_EfficientNetLite_Streamlit_v0/runs/train/custom_yolov5s_results12/weights/best.pt"
data_yaml_path = "/content/YOLOv5_EfficientNetLite_Streamlit_v0/Medicine-Detection-9/data.yaml"

# Fungsi untuk membaca nama kelas dari file YAML
def load_class_names(yaml_path):
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
        return data['names']

class_names = load_class_names(data_yaml_path)

# Fungsi untuk mendapatkan direktori terbaru
def get_latest_directory(base_path):
    exp_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and d.startswith('exp')]
    if not exp_dirs:
        return None

    def parse_exp_number(dir_name):
        if dir_name == 'exp':
            return 0
        elif dir_name.startswith('exp'):
            try:
                return int(dir_name[3:])
            except ValueError:
                return -1
        else:
            return -1

    latest_dir_name = max(exp_dirs, key=parse_exp_number)
    return os.path.join(base_path, latest_dir_name)

def run_detection(image_path):
    output_base_dir = '/content/YOLOv5_EfficientNetLite_Streamlit_v0/runs/detect'
    subprocess.run([
        'python', '/content/YOLOv5_EfficientNetLite_Streamlit_v0/detect.py',
        '--weights', cfg_model_path,
        '--data', data_yaml_path,
        '--img', '512',
        '--conf', '0.4',
        '--source', image_path,
        '--save-txt',
    ])
    return get_latest_directory(output_base_dir)

# Fungsi untuk membaca kelas yang terdeteksi dari file hasil deteksi (Perlu disesuaikan)
def read_detected_classes(detected_image_dir, label_dir='labels'):
    detected_classes = set()
    label_path = os.path.join(detected_image_dir, label_dir)
    for file in os.listdir(label_path):
        if file.endswith('.txt'):
            with open(os.path.join(label_path, file), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    class_id = int(line.split()[0])
                    detected_classes.add(class_names[class_id])
    return detected_classes

# Fungsi untuk menampilkan tabel kelas terdeteksi dan tidak terdeteksi
def display_detection_tables(detected_classes):
    st.subheader('Class Detection')

    detected_df = pd.DataFrame({
        'Detected Class': list(detected_classes)
    })
    undetected_classes = set(class_names) - detected_classes
    undetected_df = pd.DataFrame({
        'Undetected Class': list(undetected_classes)
    })

    detected_df.index = pd.Index(range(1, len(detected_classes) + 1), name='No')
    undetected_df.index = pd.Index(range(1, len(undetected_classes) + 1), name='No')

    col1, col2 = st.columns(2)
    with col1:
        st.table(detected_df)
    with col2:
        st.table(undetected_df)

# Fungsi untuk input gambar dan proses deteksi
def imageInput(src):
    if src == 'Upload your own data.':
        image_file = st.file_uploader("Upload An Image", type=['png', 'jpeg', 'jpg'])
        col1, col2 = st.columns(2)
        if image_file is not None:
            img = Image.open(image_file)
            with col1:
                st.image(img, caption='Uploaded Image', use_column_width=True)
            ts = datetime.timestamp(datetime.now())
            upload_dir = '/content/YOLOv5_EfficientNetLite_Streamlit_v0/Medicine-Detection-9/test/images'
            os.makedirs(upload_dir, exist_ok=True)
            imgpath = os.path.join(upload_dir, str(ts) + image_file.name)
            with open(imgpath, mode="wb") as f:
                f.write(image_file.getbuffer())
            detected_image_dir = run_detection(imgpath)

            detected_images = []
            for extension in ['*.jpg', '*.jpeg', '*.png']:
                detected_images.extend(glob.glob(os.path.join(detected_image_dir, extension)))
            with col2:
                for detected_image_path in detected_images:
                    img_ = Image.open(detected_image_path)
                    st.image(img_, caption='Detected Image', use_column_width=True)
           
            detected_classes = read_detected_classes(detected_image_dir)
            display_detection_tables(detected_classes)

# Fungsi utama untuk menjalankan aplikasi Streamlit
def main():
    st.sidebar.title('⚙️ Options')
    datasrc = st.sidebar.radio("Select input source.", ['Upload your own data.'])
    option = st.sidebar.radio("Select input type.", ['Image'])
    st.header('🤚 KNOW-STOCK: Optimalization Knowledge and Stock Management Medicine in Stall Local with Object Detection')
    st.subheader('👈🏽 Select options left-handed menu bar.')
    
    if option == "Image":
        imageInput(datasrc)

if __name__ == '__main__':
    main()
