import torch
from torchvision import transforms
from PIL import Image
import streamlit as st

# Assuming you have already loaded your model and class labels
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, n_class=6):
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, n_class)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class BaseResNet(nn.Module):
    def __init__(self, n_class=6):
        super(BaseResNet, self).__init__()
        self.resnet = ResNet(n_class=n_class)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.resnet(x)
        x = self.dropout(x)
        return x

# Use BaseResNet as the model
model = BaseResNet(n_class=6)
model.eval()
loaded_model = torch.load("checkpoint.pth", map_location=torch.device('cpu'))
model.load_state_dict(loaded_model[0]['model_state_dict'])

cabe_label = ['Healthy', 'Leaf Curl', 'Leaf Spot', 'Powdery Mildew', 'White Fly', 'Yellowish']

# Fungsi untuk menampilkan gambar dengan label dan persentase prediksi
def display_image_with_prediction(image, predicted_class_label, confidence):
    st.image(image, use_column_width=True)
    st.write(f'Prediction: {predicted_class_label}\nConfidence: {confidence:.2f}%')

# Fungsi untuk memuat dan memproses gambar uji
def load_and_process_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    ])
    img = transform(image).unsqueeze(0)
    return img

# Fungsi untuk mendapatkan label kelas
def get_class_label(class_index, class_labels):
    return class_labels[class_index]


# Streamlit App
st.set_page_config(
    page_title="Spicy-Fi",
    page_icon="🌿",
    layout="wide"
)

# Sidebar (Bagian Kiri)
col1_sidebar, col2_sidebar, col3_sidebar = st.sidebar.columns(3)

with col2_sidebar:
    st.image("FF.png", use_column_width=True)

st.sidebar.markdown("<h1 style='text-align: center;'>Masukkan Foto Daun Cabai Anda</h1>", unsafe_allow_html=True)
st.sidebar.markdown('<hr style="border: 1px solid">', unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader("", type=["jpg", "jpeg", "png", "gif"])

# Halaman Utama (Bagian Kanan)
col1_HU, col2_HU = st.columns(2)
gif_url = "cabe_gif.gif"
with col1_HU:
    st.image(gif_url)

with col2_HU: 
    st.markdown("<h1 style='margin-left:40px;margin-top: 79px;margin-bottom: 20px;'>S  p  i  c  y  - F  i</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='margin-left:40px;margin-bottom: 10px; color:#2e8b57'> A tool to detect pests on the leaves of your chili plants.</h2>", unsafe_allow_html=True)
    st.write("<h4 style='margin-left:40px;font-weight:lighter;margin-bottom: 20px; color:white'> Created By Future Founders</h4>", unsafe_allow_html=True)
    st.write("<h4 style='margin-left:40px;font-weight:lighter;margin-bottom: 20px; color:white'> Supported By Startup Campus & Kampus Merdeka</h4>", unsafe_allow_html=True)

st.markdown('<hr style="border: 1px solid">', unsafe_allow_html=True)

st.markdown("<h1 style='margin-top: -10px; color:#2e8b57'>Overview</h1>", unsafe_allow_html=True)
st.write("<p style='text-align:justify;line-height:35px;margin-bottom: 20px; color:white'>Spicy-Fi adalah proyek computer vision yang bertujuan untuk mendeteksi hama pada tanaman cabai menggunakan arsitektur ResNet Disini, kita dapat mengidentifikasi beberapa jenis hama pada daun cabai, termasuk Leaf Curl, Leaf Spot, Powdery Mildew, dan serangan Whitefly, serta mengklasifikasikan daun yang sehat. Spicy-Fi diharapkan dapat memberikan edukasi tentang penyakit tanaman cabai, membantu mereka mengidentifikasi dan mengatasi masalah potensial.</p>", unsafe_allow_html=True)



# ... (kode sebelumnya tetap tidak berubah)

# Baris Ketiga
col1_HU, col2_HU = st.columns(2)

if uploaded_file:
    # Load and preprocess the image
    test_image = Image.open(uploaded_file)
    test_image_resized = test_image.resize((250,250))  # Ubah ukuran sesuai kebutuhan Anda

    # Tampilkan foto yang diunggah di kolom pertama (kolom kiri) dengan border
    with col1_HU:
        st.image(test_image_resized, use_column_width=True, caption="Uploaded Image")

    # Tombol Identifikasi di kolom kedua (kolom kanan)
    with col2_HU:
        # Tambahkan kontrol percabangan untuk menampilkan tombol identifikasi setelah gambar diupload
        if st.button("Identifikasi"):
            # Load and preprocess the image
            test_image_tensor = load_and_process_image(test_image_resized)

            # Perform inference
            with torch.no_grad():
                output = model(test_image_tensor)

            # Get predicted class and confidence
            predicted_class_index = torch.argmax(output).item()
            predicted_class_label = get_class_label(predicted_class_index, cabe_label)
            confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted_class_index].item() * 100

            st.markdown(f'<h5 style="margin-top: 20px;background-color:#cd5c5c;width:230px;padding:20px;border-radius:8px;">Prediction: {predicted_class_label}</h5>', unsafe_allow_html=True)
            st.markdown(f'<h5 style="margin-top: 20px;background-color:#cd5c5c;width:230px;padding:20px;border-radius:8px">Confidence: {confidence:.2f}%</h5>', unsafe_allow_html=True)

            if predicted_class_label == 'Leaf Curl':
                st.markdown("<h5 style='margin-top: 20px;margin-bottom: 20px;background-color:#cd5c5c;text-align:justify;padding:20px;border-radius:8px'>Deskripsi : Daun-daun yang terinfeksi cenderung melengkung atau keriting. Penyebabnya dapat berupa serangga, virus, atau faktor lingkungan.</h5>", unsafe_allow_html=True)
                st.markdown("<h5 style='margin-bottom: 20px;background-color:#cd5c5c;text-align:justify;padding:20px;border-radius:8px'>Penanganan Leaf Curl : Cabut dan hancurkan tanaman yang terinfeksi untuk mencegah penyebaran.</h5>", unsafe_allow_html=True)
            elif predicted_class_label == 'Leaf Spot':
                st.markdown("<h5 style='margin-top: 20px;margin-bottom: 20px;background-color:#cd5c5c;text-align:justify;padding:20px;border-radius:8px'>Deskripsi : Bercak-daun dapat disebabkan oleh berbagai patogen, seperti jamur atau bakteri. Gejala umumnya berupa bercak-bercak pada daun yang dapat berwarna hitam, coklat, atau merah keunguan.</h5>", unsafe_allow_html=True)
                st.markdown("<h5 style='margin-bottom: 20px;background-color:#cd5c5c;text-align:justify;padding:20px;border-radius:8px'>Penanganan Leaf Spot : Gunakan fungisida atau bakterisida yang sesuai. Jaga kebersihan lingkungan dan tanaman.</h5>", unsafe_allow_html=True)
            elif predicted_class_label == 'Powdery Mildew':
                st.markdown("<h5 style='margin-top: 20px;margin-bottom: 20px;background-color:#cd5c5c;text-align:justify;padding:20px;border-radius:8px'>Deskripsi : Jamur berbentuk serbuk putih pada permukaan daun. Biasanya disebabkan oleh kondisi lembab dan suhu yang tinggi.</h5>", unsafe_allow_html=True)
                st.markdown("<h5 style='margin-bottom: 20px;background-color:#cd5c5c;width:500px;text-align:justify;padding:20px;border-radius:8px'>Penanganan Powdery Mildew : Gunakan fungisida, pastikan sirkulasi udara yang baik, dan hindari kelembaban berlebihan.</h5>", unsafe_allow_html=True)
            elif predicted_class_label == 'White Fly':
                st.markdown("<h5 style='margin-top: 20px;margin-bottom: 20px;background-color:#cd5c5c;text-align:justify;padding:20px;border-radius:8px'>Deskripsi: Serangga kecil berwarna putih yang menempel pada bagian bawah daun dan dapat menyebabkan kerusakan dengan menghisap cairan </h5>", unsafe_allow_html=True)
                st.markdown("<h5 style='margin-bottom: 20px;background-color:#cd5c5c;text-align:justify;padding:20px;border-radius:8px'>Penanganan White Fly : Gunakan insektisida yang sesuai, pertahankan kebersihan lingkungan</h5>", unsafe_allow_html=True)
            elif predicted_class_label == 'Yellowish':
                st.markdown("<h5 style='margin-top: 20px;margin-bottom: 20px;background-color:#cd5c5c;text-align:justify;padding:20px;border-radius:8px'>Deskripsi : Daun-daun yang berubah warna menjadi kuning, biasanya disebabkan oleh kekurangan nutrisi, penyakit, atau kondisi lingkungan.</h5>", unsafe_allow_html=True)
                st.markdown("<h5 style='margin-bottom: 20px;background-color:#cd5c5c;text-align:justify;padding:20px;border-radius:8px'>Penanganan Yellowish : Memberikan pupuk dengan kandungan nutrisi yang tepat dapat membantu mengobati</h5>", unsafe_allow_html=True)
            elif predicted_class_label == 'Healthy':
                st.markdown("<h5 style='margin-bottom: 20px;background-color:#cd5c5c;margin-top:20px;text-align:justify;padding:20px;border-radius:8px'>Deskripsi : Tanaman yang sehat memiliki daun berwarna hijau cerah, batang yang kuat, dan pertumbuhan yang baik tanpa gejala penyakit.</h5>", unsafe_allow_html=True)
                # Tambahkan informasi khusus untuk tanaman yang sehat di sini
