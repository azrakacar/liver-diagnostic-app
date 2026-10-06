# 🩺 Hibrit Karaciğer Teşhis Web Uygulaması

**Flask**, **Scikit-learn** ve **TensorFlow/Keras** kullanılarak geliştirilmiş, profesyonel ve yapay zeka destekli bir hibrit karaciğer teşhis web uygulamasıdır. Bu proje, kan tahlili verilerini (tablo verileri) ultrason görüntüleme (Yapay Sinir Ağları - CNN) ile birleştirerek hastalara kapsamlı, anlaşılır ve empatik bir tıbbi rehberlik sunmayı hedefler.

## 🚀 Öne Çıkan Özellikler

* **Hibrit Analiz Motoru:** Yapılandırılmış klinik verileri (kan değerleri) ile derin öğrenme tabanlı görüntü işlemeyi ortak bir karar mekanizmasında birleştirir.
* **Tablolu Veri Modeli:** En doğru teşhis tahmincisini seçmek için birden fazla makine öğrenmesi algoritmasını (`Random Forest`, `KNN`, `Logistic Regression`, `SVM`) karşılaştırır.
* **Derin Öğrenme (CNN) Görüntüleme:** Sınıf yanlılığını (bias) önlemek için dengeli sınıf ağırlıkları (`class weights`) kullanarak ultrason taramalarını 3 sınıflı bir yapıda (**Benign - İyi Huylu**, **Malignant - Kötü Huylu**, **Normal**) değerlendirir.
* **Empatik Hasta Arayüzü:** Dinamik, renk kodlu tıbbi rapor şablonu; hasta motivasyonu, risk analizleri ve semptomlar ile tedavi yolları üzerine detaylı eğitim rehberleri içerir.

* ## 🛠️️ Kullanılan Teknolojiler

* * **Arka Plan (Backend):** Python, Flask, Jinja2
* **Makine Öğrenmesi (Tablo):** Scikit-learn, Pandas, NumPy, Joblib
* **Derin Öğrenme (Görüntü):** TensorFlow, Keras, Evrişimli Sinir Ağları (CNN)
* **Ön Plan (Frontend):** HTML5, Bootstrap 5, CSS3

* ## 📂 Proje Mimarisi

* ```text
├── dataset/                    # Klinik CSV verileri ve ultrason görselleri
├── saved_models/               # Eğitilmiş .pkl (tablo) ve .h5 (görüntü) modelleri
├── static/                     # CSS, JS ve görsel dosyaları
├── templates/                  # HTML şablonları (result.html dahil)
├── app.py                      # Ana Flask web uygulaması
├── image_model.py              # CNN görüntü modeli eğitim betiği
├── train_tabular.py            # Tablolu veri model karşılaştırma ve eğitim betiği
├── requirements.txt            # Python bağımlılıkları
└── README.md                   # Proje dokümantasyonu

* ## ⚙️ Yerel Ortamda Çalıştırma
Projeyi kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyebilirsiniz:

1. Projeyi Klonlayın

git clone [https://github.com/KULLANICI_ADIN/liver-diagnostic-app.git](https://github.com/KULLANICI_ADIN/liver-diagnostic-app.git)
cd liver-diagnostic-app

2. Bağımlılıkları Yükleyin

pip install -r requirements.txt

3. Uygulamayı Başlatın

python app.py

* ## 📊 Model Eğitim Süreci

Tablo Verileri: Eksik değerler için medyan doldurma ve hasta cinsiyeti için kategorik dönüşüm uygulanarak en iyi performans gösteren model optimize edilmiştir.

Görüntü Verileri: Sınıf eşleştirmeleri (['Benign', 'Malignant', 'Normal']) sabitlenmiş ve adil/doğru tahmin dağılımları sağlamak için otomatik sınıf ağırlığı dengelemesi (class-weight balancing) ile özel dizin yığın yüklemesi kullanılarak işlenmiştir.

* ## 📝 Lisans

Bu proje açık kaynaklıdır ve MIT Lisansı altında sunulmuştur.


