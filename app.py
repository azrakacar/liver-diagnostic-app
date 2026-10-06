import os
import numpy as np
import joblib
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'dataset/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

tabular_model = joblib.load('saved_models/liver_tabular_model.pkl')
image_model = load_model('saved_models/liver_image_model.h5')

CLASSES = ['Benign', 'Malignant', 'Normal']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = float(request.form.get('age', 0))
        gender = float(request.form.get('gender', 0))
        total_bilirubin = float(request.form.get('total_bilirubin', 0))
        direct_bilirubin = float(request.form.get('direct_bilirubin', 0))
        alkaline_phosphotase = float(request.form.get('alkaline_phosphotase', 0))
        alamine_aminotransferase = float(request.form.get('alamine_aminotransferase', 0))
        aspartate_aminotransferase = float(request.form.get('aspartate_aminotransferase', 0))
        total_proteins = float(request.form.get('total_proteins', 0))
        albumin = float(request.form.get('albumin', 0))
        albumin_globulin_ratio = float(request.form.get('albumin_globulin_ratio', 0))
        
        features = [[
            age, gender, total_bilirubin, direct_bilirubin, 
            alkaline_phosphotase, alamine_aminotransferase, 
            aspartate_aminotransferase, total_proteins, 
            albumin, albumin_globulin_ratio
        ]]
        
        tabular_pred = tabular_model.predict(features)[0]
        
        file = request.files['image']
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        img = image.load_img(filepath, target_size=(128, 128)) 
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0) / 255.0
        
        image_preds = image_model.predict(x)
        image_pred_class = CLASSES[np.argmax(image_preds)]
        image_confidence = float(np.max(image_preds)) * 100
        
        # Detaylı Belirtiler ve Tedavi Yöntemleri İçeren Matris
        if image_pred_class == 'Normal' and tabular_pred == 0:
            durum_tipi = "success"
            baslik = "Mükemmel Sağlık Durumu: Karaciğeriniz Çok Temiz!"
            olasi_durum = "Sağlıklı Karaciğer Dokusu"
            aciklama = "Kan tahlili enzimleriniz ve ultrason görüntü taramanız tamamen normal aralıktadır. Karaciğer parankim yapısında herhangi bir bozulma veya lezyon yoktur."
            belirtiler = "Sağlıklı bir karaciğerde anormal bir şikayet görülmez. Yüksek enerji seviyesi, düzenli sindirim ve ideal metabolizma hızı görülür."
            tedavi_yontemleri = "Herhangi bir tedaviye ihtiyaç yoktur. Korunma amacıyla bol su tüketimi, sebze/meyve ağırlıklı dengeli beslenme ve düzenli fiziksel aktivite önerilir."
            motivasyon = "Harika bir yaşam tarzına sahip olduğunuz her halinizden belli! Sağlığınızı bu şekilde korumaya devam edin. 🎉"
        elif image_pred_class == 'Benign':
            durum_tipi = "warning"
            baslik = "İyi Huylu (Benign) Bulgular / Takip Gerektiren Durumlar"
            olasi_durum = "Karaciğer Yağlanması (Steatoz) veya Basit Kist İhtimali"
            aciklama = "Görüntü ve tahlil kombinasyonunda kanser özelliği taşımayan, iyi huylu yapı bulguları saptanmıştır. Genellikle hafif karaciğer yağlanması (hepatosteatoz) veya iyi huylu kistlere işaret eder."
            belirtiler = "Sağ üst karın bölgesinde hafif dolgunluk veya künt ağrı, kronik hafif yorgunluk, halsizlik ve bazen kilo artışı ile kendini gösterebilir."
            tedavi_yontemleri = "Yaşam tarzı değişiklikleri esastır. Şeker, unlu mamuller, alkol ve doymuş yağlardan kaçınmak, düzenli kardiyo egzersizleri yapmak ve Gastroenteroloji kontrolünde dönemsel ultrason takibi yaptırmak."
            motivasyon = "Korkmanıza hiç gerek yok; bu durumlar tıp dünyasında çok yaygındır ve disiplinli bir yaşam tarzıyla tamamen kontrol altına alınabilir. Moralinizi yüksek tutun! 💙"
        else:
            durum_tipi = "danger"
            baslik = "Kritik Bulgular: Uzman İncelemesi ve İleri Tetkik Şart"
            olasi_durum = "Hepatit, Fibrozis, Siroz veya Malign (Kötü Huylu) Kitle Şüphesi"
            aciklama = "Yapay zeka modellerimiz, karaciğer dokusunda ve kan değerlerinde normal dışı, ileri düzey takip ve uzman değerlendirmesi gerektiren ciddi risk faktörleri tespit etmiştir."
            belirtiler = "Aşırı ve geçmeyen halsizlik, ciltte ve göz akında sararma (sarılık), koyu renkli idrar, açık renkli dışkı, karında şişlik (asit) ve ani iştah/kilo kaybı."
            tedavi_yontemleri = "Zaman kaybetmeden Dahiliye veya Gastroenteroloji uzmanına başvurulmalıdır. Nedene bağlı olarak antiviral ilaçlar, hedeflenmiş medikal tedaviler, ileri tetkik (biyopsi/MR) veya cerrahi değerlendirme süreçleri uygulanır."
            motivasyon = "Unutmayın ki modern tıp ve erken teşhis hayat kurtarır. Bu yolda asla yalnız değilsiniz; güçlü durun ve sağlığınız için en doğru adımı hemen atın. Yanınızdayız! 🌟"

        sonuc_ozeti = {
            'tip': durum_tipi,
            'baslik': baslik,
            'olasi_durum': olasi_durum,
            'aciklama': aciklama,
            'belirtiler': belirtiler,
            'tedavi_yontemleri': tedavi_yontemleri,
            'motivasyon': motivasyon,
            'tabular_sonuc': "Riskli / Fonksiyon Bozukluğu Şüphesi" if tabular_pred == 1 else "Normal Sınırlar",
            'image_sonuc': image_pred_class,
            'image_guven': f"{image_confidence:.2f}%"
        }
        
        return render_template('result.html', sonuc=sonuc_ozeti)
        
    except Exception as e:
        return f"Bir hata oluştu: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
    