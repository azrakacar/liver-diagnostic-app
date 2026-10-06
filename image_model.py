import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import os

print("1. Görüntü veri seti yükleniyor ve boyutlandırılıyor...")
image_size = (128, 128)
batch_size = 32

data_dir = 'dataset/images'

if not os.path.exists(data_dir):
    print(f"Uyarı: '{data_dir}' klasörü bulunamadı.")
else:
    # Sınıf sıralamasını sabitliyoruz
    class_names = ['Benign', 'Malignant', 'Normal']

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=image_size,
        batch_size=batch_size,
        class_names=class_names
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=image_size,
        batch_size=batch_size,
        class_names=class_names
    )

    print("\n2. Sınıf Dengesizliği (Class Imbalance) Analiz Ediliyor ve Ağırlıklar Hesaplanıyor...")
    y_train_labels = []
    for images, labels in train_ds:
        y_train_labels.extend(labels.numpy())
    y_train_labels = np.array(y_train_labels)

    # Otomatik sınıf ağırlığı hesaplama
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_labels),
        y=y_train_labels
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"Hesaplanan Sınıf Ağırlıkları: {class_weight_dict}")

    print("\n3. CNN Modeli Mimarisi Oluşturuluyor...")
    model = models.Sequential([
        layers.Rescaling(1./255, input_shape=(128, 128, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(3, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\n4. Görüntü Modeli Sınıf Ağırlıklarıyla Eğitiliyor...")
    epochs = 15
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weight_dict  # <-- Model artık adil öğrenmeye zorlanıyor!
    )

    print("\n5. Eğitilen Görüntü Modeli Kaydediliyor...")
    os.makedirs('saved_models', exist_ok=True)
    model.save('saved_models/liver_image_model.h5')
    print("Görüntü modeli başarıyla 'saved_models/liver_image_model.h5' olarak kaydedildi!")