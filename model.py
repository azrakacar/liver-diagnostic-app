import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import os

print("1. Veri seti yükleniyor ve ön işleme yapılıyor...")
dataset_path = 'dataset/Indian Liver Patient Dataset (ILPD).csv'
df = pd.read_csv(dataset_path)

# Cinsiyet sütununu sayıya çevirme
if 'gender' in df.columns:
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0, 'male': 1, 'female': 0})

# Eksik değerleri medyan ile doldurma
df.fillna(df.median(numeric_only=True), inplace=True)

# Özellikler (X) ve Hedef (y) ayrımı
X = df.drop(columns=['is_patient'])
y = df['is_patient']

# Eğitim ve Test olarak bölme (%80 eğitim, %20 test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n2. 4 Farklı Model Eğitiliyor ve Test Ediliyor...")

# Kullanacağımız 4 farklı model (Gradient Boosting yerine KNN eklendi)
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Support Vector Machine (SVM)": SVC(probability=True, random_state=42)
}

model_scores = {}
best_model = None
best_accuracy = 0
best_model_name = ""

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    model_scores[name] = acc
    print(f"-> {name} Doğruluk Oranı: % {acc * 100:.2f}")
    
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_model_name = name

print(f"\n🏆 En Başarılı Model: {best_model_name} (% {best_accuracy * 100:.2f})")

print("\n3. Görselleştirmeler Hazırlanıyor (Isı Haritaları)...")
plt.figure(figsize=(14, 6))

# --- Grafik 1: 4 Modelin Başarı Karşılaştırması ---
plt.subplot(1, 2, 1)
model_names = list(model_scores.keys())
accuracies = [v * 100 for v in model_scores.values()]
sns.barplot(x=accuracies, y=model_names, palette="magma")
plt.title("4 Farklı Modelin Başarı Karşılaştırma Grafiği")
plt.xlabel("Doğruluk Oranı (%)")
plt.ylabel("Modeller")
plt.xlim(0, 100)

# --- Grafik 2: En İyi Modelin Isı Haritası (Confusion Matrix) ---
plt.subplot(1, 2, 2)
best_y_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, best_y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title(f"En İyi Model ({best_model_name}) Isı Haritası")
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek Değer")

plt.tight_layout()
plt.show()

print("\n4. En iyi model kaydediliyor...")
os.makedirs('saved_models', exist_ok=True)
joblib.dump(best_model, 'saved_models/liver_tabular_model.pkl')
print(f"En başarılı model ({best_model_name}) 'saved_models/liver_tabular_model.pkl' olarak kaydedildi!")