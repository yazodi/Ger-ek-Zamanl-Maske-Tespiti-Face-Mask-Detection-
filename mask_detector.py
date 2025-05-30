import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import datetime
import os

# Model ve cascade dosyasını yükle
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = load_model('mask_detector_model.h5')

# Ekran görüntülerinin kaydedileceği klasör
os.makedirs('screenshots', exist_ok=True)

# Kamera başlat
cap = cv2.VideoCapture(0)

print("[INFO] Kamera başlatıldı. Çıkmak için 'q' tuşuna basın.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Kameradan görüntü alınamadı.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (224, 224))
        face_array = img_to_array(face_resized)
        face_array = np.expand_dims(face_array, axis=0) / 255.0

        # Tahmin et
        try:
            (mask, no_mask) = model.predict(face_array)[0]
        except Exception as e:
            print(f"[ERROR] Tahmin sırasında hata oluştu: {e}")
            continue

        label = "Maskeli" if mask > no_mask else "Maskesiz"
        color = (0, 255, 0) if label == "Maskeli" else (0, 0, 255)

        # Maske yoksa ekran görüntüsünü kaydet
        if label == "Maskesiz":
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"screenshots/nomask_{timestamp}.png"
            cv2.imwrite(filename, frame)

        # Görüntüye kutu ve metin ekle
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow('Maske Tespit Sistemi', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Uygulamadan çıkılıyor...")
        break

cap.release()
cv2.destroyAllWindows()
