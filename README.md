# 😷 Real-Time Face Mask Detection with OpenCV and Keras
### 👁️‍🗨️ Gerçek Zamanlı Maske Tespiti (OpenCV + Keras)

This project detects whether a person is wearing a face mask in real-time using your webcam.  
It uses OpenCV for face detection and a pre-trained Keras deep learning model for mask classification.

Bu proje, kameradan alınan görüntüde kişinin maske takıp takmadığını gerçek zamanlı olarak tespit eder.  
Yüz algılama için OpenCV, sınıflandırma için önceden eğitilmiş bir Keras modeli kullanır.

---

## 📦 Requirements / Gereksinimler

```bash
pip install opencv-python keras numpy


Make sure the following files are in the same folder as your main Python file:
Aşağıdaki dosyaların Python dosyanızla aynı klasörde olduğundan emin olun:

mask_detector_model.h5 – Trained Keras model
buradan indirebilirsiniz
https://huggingface.co/spaces/sonerkar/mask-detector-tr/blob/b613486920b138032e30835a63ff8c4556f2963a/model.h5

haarcascade_frontalface_default.xml – Face detection cascade

mask_detector.py – Main Python code


 How to Run / Nasıl Çalıştırılır?


Webcam will start

Faces will be detected and classified as Maskeli or Maskesiz

If no mask is detected, a screenshot will be saved in the screenshots/ folder

Press q to quit the application


Project Structure / Proje Yapısı

mask-detector/
├── mask_detector.py
├── mask_detector_model.h5
├── haarcascade_frontalface_default.xml
├── screenshots/
└── README.md


 Features / Özellikler
✅ Real-time webcam input

✅ Face detection with OpenCV

✅ Mask classification using a pre-trained deep learning model

✅ Automatic screenshot saving for unmasked individuals

✅ Offline functionality


📷 Screenshot Example / Ekran Görüntüsü Örneği
📌 You can add a real-time screenshot of your project in action here
📌 Buraya uygulamanın çalışır hâlinden bir ekran görüntüsü ekleyebilirsin


👤 Author / Yazar
GitHub: @yazodi

 License / Lisans
This project is open-source and free to use for educational and personal purposes.
Bu proje açık kaynaklıdır ve eğitimsel ya da kişisel amaçlarla serbestçe kullanılabilir.

---

