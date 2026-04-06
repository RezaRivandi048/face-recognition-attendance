# Face Recognition Attendance System

Proyek ini merupakan implementasi sistem absensi mahasiswa otomatis menggunakan metode face recognition berbasis pengolahan citra.

## Fitur
- Deteksi wajah (Haar Cascade)
- Pengenalan wajah (LBPH)
- Absensi otomatis
- Penyimpanan ke CSV

## Cara Menjalankan

1. Ambil dataset:
python src/collect_data.py

2. Training:
python src/train_model.py

3. Jalankan absensi:
python src/recognize.py

## Teknologi
- Python
- OpenCV
- NumPy
