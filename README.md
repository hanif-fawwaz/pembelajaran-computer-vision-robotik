# 🤖 Modul Pembelajaran Computer Vision & OpenCV untuk Robotika

Selamat datang di repositori pembelajaran **Computer Vision Dasar hingga Aplikasi Robotika**.  
Repositori ini dirancang dari nol, terstruktur secara bertahap (12 Modul Lengkap), ramah pemula, dan langsung dapat dipraktikkan untuk membangun sistem penglihatan robot otonom (*autonomous robot vision*).

Setiap modul dilengkapi dengan kode praktikum berpenjelasan detail, visualisasi hasil, aset simulasi otomatis, serta **tautan referensi ilmiah dan dokumentasi resmi** agar Anda dapat mendalami teorinya secara menyeluruh.

---

## 🎯 Silabus Kurikulum Lengkap (Modul 01 - 12)

| File Modul | Topik Utama | Konsep Kunci |
|---|---|---|
| **`01_pengenalan_dan_membaca_citra.py`** | Pengenalan Citra Digital | `cv2.imread`, piksel, resolusi, `cv2.imshow`, `cv2.imwrite` |
| **`02_struktur_pixel_dan_warna.py`** | Ruang Warna & Piksel | BGR vs RGB, format HSV, split saluran (B, G, R / H, S, V) |
| **`03_grayscale_histogram_dan_thresholding.py`** | Histogram & Thresholding | Grayscale, plot histogram, Binary, Otsu, Adaptive Thresholding |
| **`04_segmentasi_dan_deteksi_warna_hsv.py`** | Color Segmentation & Tracking | Masking HSV, morfologi noise, bitwise AND, Contours, Centroid (X,Y) |
| **`05_blur_filtering_dan_canny_edge_detection.py`** | Filtering, Blurring & Edge Detection | Gaussian/Median/Bilateral Blur, Sobel, Laplacian, Canny Edge Detection |
| **`06_kontur_morfologi_dan_deteksi_bentuk.py`** | Kontur, Morfologi & Deteksi Bentuk | Operasi Opening/Closing, Luas (Area), Keliling, Bounding Box, Aproksimasi Bentuk Geometri |
| **`07_operasi_geometri_dan_transformasi_spasial.py`** | Operasi Geometri & Transformasi Spasial | Interpolasi Resize, Cropping ROI, Translasi, Rotasi Sudut, Perspective Warp (Bird's Eye View) |
| **`08_pemrosesan_video_dan_webcam_realtime.py`** | Video & Kamera Real-Time | VideoCapture, Looping Stream, HUD Overlay, Moving Average FPS Counter, Interactive Hotkeys |
| **`09_deteksi_garis_dan_line_tracking_hough.py`** | Deteksi Garis & Line Follower | HoughLinesP, Masker Trapesium ROI, Pemisahan Marka Kiri/Kanan, Steering Angle, Offset Deviasi |
| **`10_deteksi_lingkaran_dan_bola_hough_circle.py`** | Deteksi Bola & Lingkaran | HoughCircles (HOUGH_GRADIENT), Median Blur, Radius Filter, Estimasi Jarak Kamera Fisik |
| **`11_deteksi_gerakan_background_subtraction.py`** | Deteksi Gerakan (Motion Sentry) | BackgroundSubtractorMOG2, Shadow Elimination, Frame Differencing, Motion Security Alarm |
| **`12_capstone_robot_vision_controller_pid.py`** | **Capstone Project: Robot Vision Controller** | **Integrasi End-to-End Pipeline, Pengendali PID Kemudi & Throttle, Protokol Serial UART, Kokpit HUD** |

---

## 📂 Struktur Direktori Repositori

```text
dasar_vision_opencv/
├── README.md                                           # Dokumentasi dan panduan praktikum (file ini)
├── requirements.txt                                    # Dependensi pustaka Python
├── images/                                             # Aset gambar uji, video simulasi & visualisasi
│   ├── sample.jpg                                      # Gambar sampel 4-kuadran warna
│   ├── shapes_sample.jpg                               # Gambar sampel bentuk geometri & derau
│   ├── perspective_sample.jpg                          # Gambar sampel dokumen miring (perspektif)
│   ├── track_line_sample.jpg                           # Gambar sampel lintasan jalan marka aspal
│   ├── circles_sample.jpg                              # Gambar sampel bola & rintangan geometris
│   ├── sample_video.mp4                                # Video simulasi bola bergerak untuk uji stream
│   ├── motion_sample.mp4                               # Video simulasi objek melintas untuk deteksi gerak
│   ├── capstone_tracking_sample.mp4                    # Video simulasi manuver target dinamis capstone
│   ├── sample_grayscale.png                            # Hasil konversi citra grayscale
│   ├── perbandingan_ruang_warna.png                    # Hasil visualisasi komparasi channel
│   ├── hasil_thresholding.png                          # Hasil perbandingan thresholding & histogram
│   ├── hasil_deteksi_warna.png                         # Hasil tracking objek & koordinat centroid
│   ├── hasil_filtering_dan_canny.png                   # Hasil komparasi blur, sobel & Canny edges
│   ├── hasil_deteksi_bentuk.png                        # Hasil klasifikasi bentuk & bounding box
│   ├── hasil_transformasi_geometri.png                 # Hasil komparasi resize, rotasi, & perspective warp
│   ├── hasil_line_tracking.png                         # Hasil evaluasi navigasi garis lintasan robot
│   ├── hasil_hough_circles.png                         # Hasil deteksi lingkaran & estimasi jarak bola
│   ├── hasil_deteksi_gerakan.png                       # Hasil deteksi gerak side-by-side MOG2
│   └── hasil_capstone_controller.png                   # Hasil kokpit kontroler navigasi robot PID
├── 01_pengenalan_dan_membaca_citra.py                  # Modul 01
├── 02_struktur_pixel_dan_warna.py                      # Modul 02
├── 03_grayscale_histogram_dan_thresholding.py          # Modul 03
├── 04_segmentasi_dan_deteksi_warna_hsv.py              # Modul 04
├── 05_blur_filtering_dan_canny_edge_detection.py       # Modul 05
├── 06_kontur_morfologi_dan_deteksi_bentuk.py           # Modul 06
├── 07_operasi_geometri_dan_transformasi_spasial.py     # Modul 07
├── 08_pemrosesan_video_dan_webcam_realtime.py          # Modul 08
├── 09_deteksi_garis_dan_line_tracking_hough.py         # Modul 09
├── 10_deteksi_lingkaran_dan_bola_hough_circle.py       # Modul 10
├── 11_deteksi_gerakan_background_subtraction.py        # Modul 11
└── 12_capstone_robot_vision_controller_pid.py          # Modul 12 (Capstone)
```

---

## ⚙️ Persiapan & Instalasi

Pastikan komputer/laptop Anda telah terpasang **Python 3.8+**. Jalankan perintah instalasi berikut:

```bash
# 1. Masuk ke direktori modul
cd dasar_vision_opencv

# 2. Instal pustaka yang dibutuhkan
pip install -r requirements.txt
```

Dependensi:
- `opencv-python`: Pustaka inti Computer Vision
- `numpy`: Pemrosesan komputasi matriks dan array citra
- `matplotlib`: Visualisasi grafik, histogram, dan komparasi multi-plot

---

## 📖 Ringkasan Teori & Referensi Rinci Per Modul

Berikut adalah pembahasan konsep inti beserta **sumber dokumentasi resmi, buku teks, dan paper ilmiah** untuk memperdalam materi setiap modul:

### 🔹 Modul 01: Pengenalan Citra Digital, Piksel, dan I/O OpenCV
* **Konsep Inti:** Citra digital sebagai matriks angka multidimensi (NumPy Array), resolusi ($W \times H$), koordinat kartesius terbalik OpenCV (sumbu Y mengarah ke bawah), serta fungsi I/O dasar (`imread`, `imshow`, `imwrite`, `waitKey`).
* **📚 Sumber Belajar Lanjutan:**
  * [Dokumentasi Resmi OpenCV: Getting Started with Images](https://docs.opencv.org/4.x/db/deb/tutorial_display_image.html)
  * [PyImageSearch: Loading and Displaying Images with OpenCV](https://pyimagesearch.com/2021/01/20/opencv-load-image-cv2-imread/)
  * *Buku Referensi:* "Digital Image Processing" oleh Rafael C. Gonzalez & Richard E. Woods (Bab 2: Digital Image Fundamentals).

---

### 🔹 Modul 02: Ruang Warna (BGR, RGB, HSV) & Channel Splitting
* **Konsep Inti:** Alasan historis OpenCV menggunakan format BGR; Ruang warna HSV yang memisahkan kromatisitas/warna murni (*Hue*) dari pencahayaan (*Value*); Channel splitting dan manipulasi intensitas per saluran warna.
* **📚 Sumber Belajar Lanjutan:**
  * [Dokumentasi Resmi OpenCV: Changing Colorspaces (cvtColor)](https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html)
  * [LearnOpenCV: Color Spaces in OpenCV (RGB, HSV, LAB)](https://learnopencv.com/color-spaces-in-opencv-cpp-python/)
  * [Wikipedia: HSL and HSV Color Models](https://en.wikipedia.org/wiki/HSL_and_HSV)

---

### 🔹 Modul 03: Grayscale, Histogram Intensitas, dan Teknik Thresholding
* **Konsep Inti:** Distribusi intensitas cahaya via Histogram; Simple Binary Thresholding; Algoritma Otsu untuk mencari nilai ambang optimal otomatis berbasis variansi inter-class; Adaptive Gaussian Thresholding untuk kondisi pencahayaan tidak merata/bayangan.
* **📚 Sumber Belajar Lanjutan:**
  * [Dokumentasi Resmi OpenCV: Image Thresholding & Otsu Binarization](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html)
  * [Dokumentasi Resmi OpenCV: Histograms - Find, Plot, Analyze](https://docs.opencv.org/4.x/d1/db7/tutorial_py_histogram_begins.html)
  * *Paper Orisinal:* Nobuyuki Otsu (1979), *"A Threshold Selection Method from Gray-Level Histograms"*, IEEE Transactions on Systems, Man, and Cybernetics.

---

### 🔹 Modul 04: Segmentasi Warna HSV dan Pelacakan Objek (Centroid)
* **Konsep Inti:** Isolasi rentang warna (`cv2.inRange`); Operasi logika piksel `cv2.bitwise_and`; Ekstraksi kontur luar (`cv2.findContours`); Momen citra (Image Moments) untuk menghitung koordinat titik pusat massa ($cx = M_{10}/M_{00}, cy = M_{01}/M_{00}$).
* **📚 Sumber Belajar Lanjutan:**
  * [Dokumentasi Resmi OpenCV: Image Moments & Contours](https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html)
  * [PyImageSearch: Ball Tracking with OpenCV](https://pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/)
  * [LearnOpenCV: Find Center of Blob (Centroid) using OpenCV](https://learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/)

---

### 🔹 Modul 05: Filtering, Blurring, Gradien Sobel & Canny Edge Detection
* **Konsep Inti:** Konvolusi kernel 2D; Perbandingan Gaussian Blur vs Median Blur (efektif untuk *salt-and-pepper noise*) vs Bilateral Filter (mempertahankan ketajaman tepi); Operator turunan spasial Sobel & Laplacian; 4 Tahap Algoritma Canny (*Gaussian Filter, Gradient Intensity, Non-Maximum Suppression, Hysteresis Thresholding*).
* **📚 Sumber Belajar Lanjutan:**
  * [Dokumentasi Resmi OpenCV: Smoothing Images](https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html)
  * [Dokumentasi Resmi OpenCV: Canny Edge Detection](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html)
  * *Paper Orisinal:* John Canny (1986), *"A Computational Approach to Edge Detection"*, IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI).

---

### 🔹 Modul 06: Kontur, Operasi Morfologi, Bounding Box, dan Shape Detection
* **Konsep Inti:** Morfologi matematika (Opening = Erosi dilanjut Dilatasi untuk melenyapkan noise; Closing = Dilatasi dilanjut Erosi untuk menutup lubang objek); Bounding Box tegak vs Bounding Box berotasi (`minAreaRect`) untuk sudut orientasi capit robot; Aproksimasi kurva poligon Douglas-Peucker (`cv2.approxPolyDP`) dan metrik circularity ($\frac{4\pi A}{P^2}$).
* **📚 Sumber Belajar Lanjutan:**
  * [Dokumentasi Resmi OpenCV: Morphological Transformations](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html)
  * [Dokumentasi Resmi OpenCV: Structural Analysis & Shape Descriptors](https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html)
  * [PyImageSearch: OpenCV Shape Detection](https://pyimagesearch.com/2016/02/08/opencv-shape-detection/)

---

### 🔹 Modul 07: Operasi Geometri & Transformasi Spasial (Bird's Eye View)
* **Konsep Inti:** Algoritma interpolasi citra (`INTER_AREA` untuk downscaling cepat tanpa moiré vs `INTER_CUBIC` untuk perbesaran halus); Pemotongan Region of Interest (ROI); Matriks Affine $2 \times 3$ (translasi & rotasi sudut bebas); Matriks Homografi Perspektif $3 \times 3$ (`warpPerspective`) untuk mengubah sudut pandang miring kamera menjadi tegak lurus dari atas (*Inverse Perspective Mapping*).
* **📚 Sumber Belajar Lanjutan:**
  * [Dokumentasi Resmi OpenCV: Geometric Transformations of Images](https://docs.opencv.org/4.x/da/d54/tutorial_py_geometric_transformations.html)
  * [PyImageSearch: 4 Point OpenCV getPerspectiveTransform Example](https://pyimagesearch.com/2014/08/25/4-point-opencv-getperspectivetransform-example/)
  * *Buku Referensi:* "Multiple View Geometry in Computer Vision" oleh Richard Hartley & Andrew Zisserman (Bab 2: Projective Geometry and 2D Homographies).

---

### 🔹 Modul 08: Pemrosesan Video & Kamera Webcam Real-Time dengan FPS Counter
* **Konsep Inti:** Pengambilan aliran frame video kontinu (`cv2.VideoCapture`); Perhitungan FPS instan dan rata-rata bergerak (*moving average*); Pengukuran latensi komputasi per frame (ms) agar tidak memperlambat kendali aktuator robot ($>30\text{ FPS}$); Heads-Up Display (HUD) overlay teks dan keyboard events.
* **📚 Sumber Belajar Lanjutan:**
  * [Dokumentasi Resmi OpenCV: Getting Started with Videos](https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html)
  * [PyImageSearch: Increasing webcam FPS with Python and OpenCV](https://pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/)
  * [LearnOpenCV: How to find FPS of a video or camera in OpenCV](https://learnopencv.com/how-to-find-frame-rate-or-frames-per-second-fps-in-opencv-python-cpp/)

---

### 🔹 Modul 09: Deteksi Garis (Line Tracking) Menggunakan Hough Line Transform
* **Konsep Inti:** Teori ruang akumulator Hesse ($r = x\cos\theta + y\sin\theta$); Probabilistic Hough Lines (`cv2.HoughLinesP`); Masking poligon trapesium jalan; Klasifikasi garis marka kiri vs kanan berbasis gradien kemiringan ($m = \Delta y / \Delta x$); Estimasi sudut kemudi (*Steering Angle*) dan deviasi offset untuk mobil otonom / AGV.
* **📚 Sumber Belajar Lanjutan:**
  * [Dokumentasi Resmi OpenCV: Hough Line Transform](https://docs.opencv.org/4.x/d9/db0/tutorial_hough_lines.html)
  * [Towards Data Science: Finding Lane Lines Simple Pipeline (Udacity Self-Driving Car Nanodegree)](https://towardsdatascience.com/finding-lane-lines-simple-pipeline-f191a3294f33)
  * *Paper Orisinal:* Richard O. Duda & Peter E. Hart (1972), *"Use of the Hough Transformation to Detect Lines and Curves in Pictures"*, Communications of the ACM.

---

### 🔹 Modul 10: Deteksi Lingkaran & Bola Menggunakan Hough Circle Transform
* **Konsep Inti:** Persamaan lingkaran $(x - a)^2 + (y - b)^2 = r^2$; Metode 2-1 Hough Gradient (`cv2.HOUGH_GRADIENT`) yang memanfaatkan gradien tepi untuk memprediksi pusat bundar; Filter parameter `param1`, `param2`, dan radius range; Model kamera lubang jarum (*Pinhole Camera Model*) untuk mengestimasi jarak fisik bola dalam centimeter ($cm$).
* **📚 Sumber Belajar Lanjutan:**
  * [Dokumentasi Resmi OpenCV: Hough Circle Transform](https://docs.opencv.org/4.x/d4/d70/tutorial_hough_circle.html)
  * [PyImageSearch: Find Distance from Camera to Object/Marker using OpenCV](https://pyimagesearch.com/2015/01/19/find-distance-camera-objectmarker-using-python-opencv/)
  * [LearnOpenCV: Circle Detection using OpenCV Python / C++](https://learnopencv.com/hough-transform-with-opencv-c-plus-plus-python/)

---

### 🔹 Modul 11: Deteksi Gerakan (Motion Detection & Background Subtraction MOG2)
* **Konsep Inti:** Frame Differencing dasar ($|\text{Frame}_t - \text{Frame}_{t-1}|$); Model adaptif Mixture of Gaussians (MOG2); Eliminasi bayangan dinamis (*Shadow Rejection* nilai 127); Pemfilteran noise kontur gerak untuk sistem alarm keamanan sentri robot (*Robot Security Alarm*).
* **📚 Sumber Belajar Lanjutan:**
  * [Dokumentasi Resmi OpenCV: How to Use Background Subtraction Methods](https://docs.opencv.org/4.x/d1/dc5/tutorial_background_subtraction.html)
  * *Paper Orisinal:* Zoran Zivkovic (2004), *"Improved adaptive Gaussian mixture model for background subtraction"*, International Conference on Pattern Recognition (ICPR).
  * [PyImageSearch: Basic motion detection and tracking with Python and OpenCV](https://pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/)

---

### 🔹 Modul 12 (Capstone): Robot Vision Controller Terpadu dengan PID & Protokol Serial
* **Konsep Inti:** Desain arsitektur *vision-guided robotics*; Algoritma kendali umpan balik PID ($K_p, K_i, K_d$) dengan *anti-windup clamping*; Penerapan zona toleransi (*deadzone*) untuk meredam jitter motor; Desain struktur paket komunikasi serial UART (`<CMD, STEER, THROTTLE, FLAG>`) untuk mikrokontroler (Arduino/ESP32/ROS); Tampilan kokpit HUD lengkap.
* **📚 Sumber Belajar Lanjutan:**
  * [ArduPilot / PID Control Theory: Understanding PID Controllers in Robotics](https://ardupilot.org/copter/docs/tuning.html)
  * [MATLAB & Simulink: Understanding PID Control, Part 1: What is PID?](https://www.youtube.com/watch?v=UR0hOmjaHp0)
  * [PyImageSearch: Pan/Tilt Object Tracking with Python and OpenCV](https://pyimagesearch.com/2019/04/01/pan-tilt-face-tracking-with-a-raspberry-pi-and-opencv/)
  * [Arduino Official Guide: Serial Communication Protocol & String Parsing](https://docs.arduino.cc/language-reference/en/functions/communication/serial/)

---

## 🚀 Cara Menjalankan Seluruh Modul

Jalankan skrip modul yang ingin dipelajari langsung dari terminal:

```bash
# Modul 01: Membaca, menampilkan, dan menyimpan citra
python 01_pengenalan_dan_membaca_citra.py

# Modul 02: Struktur piksel, BGR vs RGB, dan eksplorasi HSV
python 02_struktur_pixel_dan_warna.py

# Modul 03: Grayscale, analisis histogram, dan teknik thresholding
python 03_grayscale_histogram_dan_thresholding.py

# Modul 04: Color segmentation, masking, deteksi kontur, dan tracking koordinat
python 04_segmentasi_dan_deteksi_warna_hsv.py

# Modul 05: Filtering noise, blurring, gradien Sobel, dan Canny Edge Detection
python 05_blur_filtering_dan_canny_edge_detection.py

# Modul 06: Kontur, operasi morfologi, bounding box, dan klasifikasi bentuk geometri
python 06_kontur_morfologi_dan_deteksi_bentuk.py

# Modul 07: Operasi geometri, resize interpolasi, rotasi, dan perspective warp (Bird's Eye View)
python 07_operasi_geometri_dan_transformasi_spasial.py

# Modul 08: Pemrosesan video & kamera webcam real-time dengan HUD dan FPS counter
python 08_pemrosesan_video_dan_webcam_realtime.py

# Modul 09: Deteksi garis lintasan marka jalan dan kalkulasi sudut kemudi robot
python 09_deteksi_garis_dan_line_tracking_hough.py

# Modul 10: Deteksi bola/lingkaran dan estimasi jarak fisik objek
python 10_deteksi_lingkaran_dan_bola_hough_circle.py

# Modul 11: Deteksi gerakan penyusup & background subtraction MOG2
python 11_deteksi_gerakan_background_subtraction.py

# Modul 12 (Capstone): Sistem kontroler penglihatan robot terpadu dengan PID dan protokol serial
python 12_capstone_robot_vision_controller_pid.py
```

> 💡 **Fitur Plug-and-Play Otomatis:** Semua modul video (08, 11, dan 12) memiliki fitur fallback cerdas. Jika webcam komputer tidak aktif/tidak ada, program otomatis memutar video simulasi yang disediakan di folder `images/` sehingga siapa pun yang mendownload repositori GitHub Anda dapat langsung mempraktikkannya tanpa error!

---

## 👥 Kontributor & Acknowledgments
- **Hanif Arib Fawwaz** ([@hanif-fawwaz](https://github.com/hanif-fawwaz)) - Inisiator, Pengembang & Desain Kurikulum Modul
- **Antigravity AI (Google DeepMind)** - AI Pair Programmer & Arsitektur Modul Pembelajaran Robotika Vision

