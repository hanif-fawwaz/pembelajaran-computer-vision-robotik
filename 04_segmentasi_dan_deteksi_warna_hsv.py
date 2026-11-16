"""
=============================================================================
Modul 04: Segmentasi Warna dan Deteksi Objek Berdasarkan Warna (HSV)
=============================================================================
Deskripsi:
Program ini adalah teknik paling populer dalam Robotika Vision (Line follower warna,
ball tracking, marker tracking):
1. Mengonversi citra BGR ke ruang warna HSV
2. Menentukan rentang ambang batas bawah dan atas warna (Lower & Upper HSV bounds)
3. Membuat masker biner citra (cv2.inRange)
4. Membersihkan noise pada masker menggunakan operasi morfologi (Morphological Opening & Closing)
5. Mengisolasi objek target menggunakan Bitwise AND (cv2.bitwise_and)
6. Menemukan kontur, menghitung luas area, dan titik tengah (Centroid / koordinat X, Y)
7. Menggambar Bounding Box dan titik koordinat untuk kendali robot

Penulis: Pembelajaran Robotik & Vision
=============================================================================
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_dir, "images", "sample.jpg")

    # 1. Membaca gambar
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print("[ERROR] File gambar tidak ditemukan!")
        return

    # 2. Konversi ruang warna dari BGR ke HSV
    # Di OpenCV: H berkisar 0-179, S berkisar 0-255, V berkisar 0-255
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # =========================================================================
    # MENENTUKAN RENTANG WARNA TARGET (Contoh: Objek Berwarna Biru)
    # =========================================================================
    # Panduan umum nilai Hue di OpenCV:
    # Merah: 0-10 & 160-179, Kuning: 20-35, Hijau: 35-85, Biru: 90-130
    lower_blue = np.array([90, 80, 80], dtype=np.uint8)
    upper_blue = np.array([130, 255, 255], dtype=np.uint8)

    # 3. Membuat masker biner: piksel dalam rentang menjadi 255 (putih), sisanya 0 (hitam)
    mask = cv2.inRange(img_hsv, lower_blue, upper_blue)

    # 4. Operasi Morfologi (Noise Filtering)
    # Membersihkan bintik-bintik kecil (noise) dan menutup celah kosong pada objek
    kernel = np.ones((5, 5), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Menghilangkan noise luar
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel) # Menutup lubang dalam

    # 5. Mengisolasi objek warna menggunakan Bitwise AND
    # Hanya bagian putih pada mask_clean yang akan mempertahankan warna aslinya
    segmented_result = cv2.bitwise_and(img_bgr, img_bgr, mask=mask_clean)

    # =========================================================================
    # DETEKSI POSISI OBJEK (CONTOURS & CENTROID UNTUK ROBOT)
    # =========================================================================
    # Mencari kontur (garis tepi luar) dari masker objek
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_tracking = img_bgr.copy()
    print(f"\n[INFO] Jumlah kontur warna biru ditemukan: {len(contours)}")

    for idx, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        # Filter ukuran objek (abaikan jika terlalu kecil / noise)
        if area > 500:
            # 1. Bounding Box (Kotak pembatas objek)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img_tracking, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 2. Centroid (Titik Pusat Koordinat Objek)
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Gambar titik merah di tengah objek
                cv2.circle(img_tracking, (cx, cy), 6, (0, 0, 255), -1)
                
                # Tulis label koordinat
                label = f"Target (X:{cx}, Y:{cy})"
                cv2.putText(img_tracking, label, (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                
                print(f" -> Objek #{idx + 1}: Koordinat Pusat = ({cx}, {cy}), Luas Area = {area:.0f} px")

    # =========================================================================
    # VISUALISASI HASIL DENGAN MATPLOTLIB
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Modul 04: Segmentasi Warna dan Deteksi Objek (HSV Color Tracking)", fontsize=15)

    # 1. Citra Asli (RGB)
    axes[0, 0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("1. Citra Asli Input")
    axes[0, 0].axis("off")

    # 2. Masker Biner Hasil Thresholding HSV
    axes[0, 1].imshow(mask_clean, cmap="gray")
    axes[0, 1].set_title("2. Binary Mask (Objek Biru = Putih)")
    axes[0, 1].axis("off")

    # 3. Hasil Segmentasi (Hanya Objek Target yang Berwarna)
    axes[1, 0].imshow(cv2.cvtColor(segmented_result, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title("3. Segmented Result (Bitwise AND)")
    axes[1, 0].axis("off")

    # 4. Deteksi Objek & Centroid untuk Robotik
    axes[1, 1].imshow(cv2.cvtColor(img_tracking, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title("4. Robot Tracking View (Bounding Box & Koordinat)")
    axes[1, 1].axis("off")

    plt.tight_layout()
    output_path = os.path.join(current_dir, "images", "hasil_deteksi_warna.png")
    plt.savefig(output_path)
    print(f"[INFO] Visualisasi deteksi warna disimpan ke: {output_path}")

    print("\n[PETUNJUK] Tutup jendela matplotlib yang muncul untuk menyelesaikan program.")
    plt.show()

if __name__ == "__main__":
    main()

