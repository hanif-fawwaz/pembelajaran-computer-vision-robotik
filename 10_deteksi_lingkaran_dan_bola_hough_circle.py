"""
=============================================================================
Modul 10: Deteksi Lingkaran & Bola Menggunakan Hough Circle Transform
=============================================================================
Deskripsi:
Program ini membahas teknik mendeteksi bentuk lingkaran dan objek bola di arena robot:
1. Konsep matematis Hough Circle Transform (Persamaan Lingkaran: $(x - a)^2 + (y - b)^2 = r^2$):
   - Ruang akumulator 3D: Pusat $(a, b)$ dan Radius $r$.
   - Metode 2-1 Hough Gradient (cv2.HOUGH_GRADIENT): Pendekatan cepat berbasis gradien intensitas.
2. Pra-pemrosesan citra khusus Hough Circle:
   - Median Blur (Sangat dianjurkan untuk menghaluskan permukaan bola tanpa merusak tepi).
3. Parameter Kunci cv2.HoughCircles:
   - dp: Rasio kebalikan resolusi akumulator (umumnya 1.0 s/d 1.5).
   - minDist: Jarak minimum antarpusat lingkaran (mencegah deteksi lingkaran ganda/berimpit).
   - param1: Ambang batas atas untuk Canny Edge internal (param1/2 adalah ambang bawah).
   - param2: Ambang voting akumulator (semakin kecil semakin banyak lingkaran terdeteksi/false positive).
   - minRadius & maxRadius: Filter rentang ukuran bola yang ingin dicari (px).
4. Estimasi Jarak Objek (Distance Estimation / Pinhole Camera Model):
   - Menghitung perkiraan jarak kamera ke bola berdasarkan diameter piksel yang terukur:
     $Distance = (Real\_Diameter \times Focal\_Length) / Pixel\_Diameter$
5. Visualisasi Bounding Circle, Titik Pusat $(cx, cy)$, dan Pengujian Objek Palsu (Kotak/Segitiga).

Relevansi Robotik:
- Robot Sepak Bola Beroda/Humanoid (RoboSoccer / KRSBI) untuk mendeteksi bola oranye
- Robot pemungut bola tenis / golf otonom
- Deteksi roda, bearing, atau lubang baut pada sistem inspeksi industri

Penulis: Pembelajaran Robotik & Vision
=============================================================================
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_dir, "images", "circles_sample.jpg")

    # 1. Membaca gambar arena
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print("[ERROR] File 'circles_sample.jpg' tidak ditemukan!")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w, _ = img_rgb.shape

    # 2. Filtering Pra-pemrosesan
    # Median Blur sangat direkomendasikan sebelum HoughCircles untuk mereduksi derau
    # tanpa mengaburkan ketajaman kurva lingkaran
    blurred = cv2.medianBlur(img_gray, 5)

    # =========================================================================
    # DETEKSI LINGKARAN MENGGUNAKAN cv2.HoughCircles
    # =========================================================================
    # cv2.HOUGH_GRADIENT: Algoritma berbasis gradien vektor
    # dp = 1.2: Resolusi akumulator
    # minDist = 50: Minimal jarak antarpusat 50 px agar lingkaran yang sama tidak terdeteksi 2 kali
    # param1 = 100: Ambang batas Canny edge internal
    # param2 = 30: Ambang voting akumulator (kualitas kebundaran)
    # minRadius = 15, maxRadius = 80: Batasan ukuran bola yang dicari
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=60,
        param1=100,
        param2=30,
        minRadius=20,
        maxRadius=70
    )

    img_circles_vis = img_rgb.copy()
    img_target_tracking = img_rgb.copy()

    # Konstanta untuk simulasi estimasi jarak (Pinhole Camera Model)
    # Misal diameter nyata bola tenis meja/golf = 4.0 cm, focal length simulasi = 600 px
    REAL_BALL_DIAMETER_CM = 4.0
    FOCAL_LENGTH_PX = 600.0

    print("\n--- HASIL DETEKSI HOUGH CIRCLE TRANSFORM ---")

    if circles is not None:
        # Konversi koordinat dan radius menjadi integer bulat
        circles_round = np.round(circles[0, :]).astype("int")
        print(f"[INFO] Jumlah objek lingkaran/bola valid terdeteksi: {len(circles_round)}")

        for idx, (x, y, r) in enumerate(circles_round):
            # Perhitungan diameter terukur dalam piksel
            diameter_px = 2 * r

            # Estimasi jarak fisik dari kamera robot ke bola
            # Distance = (Real_Diameter * Focal_Length) / Pixel_Diameter
            estimated_distance_cm = (REAL_BALL_DIAMETER_CM * FOCAL_LENGTH_PX) / float(diameter_px)

            # Hitung offset deviasi sudut terhadap pusat pandangan kamera robot (w // 2)
            offset_x = x - (w // 2)

            print(f" -> Bola #{idx + 1}: Pusat=({x}, {y}) | Radius={r} px | Est. Jarak={estimated_distance_cm:.1f} cm | Offset={offset_x} px")

            # A. Gambar visualisasi lingkaran dan titik tengah pada img_circles_vis
            cv2.circle(img_circles_vis, (x, y), r, (0, 255, 0), 3)       # Lingkaran luar hijau
            cv2.circle(img_circles_vis, (x, y), 4, (255, 0, 0), -1)     # Titik pusat merah

            # B. Gambar HUD navigasi target pada img_target_tracking
            # Bounding box kotak di sekeliling bola
            cv2.rectangle(img_target_tracking, (x - r, y - r), (x + r, y + r), (0, 220, 255), 2)
            cv2.circle(img_target_tracking, (x, y), 5, (0, 0, 255), -1)

            # Label data untuk navigasi robot
            label_id = f"Bola #{idx+1} (R:{r}px)"
            label_dist = f"Jarak: {estimated_distance_cm:.0f}cm"
            cv2.putText(img_target_tracking, label_id, (x - r, y - r - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 2)
            cv2.putText(img_target_tracking, label_dist, (x - r, y - r - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

            # Garis pandu dari dasar tengah kamera menuju bola
            cv2.line(img_target_tracking, (w // 2, h), (x, y), (255, 100, 0), 2)
    else:
        print("[WARNING] Tidak ada lingkaran yang memenuhi kriteria voting param2!")

    # =========================================================================
    # VISUALISASI HASIL DENGAN MATPLOTLIB
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("Modul 10: Deteksi Lingkaran / Bola & Estimasi Jarak (Hough Circle Transform)", fontsize=15)

    # 1. Citra Asli Input
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("1. Citra Input Arena (Ada Bola, Kotak & Segitiga)")
    axes[0, 0].axis("off")

    # 2. Hasil Median Blur Preprocessing
    axes[0, 1].imshow(blurred, cmap="gray")
    axes[0, 1].set_title("2. Median Blur Grayscale (Peredam noise permukaan bola)")
    axes[0, 1].axis("off")

    # 3. Lingkaran Terdeteksi Hough Circle
    axes[1, 0].imshow(img_circles_vis)
    axes[1, 0].set_title("3. Hough Circles Terdeteksi\n(Objek kotak & segitiga berhasil diabaikan!)")
    axes[1, 0].axis("off")

    # 4. HUD Navigasi Robot & Estimasi Jarak Objek
    axes[1, 1].imshow(img_target_tracking)
    axes[1, 1].set_title("4. Output Navigasi Robot (Tracking Target & Estimasi Jarak)")
    axes[1, 1].axis("off")

    plt.tight_layout()
    output_path = os.path.join(current_dir, "images", "hasil_hough_circles.png")
    plt.savefig(output_path)
    print(f"\n[INFO] Gambar visualisasi deteksi lingkaran disimpan ke: {output_path}")

    print("\n[PETUNJUK] Tutup jendela matplotlib yang muncul untuk menyelesaikan program.")
    plt.show()

if __name__ == "__main__":
    main()

