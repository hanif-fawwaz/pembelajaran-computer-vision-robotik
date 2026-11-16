"""
=============================================================================
Modul 05: Filtering, Blurring, dan Edge Detection (Sobel, Laplacian, Canny)
=============================================================================
Deskripsi:
Program ini membahas proses filtering citra dan deteksi tepi objek:
1. Pemulusan citra / reduksi derau (Noise Reduction):
   - Averaging Blur (cv2.blur)
   - Gaussian Blur (cv2.GaussianBlur)
   - Median Blur (cv2.medianBlur - sangat efektif untuk salt-and-pepper noise)
   - Bilateral Filter (cv2.bilateralFilter - menghaluskan permukaan tanpa mengaburkan tepi)
2. Deteksi gradien dan tepi dasar:
   - Sobel X & Sobel Y (gradien horizontal dan vertikal)
   - Sobel Gabungan (Magnitude gradient)
   - Laplacian (Turunan kedua untuk mendeteksi perubahan intensitas tajam)
3. Canny Edge Detection (Standar emas deteksi tepi multirata):
   - Gaussian smoothing internal
   - Gradient calculation
   - Non-maximum suppression
   - Hysteresis thresholding (minVal & maxVal)

Relevansi Robotik:
- Menemukan garis batas lintasan robot (lane boundary)
- Deteksi kontur geometri rintangan/obstacle
- Preprocessing sebelum ekstraksi fitur bentuk atau Hough Transform

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

    # 1. Membaca citra input
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print("[ERROR] File gambar tidak ditemukan!")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # =========================================================================
    # BAGIAN 1: TEKNIK BLURRING & SMOOTHING (FILTERING REDUKSI NOISE)
    # =========================================================================
    # A. Averaging Blur: Menghitung rata-rata nilai piksel tetangga dalam kernel (ksize)
    blur_avg = cv2.blur(img_rgb, (7, 7))

    # B. Gaussian Blur: Memberikan bobot lebih besar pada piksel tengah berdasarkan kurva lonceng Gaussian
    blur_gaussian = cv2.GaussianBlur(img_rgb, (7, 7), sigmaX=0)

    # C. Median Blur: Mengganti piksel tengah dengan nilai median dari kernel (sangat efektif untuk bintik hitam-putih)
    blur_median = cv2.medianBlur(img_rgb, 7)

    # D. Bilateral Filter: Menghaluskan area seragam tetapi mempertahankan garis batas/tepi objek tetap tajam
    blur_bilateral = cv2.bilateralFilter(img_rgb, d=9, sigmaColor=75, sigmaSpace=75)

    # =========================================================================
    # BAGIAN 2: OPERATOR GRADIEN & TEPI (SOBEL & LAPLACIAN)
    # =========================================================================
    # Sobel X: Mendeteksi perubahan intensitas horizontal (garis vertikal)
    sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobel_x = cv2.convertScaleAbs(sobel_x)

    # Sobel Y: Mendeteksi perubahan intensitas vertikal (garis horizontal)
    sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    abs_sobel_y = cv2.convertScaleAbs(sobel_y)

    # Gabungan Magnitude Sobel (X + Y)
    sobel_combined = cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)

    # Laplacian: Deteksi tepi menggunakan operator turunan kedua
    laplacian = cv2.Laplacian(img_gray, cv2.CV_64F, ksize=3)
    abs_laplacian = cv2.convertScaleAbs(laplacian)

    # =========================================================================
    # BAGIAN 3: CANNY EDGE DETECTION (STANDAR TERBAIK UNTUK DETEKSI TEPI)
    # =========================================================================
    # Canny membutuhkan 2 nilai ambang batas (Thresholds):
    # - minVal: Di bawah nilai ini pasti BUKAN tepi (dibuang)
    # - maxVal: Di atas nilai ini PASTI tepi valid
    # - Di antara minVal & maxVal: Hanya dianggap tepi jika terhubung dengan piksel tepi valid
    # Rekomendasi: Lakukan sedikit Gaussian Blur terlebih dahulu untuk meminimalkan false edge akibat derau kamera
    pre_blurred_gray = cv2.GaussianBlur(img_gray, (5, 5), 1.4)
    canny_edges_wide = cv2.Canny(pre_blurred_gray, 50, 150)
    canny_edges_tight = cv2.Canny(pre_blurred_gray, 100, 200)

    print("[INFO] Deteksi tepi Canny berhasil dieksekusi.")
    print(" - Canny Wide Threshold: (50, 150)")
    print(" - Canny Tight Threshold: (100, 200)")

    # =========================================================================
    # VISUALISASI HASIL DENGAN MATPLOTLIB
    # =========================================================================
    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
    fig.suptitle("Modul 05: Filtering, Blurring, dan Edge Detection", fontsize=16)

    # Baris 1: Metode Blurring / Smoothing
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("1. Citra Asli (RGB)")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(blur_gaussian)
    axes[0, 1].set_title("2. Gaussian Blur (7x7)\n(Peredam derau umum)")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(blur_bilateral)
    axes[0, 2].set_title("3. Bilateral Filter\n(Tepi objek tetap tajam)")
    axes[0, 2].axis("off")

    # Baris 2: Sobel Gradient
    axes[1, 0].imshow(abs_sobel_x, cmap="gray")
    axes[1, 0].set_title("4. Sobel X (Garis Vertikal)")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(abs_sobel_y, cmap="gray")
    axes[1, 1].set_title("5. Sobel Y (Garis Horizontal)")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(sobel_combined, cmap="gray")
    axes[1, 2].set_title("6. Sobel Gabungan (X + Y)")
    axes[1, 2].axis("off")

    # Baris 3: Laplacian & Canny Edge Detection
    axes[2, 0].imshow(abs_laplacian, cmap="gray")
    axes[2, 0].set_title("7. Laplacian (Derivatif ke-2)")
    axes[2, 0].axis("off")

    axes[2, 1].imshow(canny_edges_wide, cmap="gray")
    axes[2, 1].set_title("8. Canny Edges (Threshold: 50, 150)\n(Lebih sensitif/banyak tepi)")
    axes[2, 1].axis("off")

    axes[2, 2].imshow(canny_edges_tight, cmap="gray")
    axes[2, 2].set_title("9. Canny Edges (Threshold: 100, 200)\n(Hanya batas kontur tegas)")
    axes[2, 2].axis("off")

    plt.tight_layout()
    output_path = os.path.join(current_dir, "images", "hasil_filtering_dan_canny.png")
    plt.savefig(output_path)
    print(f"[INFO] Visualisasi perbandingan disimpan ke: {output_path}")

    print("\n[PETUNJUK] Tutup jendela matplotlib untuk menyelesaikan eksekusi program.")
    plt.show()

if __name__ == "__main__":
    main()

