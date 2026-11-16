"""
=============================================================================
Modul 03: Grayscale, Histogram Intensitas, dan Teknik Thresholding Citra
=============================================================================
Deskripsi:
Program ini membahas proses segmentasi citra dasar menggunakan intensitas cahaya:
1. Konversi citra ke Grayscale dan pemahaman histogram intensitas (distribusi nilai 0-255)
2. Binary Thresholding (Memisahkan objek latar depan vs latar belakang)
3. Inverse Binary Thresholding
4. Otsu's Thresholding (Penentuan nilai ambang batas optimal secara otomatis)
5. Adaptive Thresholding (Penyesuaian nilai ambang batas lokal untuk pencahayaan tidak merata)
6. Visualisasi histogram dan hasil segmentasi menggunakan Matplotlib

Relevansi Robotik:
- Pemisahan objek dari background secara cepat sebelum proses edge detection / contour
- Sangat efisien untuk robot dengan komputasi terbatas (misal Raspberry Pi / ESP32-CAM)

Penulis: Pembelajaran Robotik & Vision
=============================================================================
"""

import os
import cv2
import matplotlib.pyplot as plt

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_dir, "images", "sample.jpg")

    # 1. Membaca gambar dan konversi ke Grayscale
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print("[ERROR] File gambar tidak ditemukan!")
        return

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 2. Menghitung Histogram Citra Grayscale
    # cv2.calcHist(images, channels, mask, histSize, ranges)
    # Menghitung berapa banyak piksel untuk setiap nilai kecerahan 0 s/d 255
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])

    # =========================================================================
    # METODE-METODE THRESHOLDING
    # =========================================================================
    # 3. Simple Binary Thresholding
    # Piksel di atas 127 menjadi 255 (putih), di bawah 127 menjadi 0 (hitam)
    threshold_value = 127
    max_value = 255
    _, thresh_binary = cv2.threshold(img_gray, threshold_value, max_value, cv2.THRESH_BINARY)

    # 4. Binary Inverted Thresholding
    # Kebalikan dari binary: piksel di atas 127 jadi hitam, di bawah jadi putih
    _, thresh_binary_inv = cv2.threshold(img_gray, threshold_value, max_value, cv2.THRESH_BINARY_INV)

    # 5. Otsu's Thresholding (Otomatis)
    # OpenCV secara cerdas menganalisis puncak histogram untuk mencari batas pemisah paling optimal
    otsu_val, thresh_otsu = cv2.threshold(img_gray, 0, max_value, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"[INFO] Nilai threshold optimal yang ditemukan oleh Otsu: {otsu_val:.1f}")

    # 6. Adaptive Gaussian Thresholding
    # Menghitung nilai ambang batas lokal per-blok area (cocok jika ada bayangan / lampu tidak merata)
    thresh_adaptive = cv2.adaptiveThreshold(
        img_gray, max_value, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # =========================================================================
    # VISUALISASI MENGGUNAKAN MATPLOTLIB
    # =========================================================================
    fig = plt.figure(figsize=(15, 9))
    fig.suptitle("Modul 03: Grayscale, Histogram, dan Berbagai Metode Thresholding", fontsize=15)

    # Subplot 1: Citra Asli Grayscale
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(img_gray, cmap="gray")
    ax1.set_title("1. Citra Grayscale")
    ax1.axis("off")

    # Subplot 2: Histogram Intensitas
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(hist, color="black", linewidth=1.5)
    ax2.axvline(x=threshold_value, color="r", linestyle="--", label=f"Threshold Manual ({threshold_value})")
    ax2.axvline(x=otsu_val, color="g", linestyle="--", label=f"Otsu Optimal ({otsu_val:.0f})")
    ax2.set_title("2. Histogram Intensitas Piksel")
    ax2.set_xlabel("Nilai Kecerahan Piksel (0=Gelap, 255=Terang)")
    ax2.set_ylabel("Jumlah Piksel")
    ax2.set_xlim([0, 256])
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Subplot 3: Simple Binary Threshold
    ax3 = plt.subplot(2, 3, 3)
    ax3.imshow(thresh_binary, cmap="gray")
    ax3.set_title(f"3. Binary Threshold (T={threshold_value})")
    ax3.axis("off")

    # Subplot 4: Binary Inverted
    ax4 = plt.subplot(2, 3, 4)
    ax4.imshow(thresh_binary_inv, cmap="gray")
    ax4.set_title("4. Binary Inverted")
    ax4.axis("off")

    # Subplot 5: Otsu's Thresholding
    ax5 = plt.subplot(2, 3, 5)
    ax5.imshow(thresh_otsu, cmap="gray")
    ax5.set_title(f"5. Otsu's Thresholding (Auto T={otsu_val:.0f})")
    ax5.axis("off")

    # Subplot 6: Adaptive Gaussian Thresholding
    ax6 = plt.subplot(2, 3, 6)
    ax6.imshow(thresh_adaptive, cmap="gray")
    ax6.set_title("6. Adaptive Threshold (Gaussian)")
    ax6.axis("off")

    plt.tight_layout()
    output_path = os.path.join(current_dir, "images", "hasil_thresholding.png")
    plt.savefig(output_path)
    print(f"[INFO] Grafik visualisasi disimpan ke: {output_path}")

    print("\n[PETUNJUK] Tutup jendela matplotlib yang muncul untuk menyelesaikan program.")
    plt.show()

if __name__ == "__main__":
    main()

