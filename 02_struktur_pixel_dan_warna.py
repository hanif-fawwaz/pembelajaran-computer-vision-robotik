"""
=============================================================================
Modul 02: Memahami Piksel, Ruang Warna (BGR, RGB, HSV), dan Channel Splitting
=============================================================================
Deskripsi:
Program ini membahas tuntas konsep penting dalam Computer Vision:
1. Membaca dan memodifikasi nilai piksel individual (x, y)
2. Perbedaan penting format warna BGR (OpenCV) vs RGB (Matplotlib/Standar umum)
3. Memecah channel warna (Split Channel B, G, R)
4. Ruang warna HSV (Hue, Saturation, Value) & Mengapa HSV sangat penting di Robotik
5. Memvisualisasikan perbandingan warna dengan Matplotlib

Penulis: Pembelajaran Robotik & Vision
=============================================================================
"""

import os
import cv2
import matplotlib.pyplot as plt

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_dir, "images", "sample.jpg")

    # 1. Membaca gambar (Secara default OpenCV memuat dalam urutan format B-G-R)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print("[ERROR] File gambar tidak ditemukan!")
        return

    # =========================================================================
    # BAGIAN 1: Mengakses & Memodifikasi Nilai Piksel Individual
    # =========================================================================
    # Format koordinat array NumPy adalah: img[baris_y, kolom_x]
    y, x = 100, 100
    pixel_val = img_bgr[y, x]
    blue = pixel_val[0]
    green = pixel_val[1]
    red = pixel_val[2]
    print("--- NILAI PIKSEL PADA KOORDINAT (X=100, Y=100) ---")
    print(f"Format BGR : Blue={blue}, Green={green}, Red={red}")

    # =========================================================================
    # BAGIAN 2: BGR vs RGB (Konversi Ruang Warna)
    # =========================================================================
    # OpenCV menggunakan BGR karena alasan historis kamera awal.
    # Namun sebagian besar pustaka (Matplotlib, PIL, Web) menggunakan RGB.
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # =========================================================================
    # BAGIAN 3: Ruang Warna HSV (Hue, Saturation, Value)
    # =========================================================================
    # Mengapa HSV sangat krusial untuk Robotika?
    # - H (Hue, 0-179 di OpenCV) : Menunjukkan jenis warna murni (Merah, Biru, Kuning, dll).
    # - S (Saturation, 0-255)    : Kepekatan warna (0=pudar/putih, 255=sangat pekat).
    # - V (Value, 0-255)         : Tingkat kecerahan/intensitas cahaya (0=gelap gulita).
    #
    # Keunggulan: HSV memisahkan warna murni dari bayangan/kecerahan cahaya,
    # sehingga algoritma robotik tidak mudah tertipu oleh perubahan cahaya ruangan!
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # =========================================================================
    # BAGIAN 4: Memisahkan Saluran Warna (Channel Splitting)
    # =========================================================================
    b_channel, g_channel, r_channel = cv2.split(img_bgr)
    h_channel, s_channel, v_channel = cv2.split(img_hsv)

    # =========================================================================
    # BAGIAN 5: Visualisasi Komparasi Menggunakan Matplotlib
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Eksplorasi Ruang Warna & Channel Citra (OpenCV)", fontsize=16)

    # 1. Gambar BGR yang ditampilkan langsung dengan Matplotlib (Warna terbalik / salah)
    axes[0, 0].imshow(img_bgr)
    axes[0, 0].set_title("1. BGR langsung ke Matplotlib\n(Salah: Warna Merah & Biru tertukar)")
    axes[0, 0].axis("off")

    # 2. Gambar RGB yang sudah dikonversi (Warna yang benar)
    axes[0, 1].imshow(img_rgb)
    axes[0, 1].set_title("2. RGB (Tampilan Warna Asli)")
    axes[0, 1].axis("off")

    # 3. Gambar HSV
    axes[0, 2].imshow(img_hsv)
    axes[0, 2].set_title("3. Citra Format HSV\n(Representasi matematis warna)")
    axes[0, 2].axis("off")

    # 4. Channel Blue saja
    axes[1, 0].imshow(b_channel, cmap="Blues")
    axes[1, 0].set_title("4. Saluran Biru (Blue Channel)\n(Area biru terlihat paling terang)")
    axes[1, 0].axis("off")

    # 5. Channel Green saja
    axes[1, 1].imshow(g_channel, cmap="Greens")
    axes[1, 1].set_title("5. Saluran Hijau (Green Channel)\n(Area hijau/kuning terlihat terang)")
    axes[1, 1].axis("off")

    # 6. Channel Red saja
    axes[1, 2].imshow(r_channel, cmap="Reds")
    axes[1, 2].set_title("6. Saluran Merah (Red Channel)\n(Area merah/kuning terlihat terang)")
    axes[1, 2].axis("off")

    plt.tight_layout()
    output_comparison_path = os.path.join(current_dir, "images", "perbandingan_ruang_warna.png")
    plt.savefig(output_comparison_path)
    print(f"\n[INFO] Gambar perbandingan disimpan ke: {output_comparison_path}")

    # Tampilkan jendela matplotlib
    print("[PETUNJUK] Tutup jendela matplotlib yang muncul untuk mengakhiri program.")
    plt.show()

if __name__ == "__main__":
    main()

