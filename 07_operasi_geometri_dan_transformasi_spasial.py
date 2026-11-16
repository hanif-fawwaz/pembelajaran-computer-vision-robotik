"""
=============================================================================
Modul 07: Operasi Geometri & Transformasi Spasial Citra (OpenCV)
=============================================================================
Deskripsi:
Program ini membahas lengkap manipulasi spasial dan transformasi koordinat citra:
1. Resize (Penskalaan Citra) & Interpolasi:
   - cv2.INTER_AREA (Pengecilan / Shrinking optimal)
   - cv2.INTER_LINEAR & cv2.INTER_CUBIC (Pembesaran / Zooming optimal)
2. Cropping (Pemotongan ROI - Region of Interest):
   - NumPy slicing: img[y_start:y_end, x_start:x_end]
3. Translation (Pergeseran Posisi):
   - Matriks translasi 2x3 [1 0 tx; 0 1 ty] dan cv2.warpAffine
4. Rotation (Rotasi Sudut Bebas):
   - cv2.getRotationMatrix2D & cv2.warpAffine
5. Affine Transformation (Pemetaan 3 Titik):
   - Menjaga garis paralel tetap sejajar saat distorsi miring
6. Perspective Warp / Homography (Bird's Eye View - Pemetaan 4 Titik):
   - cv2.getPerspectiveTransform & cv2.warpPerspective
   - Mengubah sudut pandang miring kamera menjadi tegak lurus dari atas (Top-Down View)

Relevansi Robotik:
- Bird's Eye View (Inverse Perspective Mapping) untuk robot autonomous / mobil otonom melihat marka jalan
- Resizing frame resolusi tinggi kamera menjadi frame ringan agar FPS tinggi di mikrokontroler
- Normalisasi sudut orientasi objek sebelum proses template matching atau OCR

Penulis: Pembelajaran Robotik & Vision
=============================================================================
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_dir, "images", "perspective_sample.jpg")

    # 1. Membaca citra input
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print("[ERROR] File 'perspective_sample.jpg' tidak ditemukan!")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape
    print(f"[INFO] Resolusi citra asli: Lebar={w} px, Tinggi={h} px")

    # =========================================================================
    # BAGIAN 1: RESIZING (PENSKALAAN DENGAN METODE INTERPOLASI)
    # =========================================================================
    # A. Pengecilan (Downscaling): Gunakan INTER_AREA untuk menghindari moiré artifact
    img_small = cv2.resize(img_rgb, (w // 2, h // 2), interpolation=cv2.INTER_AREA)

    # B. Pembesaran (Upscaling): Gunakan INTER_CUBIC untuk detail lebih tajam
    # Contoh kita resize target ke ukuran spesifik (400, 300)
    img_resized = cv2.resize(img_rgb, (400, 300), interpolation=cv2.INTER_CUBIC)

    # =========================================================================
    # BAGIAN 2: CROPPING (REGION OF INTEREST - ROI)
    # =========================================================================
    # Di Python/NumPy, cropping dilakukan dengan slicing array: img[y1:y2, x1:x2]
    # Mengambil area dokumen tengah
    roi_crop = img_rgb[80:500, 150:720]

    # =========================================================================
    # BAGIAN 3: TRANSLASI (PERGESERAN KOORDINAT)
    # =========================================================================
    # Geser ke kanan (tx) sebesar 80 px, dan geser ke bawah (ty) sebesar 50 px
    tx, ty = 80, 50
    M_translation = np.float32([
        [1, 0, tx],
        [0, 1, ty]
    ])
    img_translated = cv2.warpAffine(img_rgb, M_translation, (w, h), borderValue=(50, 50, 50))

    # =========================================================================
    # BAGIAN 4: ROTASI DENGAN SUDUT BEBAS (ROTATION AROUND CENTER)
    # =========================================================================
    center = (w // 2, h // 2)
    angle_deg = 30 # Putar 30 derajat berlawanan jarum jam
    scale = 0.85   # Perkecil sedikit agar tidak terpotong tepi layar
    M_rotate = cv2.getRotationMatrix2D(center, angle_deg, scale)
    img_rotated = cv2.warpAffine(img_rgb, M_rotate, (w, h), borderValue=(50, 50, 50))

    # =========================================================================
    # BAGIAN 5: PERSPECTIVE WARP / BIRD'S EYE VIEW (PENTING UNTUK ROBOTIK)
    # =========================================================================
    # Menemukan 4 titik sudut dokumen miring pada citra (Top-Left, Top-Right, Bottom-Right, Bottom-Left)
    pts_src = np.float32([
        [200, 120],  # Titik 1: Kiri Atas
        [620, 80],   # Titik 2: Kanan Atas
        [700, 480],  # Titik 3: Kanan Bawah
        [140, 420]   # Titik 4: Kiri Bawah
    ])

    # Tentukan dimensi hasil pelurusan target (lebar=400, tinggi=300)
    target_w, target_h = 400, 300
    pts_dst = np.float32([
        [0, 0],
        [target_w, 0],
        [target_w, target_h],
        [0, target_h]
    ])

    # Hitung matriks transformasi perspektif 3x3
    M_perspective = cv2.getPerspectiveTransform(pts_src, pts_dst)

    # Terapkan transformasi perspektif untuk meratakan pandangan
    img_birds_eye = cv2.warpPerspective(img_rgb, M_perspective, (target_w, target_h))

    # Beri penanda visual pada 4 titik sudut di gambar panduan
    img_points_guide = img_rgb.copy()
    for idx, (px, py) in enumerate(pts_src):
        cv2.circle(img_points_guide, (int(px), int(py)), 8, (255, 0, 0), -1)
        cv2.putText(img_points_guide, f"P{idx+1}", (int(px) - 15, int(py) - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    # Gambar garis batas poligon 4 titik
    cv2.polylines(img_points_guide, [np.int32(pts_src)], True, (0, 255, 255), 2)

    # =========================================================================
    # VISUALISASI HASIL DENGAN MATPLOTLIB
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Modul 07: Operasi Geometri & Transformasi Spasial Citra (OpenCV)", fontsize=16)

    # 1. Citra Asli + 4 Titik Sumber Perspektif
    axes[0, 0].imshow(img_points_guide)
    axes[0, 0].set_title("1. Citra Input & 4 Sudut Perspektif (P1-P4)")
    axes[0, 0].axis("off")

    # 2. Hasil Bird's Eye View (Perspective Warp)
    axes[0, 1].imshow(img_birds_eye)
    axes[0, 1].set_title("2. Perspective Warp (Top-Down / Bird's Eye View)\n(Dokumen diluruskan secara otomatis)")
    axes[0, 1].axis("off")

    # 3. Hasil Cropping ROI
    axes[0, 2].imshow(roi_crop)
    axes[0, 2].set_title("3. Cropping ROI (Region of Interest)\n(Slicing NumPy array [y1:y2, x1:x2])")
    axes[0, 2].axis("off")

    # 4. Hasil Rotasi Sudut Bebas
    axes[1, 0].imshow(img_rotated)
    axes[1, 0].set_title("4. Rotasi Bebas (Angle: 30°, Scale: 0.85)")
    axes[1, 0].axis("off")

    # 5. Hasil Translasi (Pergeseran)
    axes[1, 1].imshow(img_translated)
    axes[1, 1].set_title("5. Translasi Citra (dx: +80px, dy: +50px)")
    axes[1, 1].axis("off")

    # 6. Hasil Resizing
    axes[1, 2].imshow(img_resized)
    axes[1, 2].set_title(f"6. Resized Interpolation (400x300)\n(Asli: {w}x{h} -> Diubah ke ukuran tetap)")
    axes[1, 2].axis("off")

    plt.tight_layout()
    output_path = os.path.join(current_dir, "images", "hasil_transformasi_geometri.png")
    plt.savefig(output_path)
    print(f"[INFO] Visualisasi perbandingan geometri disimpan ke: {output_path}")

    print("\n[PETUNJUK] Tutup jendela matplotlib yang muncul untuk menyelesaikan program.")
    plt.show()

if __name__ == "__main__":
    main()

