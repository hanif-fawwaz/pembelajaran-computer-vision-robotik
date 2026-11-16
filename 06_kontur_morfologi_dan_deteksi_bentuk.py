"""
=============================================================================
Modul 06: Kontur, Operasi Morfologi, Bounding Box, dan Deteksi Bentuk (Shape Detection)
=============================================================================
Deskripsi:
Program ini membahas tuntas ekstraksi fitur geometri objek 2D:
1. Operasi Morfologi Lanjutan:
   - Erosi (cv2.erode) & Dilatasi (cv2.dilate)
   - Opening (Erosi -> Dilatasi: hapus noise latar depan)
   - Closing (Dilatasi -> Erosi: tutup lubang kecil dalam objek)
2. Hierarki dan Ekstraksi Kontur (cv2.findContours):
   - RETR_EXTERNAL vs RETR_TREE
   - CHAIN_APPROX_SIMPLE
3. Perhitungan Properti Geometri Kontur:
   - Luas Area (cv2.contourArea)
   - Keliling Kontur (cv2.arcLength)
   - Image Moments & Centroid (cx, cy)
4. Pembatasan Area Objek (Bounding Representations):
   - Upright Bounding Box (cv2.boundingRect)
   - Minimum Area Rotated Bounding Box (cv2.minAreaRect)
   - Minimum Enclosing Circle (cv2.minEnclosingCircle)
5. Deteksi Bentuk Poligon Otomatis (Shape Classification):
   - Douglas-Peucker Polygon Approximation (cv2.approxPolyDP)
   - Klasifikasi: Segitiga (3 vertex), Kotak/Persegi Panjang (4 vertex + Aspect Ratio),
     Pentagon (5 vertex), Lingkaran / Poligon Kompleks (Circular-Metric / >6 vertex)

Relevansi Robotik:
- Membedakan rambu lalu lintas atau marker target (segitiga, lingkaran, kotak)
- Mengukur orientasi sudut (rotation angle) objek agar capit gripper robot tepat mengambil barang
- Sorting barang di conveyor belt industri berdasarkan luas area atau bentuk

Penulis: Pembelajaran Robotik & Vision
=============================================================================
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_shape(cnt):
    """
    Mengklasifikasikan bentuk objek berdasarkan jumlah sudut (vertex)
    menggunakan algoritma Douglas-Peucker (cv2.approxPolyDP).
    """
    peri = cv2.arcLength(cnt, True)
    # Epsilon toleransi aproksimasi (umumnya 2% s/d 4% dari keliling kontur)
    approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
    num_vertices = len(approx)

    shape = "Tidak Diketahui"

    if num_vertices == 3:
        shape = "Segitiga"
    elif num_vertices == 4:
        # Bedakan antara Persegi (Square) dan Persegi Panjang (Rectangle) dengan rasio aspek (Aspect Ratio)
        x, y, w, h = cv2.boundingRect(approx)
        ar = float(w) / float(h)
        if 0.90 <= ar <= 1.10:
            shape = "Persegi"
        else:
            shape = "Persegi Panjang"
    elif num_vertices == 5:
        shape = "Pentagon"
    elif num_vertices == 6:
        shape = "Hexagon"
    else:
        # Jika sudut banyak, periksa apakah mendekati lingkaran dengan rasio kebundaran (Circularity)
        area = cv2.contourArea(cnt)
        if peri > 0:
            # Formula circularity: 4 * pi * Area / (Perimeter^2)
            circularity = (4 * np.pi * area) / (peri * peri)
            if circularity > 0.75:
                shape = "Lingkaran"
            else:
                shape = f"Poligon ({num_vertices} Titik)"
        else:
            shape = f"Poligon ({num_vertices} Titik)"

    return shape, approx, num_vertices

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_dir, "images", "shapes_sample.jpg")

    # 1. Membaca gambar input
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print("[ERROR] File gambar 'shapes_sample.jpg' tidak ditemukan!")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 2. Thresholding: ubah gambar menjadi biner (objek putih, background hitam)
    # Latar gambar kita terang (245), maka gunakan cv2.THRESH_BINARY_INV
    _, thresh = cv2.threshold(img_gray, 230, 255, cv2.THRESH_BINARY_INV)

    # =========================================================================
    # BAGIAN 1: OPERASI MORFOLOGI (FILTERING & PEMBERSIHAN MASK)
    # =========================================================================
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # A. Opening (Erosi lalu Dilatasi): Sangat efektif melenyapkan bintik noise acak
    mask_opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # B. Closing (Dilatasi lalu Erosi): Menutup lubang/celah kosong di dalam objek
    mask_clean = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel)

    # =========================================================================
    # BAGIAN 2: MENCARI KONTUR (CONTOURS EXTRACTION)
    # =========================================================================
    # cv2.RETR_EXTERNAL: Hanya mendeteksi batas terluar objek (mengabaikan anak kontur di dalam)
    # cv2.CHAIN_APPROX_SIMPLE: Menghemat memori hanya menyimpan titik ujung/garis
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(f"\n[INFO] Total kontur ditemukan setelah morfologi: {len(contours)}")

    # Canvas untuk menggambar visualisasi
    img_shapes = img_rgb.copy()
    img_boxes = img_rgb.copy()

    # =========================================================================
    # BAGIAN 3 & 4: MENGHITUNG PROPERTI GEOMETRI & KLASIFIKASI BENTUK
    # =========================================================================
    for idx, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        # Filter kontur sangat kecil agar derau sisa tidak terhitung
        if area < 300:
            continue

        # A. Hitung Centroid (Titik Tengah) objek
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0

        # B. Deteksi Bentuk Objek
        shape_name, approx, vertices = detect_shape(cnt)

        print(f" -> Objek #{idx + 1}: Bentuk={shape_name:15s} | Luas={area:8.1f} px | Keliling={perimeter:6.1f} px | Sudut={vertices}")

        # Gambar garis batas kontur dan teks nama bentuk di img_shapes
        cv2.drawContours(img_shapes, [approx], -1, (255, 0, 0), 3)
        cv2.circle(img_shapes, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(img_shapes, shape_name, (cx - 40, cy - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2)
        cv2.putText(img_shapes, f"A:{int(area)}", (cx - 40, cy + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 100, 0), 2)

        # C. Bounding Box & Enclosing Circle di img_boxes
        # 1. Standard Upright Bounding Box (Garis Hijau)
        bx, by, bw, bh = cv2.boundingRect(cnt)
        cv2.rectangle(img_boxes, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)

        # 2. Minimum Area Rotated Bounding Box (Garis Biru)
        rect = cv2.minAreaRect(cnt)
        box_pts = cv2.boxPoints(rect)
        box_pts = np.int32(box_pts)
        cv2.drawContours(img_boxes, [box_pts], 0, (0, 120, 255), 2)

        # 3. Minimum Enclosing Circle (Garis Merah)
        (circle_x, circle_y), radius = cv2.minEnclosingCircle(cnt)
        cv2.circle(img_boxes, (int(circle_x), int(circle_y)), int(radius), (255, 0, 0), 2)

    # =========================================================================
    # VISUALISASI HASIL DENGAN MATPLOTLIB
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    # fig.suptitle("Modul 06: Morfologi, Bounding Box, dan Deteksi Bentuk Objek (Shape Detection)", fontsize=15)

    # 1. Citra Asli dengan Noise
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("1. Citra Input Geometri (Dengan Bintik Noise)")
    axes[0, 0].axis("off")

    # 2. Masker Biner Bersih Pasca Morfologi Opening
    axes[0, 1].imshow(mask_clean, cmap="gray")
    axes[0, 1].set_title("2. Masker Biner Bersih (Morfologi Opening & Closing)")
    axes[0, 1].axis("off")

    # 3. Klasifikasi Bentuk & Nilai Luas Area
    axes[1, 0].imshow(img_shapes)
    axes[1, 0].set_title("3. Shape Detection (Douglas-Peucker) & Luas Area")
    axes[1, 0].axis("off")

    # 4. Berbagai Model Bounding Objek
    axes[1, 1].imshow(img_boxes)
    axes[1, 1].set_title("4. Bounding Box (Hijau=Tegak, Oranye=Rotasi, Merah=Lingkaran)")
    axes[1, 1].axis("off")

    plt.tight_layout()
    output_path = os.path.join(current_dir, "images", "hasil_deteksi_bentuk.png")
    plt.savefig(output_path)
    print(f"\n[INFO] Gambar hasil visualisasi disimpan ke: {output_path}")

    print("\n[PETUNJUK] Tutup jendela matplotlib yang muncul untuk menyelesaikan program.")
    plt.show()

if __name__ == "__main__":
    main()

