"""
=============================================================================
Modul 09: Deteksi Garis (Line Tracking) Menggunakan Hough Line Transform
=============================================================================
Deskripsi:
Program ini membahas algoritma standar deteksi garis lintasan (Lane Detection / Line Tracking)
yang digunakan pada robot otonom dan sistem kemudi mandiri (Self-Driving Car):
1. Pra-pemrosesan citra jalur jalan (Grayscale & Gaussian Blur)
2. Isolasi Region of Interest (ROI Masking Poligon Segitiga/Trapesium):
   - Membatasi fokus pemrosesan hanya pada area jalanan di depan robot
3. Deteksi Tepi (Canny Edge Detection)
4. Transformasi Ruang Hough:
   - Konsep ruang parameter Hesse ($r = x \cos \theta + y \sin \theta$)
   - Standar Hough Line (cv2.HoughLines) vs Probabilistic Hough Line (cv2.HoughLinesP)
5. Pemisahan & Klasifikasi Garis Kiri vs Garis Kanan berdasarkan kemiringan (Slope $m$):
   - $m < 0$: Marka Garis Kiri
   - $m > 0$: Marka Garis Kanan
6. Penghitungan Sudut Kemudi Robot (Steering Angle) dan Titik Deviasi Tengah (Offset Error):
   - Menghitung deviasi titik tengah lintasan terhadap posisi pusat robot

Relevansi Robotik:
- Robot AGV / Line Follower di gudang pabrik berbasis kamera
- Lane Keeping Assist System (LKAS) pada mobil otonom

Penulis: Pembelajaran Robotik & Vision
=============================================================================
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_roi_mask(image):
    """
    Membuat masker poligon trapesium untuk mengisolasi area jalan di depan robot.
    Area langit dan rintangan samping akan diabaikan untuk menghemat komputasi.
    """
    h, w = image.shape[:2]
    # Koordinat 4 sudut trapesium fokus jalanan (Polygon ROI)
    pts = np.array([
        [(30, h), (220, int(h * 0.52)), (420, int(h * 0.52)), (w - 30, h)]
    ], dtype=np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, pts, 255)
    return mask, pts

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_dir, "images", "track_line_sample.jpg")

    # 1. Membaca gambar jalur jalan
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print("[ERROR] File 'track_line_sample.jpg' tidak ditemukan!")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape

    # 2. Konversi Grayscale & Gaussian Blur untuk mereduksi noise aspal
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)

    # 3. Deteksi Tepi Canny
    edges = cv2.Canny(blurred, 50, 150)

    # 4. Terapkan Masker Poligon ROI (Region of Interest)
    roi_mask, roi_pts = get_roi_mask(edges)
    edges_roi = cv2.bitwise_and(edges, edges, mask=roi_mask)

    # =========================================================================
    # TRANSFORMASI HOUGH LINE PROBABILISTIK (cv2.HoughLinesP)
    # =========================================================================
    # Parameter:
    # - rho: Resolusi jarak akumulator dalam piksel (biasanya 1 atau 2)
    # - theta: Resolusi sudut akumulator dalam radian (np.pi / 180 = 1 derajat)
    # - threshold: Jumlah voting minimal untuk menyatakan segmen garis
    # - minLineLength: Panjang segmen garis minimal yang diterima (px)
    # - maxLineGap: Jarak celah maksimum antara 2 titik agar tetap dianggap 1 garis utuh
    lines = cv2.HoughLinesP(
        edges_roi,
        rho=1,
        theta=np.pi / 180,
        threshold=35,
        minLineLength=30,
        maxLineGap=20
    )

    img_detected_lines = img_rgb.copy()
    img_robot_navigation = img_rgb.copy()

    left_lane_x = []
    right_lane_x = []
    y_eval = int(h * 0.85)  # Ketinggian evaluasi garis lintasan dekat bumper robot

    print(f"\n[INFO] Ekstraksi Garis Hough Selesai. Segmen terdeteksi: {len(lines) if lines is not None else 0}")

    if lines is not None:
        for line in lines:
            # Kompatibilitas multi-versi OpenCV: tangani format shape (N, 1, 4) maupun (N, 4)
            coords = line[0] if (isinstance(line, np.ndarray) and line.ndim > 1 and len(line[0]) == 4) else line
            x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
            # Gambar seluruh segmen mentah dengan warna biru cyan
            cv2.line(img_detected_lines, (x1, y1), (x2, y2), (0, 220, 255), 3)

            # Hitung kemiringan garis (Slope m = dy / dx)
            if x2 - x1 != 0:
                slope = (y2 - y1) / float(x2 - x1)
                # Filter kemiringan horizontal/datar yang tidak relevan dengan marka jalur
                if abs(slope) < 0.3:
                    continue

                # Estimasi titik X garis pada ketinggian y_eval menggunakan persamaan linier:
                # x = x1 + (y_eval - y1) / slope
                x_at_eval = int(x1 + (y_eval - y1) / slope)

                # Slope negatif -> Garis Kiri, Slope positif -> Garis Kanan (karena sumbu Y OpenCV mengarah ke bawah)
                if slope < 0 and x1 < w // 2:
                    left_lane_x.append(x_at_eval)
                    cv2.line(img_robot_navigation, (x1, y1), (x2, y2), (255, 0, 0), 4)  # Merah untuk kiri
                elif slope > 0 and x2 > w // 2:
                    right_lane_x.append(x_at_eval)
                    cv2.line(img_robot_navigation, (x1, y1), (x2, y2), (0, 255, 0), 4)  # Hijau untuk kanan

    # =========================================================================
    # KALKULASI SUDUT KEMUDI & OFFSET TITIK TENGAH ROBOT
    # =========================================================================
    robot_center_x = w // 2
    lane_center_x = robot_center_x  # Default jika hanya satu marka terlihat

    if left_lane_x and right_lane_x:
        avg_left = int(np.mean(left_lane_x))
        avg_right = int(np.mean(right_lane_x))
        lane_center_x = (avg_left + avg_right) // 2
    elif left_lane_x:
        avg_left = int(np.mean(left_lane_x))
        lane_center_x = avg_left + 150  # Estimasi jarak setengah lebar jalan
    elif right_lane_x:
        avg_right = int(np.mean(right_lane_x))
        lane_center_x = avg_right - 150

    # Deviasi kesalahan posisi (Error offset)
    # Positif: Lintasan berada di kanan pusat robot -> Robot harus belok kanan
    # Negatif: Lintasan berada di kiri pusat robot -> Robot harus belok kiri
    error_offset_px = lane_center_x - robot_center_x

    # Estimasi sudut kemudi (Steering Angle)
    # Formula trigonometri: tan(theta) = error / distance
    lookahead_distance = h - y_eval
    steering_angle_deg = np.degrees(np.arctan2(error_offset_px, lookahead_distance))

    # Tentukan rekomendasi manuver robot
    if abs(error_offset_px) < 15:
        nav_command = "LURUS (MAJU CEPAT)"
        cmd_color = (0, 255, 0)
    elif error_offset_px > 15:
        nav_command = f"BELOK KANAN ({steering_angle_deg:.1f} deg)"
        cmd_color = (255, 140, 0)
    else:
        nav_command = f"BELOK KIRI ({abs(steering_angle_deg):.1f} deg)"
        cmd_color = (255, 140, 0)

    print("\n--- METRIK NAVIGASI LINE TRACKING ROBOT ---")
    print(f"Pusat Kamera Robot (X)  : {robot_center_x} px")
    print(f"Pusat Lintasan Jalan (X): {lane_center_x} px")
    print(f"Error Offset Deviasi    : {error_offset_px} px")
    print(f"Sudut Kemudi (Steering) : {steering_angle_deg:.2f}°")
    print(f"Instruksi Motor Robot   : {nav_command}")

    # =========================================================================
    # GAMBAR VISUALISASI PANDUAN PADA FRAME NAVIGASI ROBOT
    # =========================================================================
    # Garis tengah sumbu kamera robot (Kuning putus-putus)
    cv2.line(img_robot_navigation, (robot_center_x, h), (robot_center_x, int(h * 0.6)), (255, 255, 0), 2)
    # Garis arah target lintasan jalan (Merah terang)
    cv2.line(img_robot_navigation, (robot_center_x, h), (lane_center_x, y_eval), (0, 0, 255), 3)
    cv2.circle(img_robot_navigation, (lane_center_x, y_eval), 8, (0, 255, 255), -1)

    # Kotak HUD Navigasi Robot
    cv2.rectangle(img_robot_navigation, (15, 15), (380, 110), (30, 30, 30), -1)
    cv2.putText(img_robot_navigation, f"NAV: {nav_command}", (25, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, cmd_color, 2)
    cv2.putText(img_robot_navigation, f"Offset Error: {error_offset_px} px", (25, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(img_robot_navigation, f"Steering Angle: {steering_angle_deg:.1f} deg", (25, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Gambar batas poligon ROI pada citra edges
    img_roi_vis = cv2.cvtColor(edges_roi, cv2.COLOR_GRAY2RGB)
    cv2.polylines(img_roi_vis, roi_pts, True, (0, 255, 0), 2)

    # =========================================================================
    # VISUALISASI HASIL DENGAN MATPLOTLIB
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("Modul 09: Deteksi Garis Lintasan & Line Follower Robot (Hough Line Transform)", fontsize=15)

    # 1. Citra Asli Input
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("1. Citra Asli Lintasan Jalan")
    axes[0, 0].axis("off")

    # 2. Canny Edges dalam Region of Interest (ROI)
    axes[0, 1].imshow(img_roi_vis)
    axes[0, 1].set_title("2. Canny Edges + Masker Trapesium ROI\n(Fokus jalanan depan robot)")
    axes[0, 1].axis("off")

    # 3. Segmen Garis Mentah Hough Lines
    axes[1, 0].imshow(img_detected_lines)
    axes[1, 0].set_title("3. Segmen Garis Mentah (cv2.HoughLinesP)")
    axes[1, 0].axis("off")

    # 4. Hasil Analisis Navigasi Robot & Steering
    axes[1, 1].imshow(img_robot_navigation)
    axes[1, 1].set_title("4. Output Navigasi Robot (Kemudi & Offset Deviasi)")
    axes[1, 1].axis("off")

    plt.tight_layout()
    output_path = os.path.join(current_dir, "images", "hasil_line_tracking.png")
    plt.savefig(output_path)
    print(f"\n[INFO] Gambar visualisasi navigasi disimpan ke: {output_path}")

    print("\n[PETUNJUK] Tutup jendela matplotlib yang muncul untuk menyelesaikan program.")
    plt.show()

if __name__ == "__main__":
    main()

