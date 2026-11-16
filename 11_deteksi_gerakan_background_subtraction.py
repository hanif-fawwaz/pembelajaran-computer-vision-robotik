"""
=============================================================================
Modul 11: Deteksi Gerakan (Motion Detection & Background Subtraction MOG2)
=============================================================================
Deskripsi:
Program ini membahas teknik mendeteksi gerakan dan mengisolasi objek bergerak dari latar belakang:
1. Metode 1: Frame Differencing Dasar (Delta Frame Absolute Difference cv2.absdiff):
   - Menghitung selisih intensitas antara 2 frame berturut-turut ($|\text{Frame}_t - \text{Frame}_{t-1}|$)
   - Cepat dan ringan, namun rentan noise dan menyisakan lubang di bagian dalam objek.
2. Metode 2: Background Subtractor MOG2 (Mixture of Gaussians 2):
   - Model probabilitas adaptif untuk mempelajari latar belakang statis seiring waktu.
   - Fitur deteksi bayangan cerdas (Shadow Detection): Bayangan diberi label abu-abu (127),
     sedangkan objek bergerak nyata diberi label putih (255).
3. Pembersihan Masker Gerakan dengan Operasi Morfologi (Opening & Dilatasi).
4. Pelacakan Bounding Box & Titik Pusat (Centroid) Objek yang Bergerak.
5. Indikator Sistem Alarm Keamanan Robot (Motion Security Alert):
   - Memicu status "GERAKAN TERDETEKSI" dan menghitung luas area gerak.
6. Dukungan sumber kamera fleksibel: Webcam langsung atau File Video Simulasi (motion_sample.mp4).

Relevansi Robotik:
- Robot patroli keamanan mandiri (Security Sentry Robot) yang mendeteksi penyusup
- Auto-wake system: Robot tidur untuk menghemat baterai dan baru aktif saat ada gerakan
- Monitoring lintasan conveyor belt industri untuk menghitung barang yang lewat

Penulis: Pembelajaran Robotik & Vision
=============================================================================
"""

import os
import time
import cv2
import numpy as np

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sample_video_path = os.path.join(current_dir, "images", "motion_sample.mp4")

    print("=================================================================")
    print("      SISTEM DETEKSI GERAKAN & BACKGROUND SUBTRACTION MOG2")
    print("=================================================================")
    print("Pilih Sumber Video:")
    print(" [1] Gunakan Kamera Webcam Utama (Device Index 0)")
    print(" [2] Gunakan File Video Simulasi Gerakan (motion_sample.mp4)")
    print("-----------------------------------------------------------------")

    # Inisialisasi sumber video
    cap = cv2.VideoCapture(0)
    source_name = "Webcam (Index 0)"

    if not cap.isOpened():
        print("[WARNING] Kamera webcam tidak ditemukan!")
        print(f"[INFO] Beralih otomatis ke video simulasi gerakan: {sample_video_path}")
        cap = cv2.VideoCapture(sample_video_path)
        source_name = "Video Simulasi (motion_sample.mp4)"

    if not cap.isOpened():
        print("[ERROR] Gagal membuka sumber video!")
        return

    # Inisialisasi Background Subtractor MOG2 dari OpenCV
    # - history: Jumlah frame terakhir yang dipelajari untuk model background (biasanya 300 - 500)
    # - varThreshold: Batas ambang jarak Mahalanobis (makin kecil makin sensitif terhadap gerakan kecil)
    # - detectShadows=True: Mengidentifikasi bayangan (ditandai warna abu-abu nilai 127)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=True)

    # Variabel untuk metode Frame Differencing
    prev_gray = None

    # Pengaturan GUI & Kontrol
    mode = 1  # 1: MOG2 Subtractor, 2: Frame Differencing
    print("\n[KONTROL INTERAKTIF]")
    print(" Tekan '1' : Mode Background Subtractor MOG2 (Direkomendasikan)")
    print(" Tekan '2' : Mode Simple Frame Differencing (AbsDiff)")
    print(" Tekan 'q' atau ESC : Keluar dari program\n")

    # Ambil frame snapshot untuk penyimpanan aset dokumentasi
    snapshot_saved = False

    while True:
        ret, frame = cap.read()
        if not ret:
            # Jika video simulasi habis, putar ulang (looping)
            if "Video Simulasi" in source_name:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Inisialisasi frame sebelumnya untuk Frame Differencing
        if prev_gray is None:
            prev_gray = blurred_gray
            continue

        # =====================================================================
        # EKSTRAKSI MASKER GERAKAN (MOTION MASK)
        # =====================================================================
        if mode == 1:
            # METODE 1: MOG2 BACKGROUND SUBTRACTION
            fg_mask_raw = bg_subtractor.apply(frame)

            # Filter bayangan: Bayangan di MOG2 bernilai 127 (abu-abu).
            # Kita hanya ingin objek gerakan solid (nilai 255), maka gunakan threshold > 200
            _, fg_mask = cv2.threshold(fg_mask_raw, 200, 255, cv2.THRESH_BINARY)
            mode_label = "MOG2 Subtractor (Adaptive)"
        else:
            # METODE 2: FRAME DIFFERENCING (ABSOLUTE DIFFERENCE)
            frame_diff = cv2.absdiff(prev_gray, blurred_gray)
            _, fg_mask = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
            prev_gray = blurred_gray.copy()
            mode_label = "Frame Differencing (|F(t) - F(t-1)|)"

        # =====================================================================
        # PEMBERSIHAN NOISE DENGAN OPERASI MORFOLOGI
        # =====================================================================
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # Opening: Menghapus bintik derau kecil
        mask_clean = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        # Dilatasi: Mempertebal area objek agar kontur tidak terputus
        mask_clean = cv2.dilate(mask_clean, kernel, iterations=2)

        # =====================================================================
        # DETEKSI KONTUR & STATUS ALARM KEAMANAN ROBOT
        # =====================================================================
        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        frame_vis = frame.copy()
        motion_detected = False
        total_motion_area = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter luas minimum untuk mengabaikan getaran angin / noise kecil
            if area > 800:
                motion_detected = True
                total_motion_area += area

                # Bounding box objek bergerak (Warna Merah Alarm)
                x, y, bw, bh = cv2.boundingRect(cnt)
                cv2.rectangle(frame_vis, (x, y), (x + bw, y + bh), (0, 0, 255), 2)

                # Titik Centroid
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.circle(frame_vis, (cx, cy), 5, (0, 255, 255), -1)

                cv2.putText(frame_vis, f"Objek Bergerak (Area:{int(area)})", (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 2)

        # =====================================================================
        # HEADS-UP DISPLAY (HUD) INDIKATOR STATUS ALARM
        # =====================================================================
        # Kotak HUD Atas
        cv2.rectangle(frame_vis, (15, 15), (420, 115), (30, 30, 30), -1)
        cv2.rectangle(frame_vis, (15, 15), (420, 115), (0, 255, 0) if not motion_detected else (0, 0, 255), 2)

        if motion_detected:
            status_text = "STATUS: [!] GERAKAN TERDETEKSI"
            status_color = (0, 0, 255)  # Merah
        else:
            status_text = "STATUS: AMAN (TIDAK ADA GERAKAN)"
            status_color = (0, 255, 0)  # Hijau

        cv2.putText(frame_vis, status_text, (25, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)
        cv2.putText(frame_vis, f"Algoritma: {mode_label}", (25, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(frame_vis, f"Total Luas Gerak: {int(total_motion_area)} px | Src: {source_name}", (25, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

        # Buat tampilan kombinasi Side-by-Side (Frame Asli + Masker Gerakan)
        mask_3ch = cv2.cvtColor(mask_clean, cv2.COLOR_GRAY2BGR)
        combined_view = np.hstack((frame_vis, mask_3ch))

        # Simpan 1 snapshot perbandingan untuk dokumentasi jika ada gerakan
        if motion_detected and not snapshot_saved:
            out_img_path = os.path.join(current_dir, "images", "hasil_deteksi_gerakan.png")
            cv2.imwrite(out_img_path, combined_view)
            snapshot_saved = True
            print(f"[INFO] Snapshot deteksi gerakan disimpan ke: {out_img_path}")

        # Tampilkan jendela GUI
        cv2.imshow("Robot Motion Sentry - OpenCV MOG2", combined_view)

        # Kontrol Keyboard
        key = cv2.waitKey(15) & 0xFF
        if key == ord('q') or key == 27:
            print("[INFO] Program dihentikan pengguna.")
            break
        elif key == ord('1'):
            mode = 1
            print("[STATUS] Beralih ke Mode 1: Background Subtractor MOG2")
        elif key == ord('2'):
            mode = 2
            print("[STATUS] Beralih ke Mode 2: Frame Differencing")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Kamera dan jendela video ditutup dengan aman.")

if __name__ == "__main__":
    main()

