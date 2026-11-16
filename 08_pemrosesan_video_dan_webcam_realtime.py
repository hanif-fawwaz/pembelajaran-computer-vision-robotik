"""
=============================================================================
Modul 08: Pemrosesan Video & Kamera Webcam Real-Time dengan FPS Counter
=============================================================================
Deskripsi:
Program ini mengajarkan cara menangani aliran data video dinamis (Video Stream):
1. Mengakses kamera webcam langsung (cv2.VideoCapture(0)) atau membaca file rekaman video.
2. Mekanisme perulangan frame (Looping Stream) dan pembacaan frame (cap.read()).
3. Perhitungan metrik performa secara presisi: FPS (Frames Per Second) dan Frame Latency.
4. Menampilkan HUD (Heads-Up Display) overlay:
   - FPS Counter (real-time vs moving average)
   - Resolusi frame dan waktu eksekusi per frame (ms)
5. Menerapkan pemrosesan citra interaktif secara real-time via keyboard:
   - Tombol '1': Mode Normal (Warna Asli)
   - Tombol '2': Mode Grayscale
   - Tombol '3': Mode Canny Edge Detection (Tepi Real-time)
   - Tombol '4': Mode Color Tracking Mask (Threshold HSV dinamis)
   - Tombol 's': Mengambil foto snapshot dari video dan menyimpannya ke disk
   - Tombol 'q' atau ESC: Keluar dari program dengan aman
6. Perekaman / Penyimpanan video hasil pemrosesan (cv2.VideoWriter).

Relevansi Robotik:
- Menjaga laju komputasi robot agar tetap sinkron dengan loop kendali motor (misal target >= 30 FPS)
- Menyaring frame video langsung dari onboard camera (USB Webcam, CSI Camera, ESP32-CAM stream)
- Debugging interaktif saat robot sedang beroperasi di lapangan

Penulis: Pembelajaran Robotik & Vision
=============================================================================
"""

import os
import time
import cv2
import numpy as np

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sample_video_path = os.path.join(current_dir, "images", "sample_video.mp4")

    print("=================================================================")
    print("      SISTEM VISION VIDEO REAL-TIME & WEBCAM - OPENCV")
    print("=================================================================")
    print("Pilih Sumber Video:")
    print(" [1] Gunakan Kamera Webcam Utama (Device Index 0)")
    print(" [2] Gunakan File Video Simulasi Bawaan (sample_video.mp4)")
    print("-----------------------------------------------------------------")

    # Buka kamera webcam default (0), jika gagal otomatis fallback ke file video simulasi
    cap = cv2.VideoCapture(0)
    source_name = "Webcam Kamera (Index 0)"

    if not cap.isOpened():
        print("[WARNING] Kamera webcam tidak terdeteksi atau sedang digunakan aplikasi lain!")
        print(f"[INFO] Beralih otomatis menggunakan file video sampel: {sample_video_path}")
        cap = cv2.VideoCapture(sample_video_path)
        source_name = "File Simulasi (sample_video.mp4)"

    if not cap.isOpened():
        print("[ERROR] Gagal membuka sumber video!")
        return

    # Mengatur resolusi capture webcam (opsional, misalnya 640x480 untuk performa tinggi)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Inisialisasi variabel penghitung performa (FPS)
    prev_time = time.time()
    fps_history = []
    mode = 1  # 1: Normal, 2: Grayscale, 3: Canny Edge, 4: HSV Tracking
    mode_names = {
        1: "Normal BGR",
        2: "Grayscale",
        3: "Canny Edge Detection",
        4: "HSV Color Tracking (Orange/Yellow)"
    }

    # Rentang HSV untuk tracking objek oranye/kuning pada mode 4
    lower_color = np.array([5, 100, 100], dtype=np.uint8)
    upper_color = np.array([25, 255, 255], dtype=np.uint8)

    print("\n[KONTROL KEYBOARD INTERAKTIF]")
    print(" Tekan '1' : Mode Normal (Warna Asli)")
    print(" Tekan '2' : Mode Grayscale")
    print(" Tekan '3' : Mode Canny Edge Detection")
    print(" Tekan '4' : Mode HSV Color Tracking")
    print(" Tekan 's' : Ambil Snapshot (Screenshot)")
    print(" Tekan 'q' atau ESC : Keluar dari program")
    print("-----------------------------------------------------------------\n")

    snapshot_counter = 0

    while True:
        start_frame_time = time.time()
        ret, frame = cap.read()

        # Jika membaca dari file video dan frame habis, putar ulang (loop)
        if not ret:
            if "File" in source_name:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                print("[INFO] Video stream terhenti.")
                break

        h, w = frame.shape[:2]

        # =====================================================================
        # PEMROSESAN CITRA DINAMIS BERDASARKAN MODE
        # =====================================================================
        if mode == 1:
            processed_frame = frame.copy()
        elif mode == 2:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            processed_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        elif mode == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
            edges = cv2.Canny(blurred, 50, 150)
            processed_frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        elif mode == 4:
            # Mode HSV Color Tracking dengan visualisasi centroid objek
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower_color, upper_color)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            processed_frame = cv2.bitwise_and(frame, frame, mask=mask)

            # Cari titik tengah objek
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) > 300:
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    cv2.rectangle(processed_frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cv2.circle(processed_frame, (cx, cy), 6, (0, 0, 255), -1)
                        cv2.putText(processed_frame, f"({cx}, {cy})", (cx + 10, cy),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # =====================================================================
        # PENGHITUNGAN FPS & PERFORMA KOMPUTASI REAL-TIME
        # =====================================================================
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time

        fps_history.append(fps)
        if len(fps_history) > 30:
            fps_history.pop(0)
        avg_fps = sum(fps_history) / len(fps_history)

        elapsed_ms = (time.time() - start_frame_time) * 1000.0

        # =====================================================================
        # HEADS-UP DISPLAY (HUD) OVERLAY
        # =====================================================================
        # Buat background transparan kecil di pojok kiri atas untuk teks
        cv2.rectangle(processed_frame, (10, 10), (320, 115), (20, 20, 20), -1)
        cv2.rectangle(processed_frame, (10, 10), (320, 115), (0, 255, 0), 1)

        # Status Teks
        cv2.putText(processed_frame, f"FPS: {fps:.1f} (Avg: {avg_fps:.1f})", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(processed_frame, f"Latency: {elapsed_ms:.1f} ms | Res: {w}x{h}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(processed_frame, f"Mode [{mode}]: {mode_names[mode]}", (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)
        cv2.putText(processed_frame, f"Source: {source_name}", (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1)

        # Tampilkan frame ke jendela GUI
        cv2.imshow("Robot Vision Stream - Real-Time Processing", processed_frame)

        # =====================================================================
        # PENANGANAN INPUT KEYBOARD
        # =====================================================================
        # Menunggu 1 ms antar frame untuk memastikan loop video berjalan lancar
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:  # 'q' atau tombol ESC untuk keluar
            print("[INFO] Pengguna menghentikan program.")
            break
        elif key in [ord('1'), ord('2'), ord('3'), ord('4')]:
            mode = int(chr(key))
            print(f"[STATUS] Berganti ke Mode {mode}: {mode_names[mode]}")
        elif key == ord('s'):
            snapshot_counter += 1
            snap_path = os.path.join(current_dir, "images", f"snapshot_{snapshot_counter}.jpg")
            cv2.imwrite(snap_path, processed_frame)
            print(f"[SNAPSHOT] Foto frame berhasil disimpan ke: {snap_path}")

    # Bersihkan memori dan lepaskan hardware kamera
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Kamera dan seluruh jendela OpenCV telah ditutup dengan aman.")

if __name__ == "__main__":
    main()

