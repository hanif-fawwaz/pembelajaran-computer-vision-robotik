"""
=============================================================================
Modul 12 (Capstone Project): Sistem Penglihatan Robot Terpadu (Robot Vision Controller)
dengan Kendali PID dan Output Komunikasi Serial / ROS
=============================================================================
Deskripsi:
Program puncak ini mengintegrasikan seluruh konsep Computer Vision (Modul 01-11)
menjadi satu sistem arsitektur penglihatan robot otonom (*Autonomous Vision-Guided Robot*):

1. Arsitektur Vision Pipeline Modular:
   - Frame Acquisition (Webcam langsung atau Video Stream simulasi)
   - Color Segmentation & Preprocessing (HSV Masking + Morphological Opening/Closing)
   - Ekstraksi Fitur Target (Contours, Enclosing Circle, & Centroid $(cx, cy)$)
2. Estimasi Deviasi Spasial Relatif:
   - Error Sumbu X ($e_x = cx - \text{frame\_center\_x}$): Koreksi arah kemudi setir (Pan / Steering)
   - Error Sumbu Y ($e_y = cy - \text{frame\_center\_y}$): Koreksi sudut elevasi kamera (Tilt)
   - Error Luas / Jarak ($e_{\text{dist}} = r_{\text{target}} - r_{\text{aktual}}$): Koreksi maju/mundur robot (Throttle)
3. Algoritma Kendali PID (Proportional-Integral-Derivative):
   - Menghitung respon koreksi halus tanpa overshoot liar:
     $u(t) = K_p e(t) + K_i \int e(t)dt + K_d \frac{de(t)}{dt}$
4. Protokol Komunikasi Serial / Microcontroller UART (Arduino, ESP32, STM32):
   - Memformat paket data string terstruktur: `<STEER, THROTTLE, TARGET_FOUND>\n`
   - Fitur Mock Serial Driver: Mensimulasikan transmisi serial tanpa memerlukan kabel fisik
5. Heads-Up Display (HUD) Kokpit Robotik Lengkap:
   - Crosshair reticle pembidik target
   - Deadzone bounding box (zona toleransi netral agar motor tidak bergetar jitter)
   - Visualisasi grafik respons output PID dan metrik FPS real-time

Relevansi Robotik:
- Self-Balancing Robot / AGV yang mengikuti target orang/barang
- Gimbal Pan-Tilt Camera Tracker 2-DOF
- Robot Sepak Bola Beroda (KRSBI) yang mengejar bola secara otonom

Penulis: Pembelajaran Robotik & Vision
=============================================================================
"""

import os
import time
import cv2
import numpy as np

# =============================================================================
# KELAS CONTROLLER PID (PROPORTIONAL - INTEGRAL - DERIVATIVE)
# =============================================================================
class PIDController:
    """
    Mengimplementasikan pengendali PID diskrit untuk menstabilkan pergerakan robot.
    """
    def __init__(self, kp=0.45, ki=0.01, kd=0.15, output_limits=(-100, 100)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min_out, self.max_out = output_limits

        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = time.time()

    def update(self, error):
        current_time = time.time()
        dt = current_time - self.prev_time
        if dt <= 0.0:
            dt = 1e-4

        # P (Proportional)
        p_term = self.kp * error

        # I (Integral) dengan Anti-Windup Clamping
        self.integral += error * dt
        self.integral = np.clip(self.integral, -50.0, 50.0)
        i_term = self.ki * self.integral

        # D (Derivative)
        derivative = (error - self.prev_error) / dt
        d_term = self.kd * derivative

        # Total Output Kontroler
        output = p_term + i_term + d_term
        output_clamped = np.clip(output, self.min_out, self.max_out)

        self.prev_error = error
        self.prev_time = current_time
        return int(output_clamped)

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = time.time()


# =============================================================================
# MOCK SERIAL DRIVER (SIMULASI KOMUNIKASI MIKROKONTROLER)
# =============================================================================
class RobotSerialInterface:
    """
    Mengirimkan paket kendali arah motor ke mikrokontroler (Arduino/ESP32).
    Format Paket UART: <CMD, STEER_PWM, THROTTLE_PWM, TARGET_FLAG>
    """
    def __init__(self, port="COM3", baudrate=115200, simulation_mode=True):
        self.simulation_mode = simulation_mode
        self.last_packet = ""

    def send_command(self, steer_val, throttle_val, target_locked):
        flag = 1 if target_locked else 0
        # Format string terstandarisasi untuk parser Serial.readStringUntil('\n') di Arduino
        packet = f"<NAV,{steer_val:+04d},{throttle_val:+04d},{flag}>\n"
        self.last_packet = packet.strip()
        return self.last_packet


# =============================================================================
# FUNGSI UTAMA (CAPSTONE VISION PIPELINE)
# =============================================================================
def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sample_video_path = os.path.join(current_dir, "images", "capstone_tracking_sample.mp4")

    print("=================================================================")
    print("  CAPSTONE PROJECT: ROBOT VISION CONTROLLER TERPADU DENGAN PID   ")
    print("=================================================================")
    print("Pilih Sumber Kamera:")
    print(" [1] Gunakan Kamera Webcam Fisik (Device Index 0)")
    print(" [2] Gunakan File Video Simulasi Target (capstone_tracking_sample.mp4)")
    print("-----------------------------------------------------------------")

    cap = cv2.VideoCapture(0)
    source_name = "Webcam (Index 0)"

    if not cap.isOpened():
        print("[WARNING] Kamera webcam fisik tidak terdeteksi.")
        print(f"[INFO] Menggunakan file video simulasi capstone: {sample_video_path}")
        cap = cv2.VideoCapture(sample_video_path)
        source_name = "Simulasi (capstone_tracking_sample.mp4)"

    if not cap.isOpened():
        print("[ERROR] Gagal membuka video feed!")
        return

    # Inisialisasi PID Controller untuk Sumbu Kemudi Horisontal (Steering)
    # Output: -100 (Belok Kiri Penuh) s/d +100 (Belok Kanan Penuh)
    steer_pid = PIDController(kp=0.35, ki=0.005, kd=0.12, output_limits=(-100, 100))

    # Inisialisasi PID Controller untuk Throttle Jarak Maju/Mundur
    # Target radius bola di layar = 45 px (jarak ideal robot dari bola)
    DESIRED_TARGET_RADIUS_PX = 45
    throttle_pid = PIDController(kp=1.2, ki=0.02, kd=0.20, output_limits=(-100, 100))

    # Inisialisasi Antarmuka Komunikasi Serial Robot
    serial_robot = RobotSerialInterface(port="COM3", baudrate=115200, simulation_mode=True)

    # Rentang Warna Target (Bola Oranye) dalam HSV
    lower_target = np.array([5, 110, 110], dtype=np.uint8)
    upper_target = np.array([24, 255, 255], dtype=np.uint8)

    # Deadzone Toleransi (px) - mencegah getaran mikroskopis saat robot sudah lurus
    DEADZONE_X_PX = 20
    DEADZONE_RADIUS_PX = 4

    prev_loop_time = time.time()
    snapshot_captured = False

    print("\n[PETUNJUK KONTROL]")
    print(" Tekan 'r' : Reset akumulator PID")
    print(" Tekan 's' : Ambil snapshot visualisasi kokpit")
    print(" Tekan 'q' atau ESC : Berhenti dan matikan sistem robotik\n")

    while True:
        start_time = time.time()
        ret, frame = cap.read()

        if not ret:
            if "Simulasi" in source_name:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break

        h, w = frame.shape[:2]
        center_screen_x = w // 2
        center_screen_y = h // 2

        # =====================================================================
        # 1. SEGMENTASI WARNA & MORFOLOGI
        # =====================================================================
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_target, upper_target)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

        # =====================================================================
        # 2. EKSTRAKSI KONTUR & CENTROID TARGET TERBESAR
        # =====================================================================
        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        target_locked = False
        target_center = None
        target_radius = 0
        error_x = 0
        error_dist = 0

        if contours:
            # Ambil kontur dengan luas area terbesar (mengabaikan pantulan/noise kecil)
            largest_cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_cnt)

            if area > 400:
                ((cx, cy), target_radius) = cv2.minEnclosingCircle(largest_cnt)
                target_center = (int(cx), int(cy))
                target_radius = int(target_radius)
                target_locked = True

                # Hitung deviasi kesalahan (Error Spasial)
                error_x = target_center[0] - center_screen_x
                error_dist = DESIRED_TARGET_RADIUS_PX - target_radius

        # =====================================================================
        # 3. KALKULASI PENGENDALI PID
        # =====================================================================
        if target_locked:
            # Periksa deadzone sumbu X
            active_error_x = 0 if abs(error_x) <= DEADZONE_X_PX else error_x
            steer_output = steer_pid.update(active_error_x)

            # Periksa deadzone jarak
            active_error_dist = 0 if abs(error_dist) <= DEADZONE_RADIUS_PX else error_dist
            throttle_output = throttle_pid.update(active_error_dist)
        else:
            # Target hilang -> Robot berhenti atau mode pencarian berputar perlahan
            steer_output = 0
            throttle_output = 0
            steer_pid.reset()
            throttle_pid.reset()

        # 4. Kirim Paket Protokol Serial UART ke Mikrokontroler
        serial_packet = serial_robot.send_command(steer_output, throttle_output, target_locked)

        # =====================================================================
        # 5. VISUALISASI KOKPIT ROBOT & HUD (HEADS-UP DISPLAY)
        # =====================================================================
        hud = frame.copy()

        # Garis Pembidik Tengah (Crosshair Reticle)
        cv2.line(hud, (center_screen_x, center_screen_y - 25), (center_screen_x, center_screen_y + 25), (100, 255, 100), 1)
        cv2.line(hud, (center_screen_x - 25, center_screen_y), (center_screen_x + 25, center_screen_y), (100, 255, 100), 1)
        # Kotak Deadzone Toleransi (Garis Putus-putus Kuning)
        cv2.rectangle(hud,
                      (center_screen_x - DEADZONE_X_PX, center_screen_y - 40),
                      (center_screen_x + DEADZONE_X_PX, center_screen_y + 40),
                      (0, 255, 255), 1)

        if target_locked:
            # Gambar Lingkaran Pembidik Target
            cv2.circle(hud, target_center, target_radius, (0, 140, 255), 2)
            cv2.circle(hud, target_center, 4, (0, 0, 255), -1)

            # Garis Vektor Deviasi dari Tengah Layar ke Objek
            cv2.line(hud, (center_screen_x, center_screen_y), target_center, (0, 255, 255), 2)

            # Label status penguncian objek
            cv2.putText(hud, f"TARGET LOCKED (R:{target_radius}px)", (target_center[0] - 60, target_center[1] - target_radius - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 2)
        else:
            cv2.putText(hud, "SEARCHING TARGET...", (center_screen_x - 90, center_screen_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        # Perhitungan Metrik FPS & Latensi
        now = time.time()
        fps = 1.0 / (now - prev_loop_time) if (now - prev_loop_time) > 0 else 0
        prev_loop_time = now
        latency_ms = (time.time() - start_time) * 1000.0

        # Panel HUD Informasi Robot (Pojok Kiri Atas)
        cv2.rectangle(hud, (15, 15), (420, 145), (25, 25, 25), -1)
        cv2.rectangle(hud, (15, 15), (420, 145), (0, 255, 0) if target_locked else (0, 0, 255), 2)

        status_text = "STATUS: TRACKING ON-LINE" if target_locked else "STATUS: TARGET LOST"
        cv2.putText(hud, status_text, (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if target_locked else (0, 0, 255), 2)
        cv2.putText(hud, f"Steer PID Output    : {steer_output:+04d} %  (Err X: {error_x:+04d} px)", (25, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(hud, f"Throttle PID Output : {throttle_output:+04d} %  (Err R: {error_dist:+03d} px)", (25, 87),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(hud, f"UART Serial Packet  : {serial_packet}", (25, 109),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(hud, f"FPS: {fps:.1f} | Latency: {latency_ms:.1f} ms | Src: {source_name[:18]}", (25, 131),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1)

        # Simpan snapshot visualisasi pertama saat target terkunci untuk aset dokumentasi README
        if target_locked and not snapshot_captured:
            out_img_path = os.path.join(current_dir, "images", "hasil_capstone_controller.png")
            cv2.imwrite(out_img_path, hud)
            snapshot_captured = True
            print(f"[INFO] Snapshot Capstone Vision Controller disimpan ke: {out_img_path}")

        # Tampilkan Jendela Kokpit Robotik
        cv2.imshow("Robot Vision Controller - Capstone Project", hud)

        # Penanganan Keyboard
        key = cv2.waitKey(15) & 0xFF
        if key == ord('q') or key == 27:
            print("[INFO] Capstone Controller dimatikan oleh operator.")
            break
        elif key == ord('r'):
            steer_pid.reset()
            throttle_pid.reset()
            print("[INFO] Pengendali PID berhasil di-reset.")
        elif key == ord('s'):
            snap_path = os.path.join(current_dir, "images", "capstone_manual_snapshot.jpg")
            cv2.imwrite(snap_path, hud)
            print(f"[SNAPSHOT] Foto manual disimpan ke: {snap_path}")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Sistem Penglihatan Robot Terpadu selesai beroperasi.")

if __name__ == "__main__":
    main()

