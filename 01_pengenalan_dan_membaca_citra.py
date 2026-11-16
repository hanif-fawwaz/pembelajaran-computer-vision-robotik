"""
=============================================================================
Modul 01: Pengenalan dan Membaca Citra Digital dengan OpenCV
=============================================================================
Deskripsi:
Program ini mengajarkan dasar paling fundamental dalam Computer Vision:
1. Cara membaca citra dari file (cv2.imread)
2. Memeriksa properti citra (dimensi, resolusi, tipe data)
3. Membaca citra dalam mode Grayscale
4. Menampilkan citra di jendela window (cv2.imshow, cv2.waitKey, cv2.destroyAllWindows)
5. Menyimpan kembali citra ke format file lain (cv2.imwrite)

Penulis: Pembelajaran Robotik & Vision
=============================================================================
"""

import os
import cv2

def main():
    # 1. Menentukan path gambar target
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_dir, "images", "sample.jpg")

    print("[INFO] Memeriksa file gambar di:", image_path)

    # 2. Membaca gambar berwarna (Default: cv2.IMREAD_COLOR / format BGR)
    img_color = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Validasi: Selalu periksa apakah gambar berhasil dimuat
    if img_color is None:
        print("[ERROR] Gambar tidak ditemukan! Pastikan path file sudah benar.")
        return

    # 3. Membaca gambar dalam mode Grayscale (Abu-abu / 1 channel)
    # Mode ini sangat sering digunakan dalam robotik untuk mempercepat pemrosesan
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 4. Memeriksa Informasi / Properti Citra (Citra adalah array NumPy)
    print("\n--- INFORMASI CITRA BERWARNA ---")
    height, width, channels = img_color.shape
    print(f"Tinggi (Height)   : {height} piksel")
    print(f"Lebar (Width)     : {width} piksel")
    print(f"Jumlah Channel    : {channels} (B, G, R)")
    print(f"Tipe Data         : {img_color.dtype} (umumnya uint8: 0 s/d 255)")
    print(f"Total Piksel      : {img_color.size}")

    print("\n--- INFORMASI CITRA GRAYSCALE ---")
    h_gray, w_gray = img_gray.shape
    print(f"Resolusi          : {w_gray} x {h_gray}")
    print(f"Jumlah Channel    : 1 (Nilai intensitas cahaya 0=Hitam, 255=Putih)")

    # 5. Menampilkan Gambar ke Jendela GUI
    # Parameter: ("Nama Jendela", variabel_gambar)
    cv2.imshow("1. Citra Asli (BGR Color)", img_color)
    cv2.imshow("2. Citra Grayscale (Hitam-Putih)", img_gray)

    # 6. Menyimpan hasil citra Grayscale ke file baru
    output_path = os.path.join(current_dir, "images", "sample_grayscale.png")
    cv2.imwrite(output_path, img_gray)
    print(f"\n[INFO] Berhasil menyimpan citra grayscale ke: {output_path}")

    print("\n[PETUNJUK] Klik jendela gambar lalu tekan tombol APAPUN pada keyboard untuk menutup jendela...")

    # cv2.waitKey(0) artinya menunggu sampai pengguna menekan sembarang tombol pada keyboard
    # Angka 0 berarti menunggu tanpa batas waktu (infinity)
    cv2.waitKey(0)

    # Menutup semua jendela OpenCV yang terbuka agar tidak membebani memori
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

