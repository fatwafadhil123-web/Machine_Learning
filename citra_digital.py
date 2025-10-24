import cv2
import numpy as np
import matplotlib.pyplot as plt

#baca gambar grayscale
gambar = cv2.imread('unpam.png',0)#pastikan gambar ada di file

#hitung histogram
histogram = cv2.calcHist([gambar],[0], None, [256], [0,256])

#tampilkan histogram
plt.figure()
plt.title('Histogram original image')
plt.xlabel('Intensitas Pixel')
plt.ylabel('Jumlah pixel')
plt.plot(histogram)
plt.xlim(0,256)
plt.show()


import cv2
import matplotlib.pyplot as plt
import numpy as np

# 1. MEMUAT GAMBAR TERLEBIH DAHULU
# Ganti 'path_to_your_image.jpg' dengan path gambar Anda
gambar = cv2.imread('unpam.png', cv2.IMREAD_GRAYSCALE)

# Periksa apakah gambar berhasil dimuat
if gambar is None:
    print("Error: Gambar tidak dapat dimuat!")
    exit()

# 2. EKUALISASI HISTOGRAM
gambar_ekualisasi = cv2.equalizeHist(gambar)

# 3. Tampilkan gambar asli dan hasil equalisasi
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.title('Gambar Asli')
plt.imshow(gambar, cmap='gray')  
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Gambar Ekualisasi')
plt.imshow(gambar_ekualisasi, cmap='gray')  
plt.axis('off')

plt.tight_layout()
plt.show()

# 4. Tampilkan histogram
plt.figure(figsize=(12, 5))

# Histogram gambar asli
plt.subplot(1, 2, 1)
histogram_asli = cv2.calcHist([gambar], [0], None, [256], [0, 256])
plt.title('Histogram Gambar Asli')
plt.plot(histogram_asli)
plt.xlim([0, 256])
plt.grid(True)

plt.subplot(1, 2, 2)
histogram_ekualisasi = cv2.calcHist([gambar_ekualisasi], [0], None, [256], [0, 256])
plt.title('Histogram Gambar Ekualisasi')
plt.plot(histogram_ekualisasi)
plt.xlim([0, 256])
plt.grid(True)

plt.tight_layout()
plt.show()