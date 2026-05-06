import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# =========================
# CONFIGURACIÓN
# =========================
ruta_base = "./resultados/filtros"
imagen_path = "imagen.jpg"

# =========================
# FUNCIONES
# =========================
def crear_carpeta(filtro):
    path = os.path.join(ruta_base, filtro, "resultado")
    os.makedirs(path, exist_ok=True)
    return path

def guardar_grid(nombre_filtro, lista_imagenes):
    carpeta = crear_carpeta(nombre_filtro)

    cols = len(lista_imagenes)
    plt.figure(figsize=(5 * cols, 5))

    for i, (titulo, img) in enumerate(lista_imagenes):
        plt.subplot(1, cols, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(titulo)
        plt.axis('off')

    plt.tight_layout()
    ruta = os.path.join(carpeta, f"{nombre_filtro}.png")
    plt.savefig(ruta)
    plt.close()

    print(f"{nombre_filtro} guardado en: {ruta}")

# =========================
# CARGAR IMAGEN
# =========================
OGimg = cv2.imread(imagen_path)
img = cv2.cvtColor(OGimg, cv2.COLOR_BGR2GRAY)

# =========================
# MEDIA
# =========================
media = [
    ("3x3", cv2.blur(img, (3,3))),
    ("7x7", cv2.blur(img, (7,7)))
]
guardar_grid("Media", media)

# =========================
# MEDIANA
# =========================
mediana = [
    ("3x3", cv2.medianBlur(img, 3)),
    ("7x7", cv2.medianBlur(img, 7))
]
guardar_grid("Mediana", mediana)

# =========================
# LOG
# =========================
img_float = img / 255.0
c = 1 / np.log(1 + np.max(img_float))
log_img = c * np.log(1 + img_float)
log_img = np.uint8(log_img * 255)

guardar_grid("Log", [("Log", log_img)])

# =========================
# KERNEL PROMEDIO
# =========================
kernel = [
    ("3x3", cv2.filter2D(img, -1, np.ones((3,3), np.float32)/9)),
    ("7x7", cv2.filter2D(img, -1, np.ones((7,7), np.float32)/49))
]
guardar_grid("Kernel", kernel)

# =========================
# GAUSS
# =========================
gauss = [
    ("3x3", cv2.GaussianBlur(img, (3,3), 0)),
    ("7x7", cv2.GaussianBlur(img, (7,7), 1))
]
guardar_grid("Gauss", gauss)

# =========================
# LAPLACE
# =========================
laplace = cv2.Laplacian(img, cv2.CV_64F)
laplace = np.uint8(np.absolute(laplace))
sharpen = cv2.subtract(img, laplace)

laplace_imgs = [
    ("Laplace", laplace),
    ("Sharpen", sharpen)
]
guardar_grid("Laplace", laplace_imgs)

# =========================
# SOBEL
# =========================
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

# Convertir a valores visibles
sobelx_abs = np.uint8(np.absolute(sobelx))
sobely_abs = np.uint8(np.absolute(sobely))

# Magnitud
magnitud = np.sqrt(sobelx**2 + sobely**2)
magnitud = np.uint8(255 * magnitud / np.max(magnitud))

# Aplicado (realce tipo Laplace/Canny)
sobel_aplicado = cv2.subtract(img, magnitud)

sobel_imgs = [
    ("Sobel X", sobelx_abs),
    ("Sobel Y", sobely_abs),
    ("Magnitud", magnitud),
    ("Aplicado", sobel_aplicado)
]

guardar_grid("Sobel", sobel_imgs)

# =========================
# CANNY
# =========================
canny = cv2.Canny(img, 100, 200)
resultado = cv2.subtract(img, canny)

canny_imgs = [
    ("Bordes", canny),
    ("Aplicado", resultado)
]
guardar_grid("Canny", canny_imgs)