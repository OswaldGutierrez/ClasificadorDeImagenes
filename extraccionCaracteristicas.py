import cv2
import numpy as np
import os
import pickle
from pathlib import Path
from skimage.feature import hog, local_binary_pattern

# ============================================================
# Configuración general
# ============================================================

TAMANO_IMAGEN = (128, 128)

DATASETS = {
    "gatosPerros": ["gatos", "perros"],
    "kiwisBananas": ["kiwis", "bananas"]
}

PARTICIONES = ["train", "test"]

# Parámetros HOG
HOG_PIXELES_POR_CELDA  = (8, 8)
HOG_CELDAS_POR_BLOQUE  = (2, 2)
HOG_ORIENTACIONES      = 9

# Parámetros LBP
LBP_RADIO              = 1
LBP_PUNTOS             = 8 * LBP_RADIO   # 8 vecinos
LBP_METODO             = "uniform"


# ============================================================
# Funciones de extracción
# ============================================================

def extraerHog(imagen):
    imagenGris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    caracteristicas = hog(
        imagenGris,
        orientations=HOG_ORIENTACIONES,
        pixels_per_cell=HOG_PIXELES_POR_CELDA,
        cells_per_block=HOG_CELDAS_POR_BLOQUE,
        block_norm="L2-Hys",
        visualize=False
    )
    return caracteristicas


def extraerLbp(imagen):

    imagenGris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    lbpImagen = local_binary_pattern(
        imagenGris,
        P=LBP_PUNTOS,
        R=LBP_RADIO,
        method=LBP_METODO
    )

    # Número de patrones posibles para método uniform
    numPatrones = LBP_PUNTOS + 2
    histograma, _ = np.histogram(
        lbpImagen.ravel(),
        bins=numPatrones,
        range=(0, numPatrones),
        density=True
    )
    return histograma


def extraerCaracteristicas(imagen):
    vectorHog = extraerHog(imagen)
    vectorLbp = extraerLbp(imagen)
    return np.concatenate([vectorHog, vectorLbp])


# ============================================================
# Procesamiento del dataset
# ============================================================

def procesarDataset(rutaBase, nombreDataset):
    rutaBase = Path(rutaBase)
    clases = DATASETS[nombreDataset]

    caracteristicasHog = {"train": [], "test": []}
    caracteristicasLbp = {"train": [], "test": []}
    etiquetas          = {"train": [], "test": []}

    extensionesValidas = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for particion in PARTICIONES:
        print(f"\n  Partición: {particion}")

        for idxClase, clase in enumerate(clases):
            rutaClase = rutaBase / nombreDataset / particion / clase

            if not rutaClase.exists():
                print(f"    [OMITIDO] No existe: {rutaClase}")
                continue

            archivos = [
                f for f in rutaClase.iterdir()
                if f.suffix.lower() in extensionesValidas
            ]

            print(f"    Clase '{clase}' (etiqueta {idxClase}): {len(archivos)} imágenes")

            for archivo in archivos:
                imagen = cv2.imread(str(archivo))
                if imagen is None:
                    print(f"      [ADVERTENCIA] No se pudo cargar: {archivo.name}")
                    continue

                # Asegurar tamaño correcto
                imagen = cv2.resize(imagen, TAMANO_IMAGEN, interpolation=cv2.INTER_AREA)

                caracteristicasHog[particion].append(extraerHog(imagen))
                caracteristicasLbp[particion].append(extraerLbp(imagen))
                etiquetas[particion].append(idxClase)

        # Convertir a arrays numpy
        caracteristicasHog[particion] = np.array(caracteristicasHog[particion])
        caracteristicasLbp[particion] = np.array(caracteristicasLbp[particion])
        etiquetas[particion]          = np.array(etiquetas[particion])

    return caracteristicasHog, caracteristicasLbp, etiquetas


def guardarCaracteristicas(caracteristicasHog, caracteristicasLbp, etiquetas, rutaSalida, nombreDataset):
    rutaDestino = Path(rutaSalida) / "caracteristicas" / nombreDataset
    rutaDestino.mkdir(parents=True, exist_ok=True)

    archivos = {
        "hog_train.pkl" : (caracteristicasHog["train"], etiquetas["train"]),
        "hog_test.pkl"  : (caracteristicasHog["test"],  etiquetas["test"]),
        "lbp_train.pkl" : (caracteristicasLbp["train"], etiquetas["train"]),
        "lbp_test.pkl"  : (caracteristicasLbp["test"],  etiquetas["test"]),
    }

    for nombreArchivo, (caracteristicas, etqs) in archivos.items():
        rutaArchivo = rutaDestino / nombreArchivo
        with open(rutaArchivo, "wb") as f:
            pickle.dump({"caracteristicas": caracteristicas, "etiquetas": etqs}, f)
        print(f"    Guardado: {rutaArchivo}")


def mostrarResumen(caracteristicasHog, caracteristicasLbp, etiquetas, nombreDataset):
    """Muestra un resumen de los vectores extraídos."""
    print(f"\n{'='*55}")
    print(f"  Resumen: {nombreDataset}")
    print(f"{'='*55}")
    for particion in PARTICIONES:
        print(f"\n  {particion}:")
        print(f"    HOG  — shape: {caracteristicasHog[particion].shape}")
        print(f"    LBP  — shape: {caracteristicasLbp[particion].shape}")
        print(f"    Etiquetas — shape: {etiquetas[particion].shape}")
        clases = DATASETS[nombreDataset]
        for idx, clase in enumerate(clases):
            cantidad = np.sum(etiquetas[particion] == idx)
            print(f"    Clase '{clase}': {cantidad} imágenes")
    print(f"{'='*55}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    rutaDatosPreprocesados = r"./datosPreprocesados"
    rutaProyecto           = r"./"

    for nombreDataset in DATASETS:
        print(f"\n{'='*55}")
        print(f"  Extrayendo características: {nombreDataset}")
        print(f"{'='*55}")

        caracteristicasHog, caracteristicasLbp, etiquetas = procesarDataset(
            rutaDatosPreprocesados,
            nombreDataset
        )

        guardarCaracteristicas(
            caracteristicasHog,
            caracteristicasLbp,
            etiquetas,
            rutaProyecto,
            nombreDataset
        )

        mostrarResumen(caracteristicasHog, caracteristicasLbp, etiquetas, nombreDataset)

    print("\nExtracción de características completada.")