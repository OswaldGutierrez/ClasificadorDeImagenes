import cv2
import os
import numpy as np
from pathlib import Path

# Configuración general
TAMANO_IMAGEN = (128, 128)
TAMANO_KERNEL_GAUSSIANO = (5, 5)
SIGMA_GAUSSIANO = 0

DATASETS = {
    "gatosPerros": ["gatos", "perros"],
    "kiwisBananas": ["kiwis", "bananas"]
}

PARTICIONES = ["train", "test"]

def cargarImagen(rutaImagen):
    imagen = cv2.imread(str(rutaImagen))
    if imagen is None:
        print(f"  [ADVERTENCIA] No se pudo cargar: {rutaImagen}")
    return imagen

def aplicarResize(imagen, tamano=TAMANO_IMAGEN):
    return cv2.resize(imagen, tamano, interpolation=cv2.INTER_AREA)

def aplicarFiltroGaussiano(imagen, tamanoKernel=TAMANO_KERNEL_GAUSSIANO, sigma=SIGMA_GAUSSIANO):
    return cv2.GaussianBlur(imagen, tamanoKernel, sigma)

def preprocesarImagen(rutaImagen):
    imagen = cargarImagen(rutaImagen)
    if imagen is None:
        return None

    imagenRedimensionada = aplicarResize(imagen)
    imagenFiltrada = aplicarFiltroGaussiano(imagenRedimensionada)

    return imagenFiltrada

def preprocesarDataset(rutaBase):
    rutaBase = Path(rutaBase)
    rutaSalida = rutaBase.parent / "datosPreprocesados"

    totalProcesadas = 0
    totalErrores = 0

    for nombreDataset, clases in DATASETS.items():
        for particion in PARTICIONES:
            for clase in clases:
                rutaEntrada = rutaBase / nombreDataset / particion / clase
                rutaDestino = rutaSalida / nombreDataset / particion / clase

                if not rutaEntrada.exists():
                    print(f"[OMITIDO] Carpeta no encontrada: {rutaEntrada}")
                    continue

                rutaDestino.mkdir(parents=True, exist_ok=True)

                extensionesValidas = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
                archivos = [
                    f for f in rutaEntrada.iterdir()
                    if f.suffix.lower() in extensionesValidas
                ]

                print(f"\nProcesando: {nombreDataset}/{particion}/{clase} ({len(archivos)} imágenes)")

                for archivo in archivos:
                    imagenProcesada = preprocesarImagen(archivo)

                    if imagenProcesada is not None:
                        rutaGuardado = rutaDestino / archivo.name
                        cv2.imwrite(str(rutaGuardado), imagenProcesada)
                        totalProcesadas += 1
                    else:
                        totalErrores += 1

    print(f"\n{'='*50}")
    print(f"Preprocesamiento completado.")
    print(f"  Imágenes procesadas: {totalProcesadas}")
    print(f"  Errores:             {totalErrores}")
    print(f"  Guardadas en:        {rutaSalida}")
    print(f"{'='*50}")

def verificarPreprocesamiento(rutaBase):
    rutaPreprocesados = Path(rutaBase).parent / "datosPreprocesados"

    if not rutaPreprocesados.exists():
        print("No se encontró la carpeta datosPreprocesados. Ejecuta primero el preprocesamiento.")
        return

    print(f"\n{'='*50}")
    print("Verificación de imágenes preprocesadas:")
    print(f"{'='*50}")

    for nombreDataset in DATASETS:
        print(f"\n  Dataset: {nombreDataset}")
        for particion in PARTICIONES:
            for clase in DATASETS[nombreDataset]:
                ruta = rutaPreprocesados / nombreDataset / particion / clase
                if ruta.exists():
                    cantidad = len(list(ruta.iterdir()))
                    print(f"    {particion}/{clase}: {cantidad} imágenes")
                else:
                    print(f"    {particion}/{clase}: carpeta no encontrada")


if __name__ == "__main__":
    rutaDatos = "D:\Proyectos\ClasificacionDeImagenes\datasets"

    preprocesarDataset(rutaDatos)
    verificarPreprocesamiento(rutaDatos)