import numpy as np
import pickle
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ============================================================
# Configuración general
# ============================================================

DATASETS = {
    "gatosPerros": ["gatos", "perros"],
    "kiwisBananas": ["kiwis", "bananas"]
}

DESCRIPTORES = ["hog", "lbp"]

# Parámetros SVM
SVM_KERNEL = "rbf"
SVM_C      = 10      # penalización por error
SVM_GAMMA  = "scale" # escala automática según los datos


# ============================================================
# Funciones de carga y guardado
# ============================================================

def cargarCaracteristicas(rutaCaracteristicas, nombreDataset, descriptor, particion):
    rutaArchivo = (
        Path(rutaCaracteristicas)
        / "caracteristicas"
        / nombreDataset
        / f"{descriptor}_{particion}.pkl"
    )

    if not rutaArchivo.exists():
        raise FileNotFoundError(f"No se encontró: {rutaArchivo}")

    with open(rutaArchivo, "rb") as f:
        datos = pickle.load(f)

    return datos["caracteristicas"], datos["etiquetas"]


def guardarModelo(pipeline, rutaModelos, nombreDataset, descriptor):
    rutaDestino = Path(rutaModelos) / "modelos" / nombreDataset
    rutaDestino.mkdir(parents=True, exist_ok=True)

    rutaArchivo = rutaDestino / f"svm_{descriptor}.pkl"
    with open(rutaArchivo, "wb") as f:
        pickle.dump(pipeline, f)

    print(f"    Modelo guardado: {rutaArchivo}")
    return rutaArchivo


# ============================================================
# Entrenamiento
# ============================================================

def entrenarSvm(xTrain, yTrain):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(
            kernel=SVM_KERNEL,
            C=SVM_C,
            gamma=SVM_GAMMA,
            probability=True  # necesario para obtener probabilidades
        ))
    ])

    pipeline.fit(xTrain, yTrain)
    return pipeline


def entrenarTodosLosModelos(rutaProyecto):
    resultados = {}

    for nombreDataset in DATASETS:
        resultados[nombreDataset] = {}

        print(f"\n{'='*55}")
        print(f"  Dataset: {nombreDataset}")
        print(f"{'='*55}")

        for descriptor in DESCRIPTORES:
            print(f"\n  Descriptor: {descriptor.upper()}")

            # Cargar datos
            xTrain, yTrain = cargarCaracteristicas(rutaProyecto, nombreDataset, descriptor, "train")
            xTest,  yTest  = cargarCaracteristicas(rutaProyecto, nombreDataset, descriptor, "test")

            print(f"    Train — {xTrain.shape[0]} imágenes, vector de {xTrain.shape[1]} características")
            print(f"    Test  — {xTest.shape[0]} imágenes, vector de {xTest.shape[1]} características")

            # Entrenar
            print(f"    Entrenando SVM (kernel={SVM_KERNEL}, C={SVM_C})...")
            pipeline = entrenarSvm(xTrain, yTrain)

            # Accuracy rápida sobre train y test
            accuracyTrain = pipeline.score(xTrain, yTrain) * 100
            accuracyTest  = pipeline.score(xTest,  yTest)  * 100
            print(f"    Accuracy train : {accuracyTrain:.2f}%")
            print(f"    Accuracy test  : {accuracyTest:.2f}%")

            # Guardar modelo
            rutaModelo = guardarModelo(pipeline, rutaProyecto, nombreDataset, descriptor)

            resultados[nombreDataset][descriptor] = {
                "accuracyTrain" : accuracyTrain,
                "accuracyTest"  : accuracyTest,
                "rutaModelo"    : str(rutaModelo)
            }

    return resultados


def mostrarResumenFinal(resultados):
    print(f"\n{'='*55}")
    print("  RESUMEN FINAL DE ENTRENAMIENTO")
    print(f"{'='*55}")
    print(f"  {'Dataset':<20} {'Descriptor':<8} {'Train':>8} {'Test':>8}")
    print(f"  {'-'*48}")

    for nombreDataset, descriptores in resultados.items():
        for descriptor, metricas in descriptores.items():
            print(
                f"  {nombreDataset:<20} {descriptor.upper():<8} "
                f"{metricas['accuracyTrain']:>7.2f}% "
                f"{metricas['accuracyTest']:>7.2f}%"
            )

    print(f"{'='*55}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    rutaProyecto = r"./"

    resultados = entrenarTodosLosModelos(rutaProyecto)
    mostrarResumenFinal(resultados)

    print("\nEntrenamiento completado.")