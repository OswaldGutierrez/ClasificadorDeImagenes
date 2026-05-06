import numpy as np
import pickle
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# Configuración general
# ============================================================

DATASETS = {
    "gatosPerros": ["gatos", "perros"],
    "kiwisBananas": ["kiwis", "bananas"]
}

DESCRIPTORES = ["hog", "lbp"]


# ============================================================
# Funciones de carga
# ============================================================

def cargarModelo(rutaProyecto, nombreDataset, descriptor):
    rutaArchivo = (
        Path(rutaProyecto)
        / "modelos"
        / nombreDataset
        / f"svm_{descriptor}.pkl"
    )
    if not rutaArchivo.exists():
        raise FileNotFoundError(f"No se encontró el modelo: {rutaArchivo}")

    with open(rutaArchivo, "rb") as f:
        pipeline = pickle.load(f)

    return pipeline


def cargarCaracteristicas(rutaProyecto, nombreDataset, descriptor, particion):
    rutaArchivo = (
        Path(rutaProyecto)
        / "caracteristicas"
        / nombreDataset
        / f"{descriptor}_{particion}.pkl"
    )
    if not rutaArchivo.exists():
        raise FileNotFoundError(f"No se encontró: {rutaArchivo}")

    with open(rutaArchivo, "rb") as f:
        datos = pickle.load(f)

    return datos["caracteristicas"], datos["etiquetas"]


# ============================================================
# Cálculo de métricas
# ============================================================

def calcularMetricas(yReal, yPredicho, nombresClases):
    matrizConfusion_ = confusion_matrix(yReal, yPredicho)

    falsosPositivos = {}
    falsosNegativos = {}
    for i, clase in enumerate(nombresClases):
        falsosPositivos[clase] = int(matrizConfusion_[:, i].sum() - matrizConfusion_[i, i])
        falsosNegativos[clase] = int(matrizConfusion_[i, :].sum() - matrizConfusion_[i, i])

    metricas = {
        "accuracy"        : accuracy_score(yReal, yPredicho) * 100,
        "precision"       : precision_score(yReal, yPredicho, average="weighted") * 100,
        "recall"          : recall_score(yReal, yPredicho, average="weighted") * 100,
        "f1Score"         : f1_score(yReal, yPredicho, average="weighted") * 100,
        "matrizConfusion" : matrizConfusion_,
        "falsosPositivos" : falsosPositivos,
        "falsosNegativos" : falsosNegativos,
        "reporteClases"   : classification_report(
            yReal, yPredicho,
            target_names=nombresClases,
            digits=4
        )
    }

    return metricas


# ============================================================
# Visualización con Pillow
# ============================================================

def cargarFuente(tamano):
    try:
        return ImageFont.truetype("arial.ttf", tamano)
    except:
        return ImageFont.load_default()


def graficarMatrizConfusion(matrizConfusion, nombresClases, titulo, rutaGuardado):
    celdaAncho, celdaAlto = 160, 120
    margenIzq, margenSup  = 140, 100
    numClases  = len(nombresClases)
    anchoTotal = margenIzq + celdaAncho * numClases + 20
    altoTotal  = margenSup + celdaAlto  * numClases + 60

    img  = Image.new("RGB", (anchoTotal, altoTotal), "white")
    draw = ImageDraw.Draw(img)

    fuente       = cargarFuente(16)
    fuenteTitulo = cargarFuente(14)

    tw = draw.textlength(titulo, font=fuenteTitulo)
    draw.text(((anchoTotal - tw) // 2, 15), titulo, fill="black", font=fuenteTitulo)

    valorMax = matrizConfusion.max()

    for i in range(numClases):
        for j in range(numClases):
            x0 = margenIzq + j * celdaAncho
            y0 = margenSup + i * celdaAlto
            x1 = x0 + celdaAncho
            y1 = y0 + celdaAlto

            intensidad = int(255 - (matrizConfusion[i, j] / valorMax) * 180)
            color = (intensidad, intensidad, 255)
            draw.rectangle([x0, y0, x1, y1], fill=color, outline="gray")

            valor = str(matrizConfusion[i, j])
            tw    = draw.textlength(valor, font=fuente)
            draw.text(
                (x0 + celdaAncho // 2 - tw // 2, y0 + celdaAlto // 2 - 10),
                valor, fill="black", font=fuente
            )

    for j, clase in enumerate(nombresClases):
        x  = margenIzq + j * celdaAncho + celdaAncho // 2
        tw = draw.textlength(clase, font=fuente)
        draw.text((x - tw // 2, margenSup - 30), clase, fill="black", font=fuente)

    for i, clase in enumerate(nombresClases):
        y = margenSup + i * celdaAlto + celdaAlto // 2 - 10
        draw.text((5, y), clase, fill="black", font=fuente)

    twPred = draw.textlength("Predicción", font=fuente)
    draw.text(((anchoTotal - twPred) // 2, altoTotal - 35), "Predicción", fill="black", font=fuente)
    draw.text((5, margenSup + (numClases * celdaAlto) // 2 - 10), "Real", fill="black", font=fuente)

    img.save(str(rutaGuardado))
    print(f"    Matriz de confusión guardada: {rutaGuardado}")


def graficarComparacionMetricas(resultadosTodos, rutaGuardado):
    etiquetas, accuracies, precisiones, recalls, f1Scores = [], [], [], [], []

    for nombreDataset, descriptores in resultadosTodos.items():
        for descriptor, metricas in descriptores.items():
            etiquetas.append(f"{nombreDataset}\n{descriptor.upper()}")
            accuracies.append(metricas["accuracy"])
            precisiones.append(metricas["precision"])
            recalls.append(metricas["recall"])
            f1Scores.append(metricas["f1Score"])

    numGrupos  = len(etiquetas)
    anchoTotal = 120 + numGrupos * 300
    altoTotal  = 520
    margenIzq  = 80
    margenSup  = 50
    areaAlto   = 350
    anchoBarra = 40
    espaciado  = 10

    img  = Image.new("RGB", (anchoTotal, altoTotal), "white")
    draw = ImageDraw.Draw(img)

    fuente       = cargarFuente(13)
    fuenteTitulo = cargarFuente(15)

    titulo = "Comparación de métricas por modelo"
    twTit  = draw.textlength(titulo, font=fuenteTitulo)
    draw.text(((anchoTotal - twTit) // 2, 12), titulo, fill="black", font=fuenteTitulo)

    colores = {
        "Accuracy" : (76,  114, 176),
        "Precisión": (85,  168, 104),
        "Recall"   : (196,  78,  82),
        "F1-Score" : (129, 114, 178)
    }
    seriesData = [
        ("Accuracy",  accuracies),
        ("Precisión", precisiones),
        ("Recall",    recalls),
        ("F1-Score",  f1Scores),
    ]

    for pct in [25, 50, 75, 100]:
        y = margenSup + areaAlto - int((pct / 110) * areaAlto)
        draw.line([(margenIzq, y), (anchoTotal - 20, y)], fill=(220, 220, 220), width=1)
        draw.text((margenIzq - 40, y - 8), f"{pct}%", fill="gray", font=fuente)

    for grupoIdx, etiqueta in enumerate(etiquetas):
        baseX = margenIzq + grupoIdx * 300 + 20
        for serieIdx, (nombre, valores) in enumerate(seriesData):
            valor     = valores[grupoIdx]
            altoBarra = int((valor / 110) * areaAlto)
            x0 = baseX + serieIdx * (anchoBarra + espaciado)
            y0 = margenSup + areaAlto - altoBarra
            x1 = x0 + anchoBarra
            y1 = margenSup + areaAlto
            draw.rectangle([x0, y0, x1, y1], fill=colores[nombre], outline="gray")
            draw.text((x0 + 2, y0 - 18), f"{valor:.1f}", fill="black", font=fuente)

        lineas = etiqueta.split("\n")
        yEtiq  = margenSup + areaAlto + 10
        for linea in lineas:
            tw = draw.textlength(linea, font=fuente)
            draw.text((baseX + 80 - tw // 2, yEtiq), linea, fill="black", font=fuente)
            yEtiq += 18

    leyendaX = margenIzq
    leyendaY = altoTotal - 35
    for nombre, color in colores.items():
        draw.rectangle([leyendaX, leyendaY, leyendaX + 18, leyendaY + 16], fill=color, outline="gray")
        draw.text((leyendaX + 22, leyendaY), nombre, fill="black", font=fuente)
        leyendaX += 130

    img.save(str(rutaGuardado))
    print(f"    Gráfica comparativa guardada: {rutaGuardado}")


# ============================================================
# Evaluación completa
# ============================================================

def evaluarTodosLosModelos(rutaProyecto):
    rutaResultados = Path(rutaProyecto) / "resultados"
    rutaResultados.mkdir(parents=True, exist_ok=True)

    resultadosTodos = {}

    for nombreDataset, clases in DATASETS.items():
        resultadosTodos[nombreDataset] = {}

        print(f"\n{'='*55}")
        print(f"  Dataset: {nombreDataset}")
        print(f"{'='*55}")

        for descriptor in DESCRIPTORES:
            print(f"\n  Descriptor: {descriptor.upper()}")

            pipeline     = cargarModelo(rutaProyecto, nombreDataset, descriptor)
            xTest, yReal = cargarCaracteristicas(rutaProyecto, nombreDataset, descriptor, "test")

            yPredicho = pipeline.predict(xTest)
            metricas  = calcularMetricas(yReal, yPredicho, clases)

            print(f"    Accuracy  : {metricas['accuracy']:.2f}%")
            print(f"    Precisión : {metricas['precision']:.2f}%")
            print(f"    Recall    : {metricas['recall']:.2f}%")
            print(f"    F1-Score  : {metricas['f1Score']:.2f}%")
            print(f"\n    Falsos Positivos: {metricas['falsosPositivos']}")
            print(f"    Falsos Negativos: {metricas['falsosNegativos']}")
            print(f"\n    Reporte por clase:\n")
            print(metricas["reporteClases"])

            tituloMatriz = f"Matriz de Confusión — {nombreDataset} ({descriptor.upper()})"
            rutaMatriz   = rutaResultados / f"matrizConfusion_{nombreDataset}_{descriptor}.png"
            graficarMatrizConfusion(metricas["matrizConfusion"], clases, tituloMatriz, rutaMatriz)

            resultadosTodos[nombreDataset][descriptor] = metricas

    rutaComparacion = rutaResultados / "comparacionMetricas.png"
    graficarComparacionMetricas(resultadosTodos, rutaComparacion)

    return resultadosTodos


def mostrarResumenFinal(resultadosTodos):
    print(f"\n{'='*70}")
    print("  RESUMEN FINAL — MÉTRICAS SOBRE CONJUNTO DE TEST")
    print(f"{'='*70}")
    print(f"  {'Dataset':<20} {'Desc':<6} {'Accuracy':>9} {'Precisión':>10} {'Recall':>8} {'F1':>8}")
    print(f"  {'-'*65}")

    for nombreDataset, descriptores in resultadosTodos.items():
        for descriptor, metricas in descriptores.items():
            print(
                f"  {nombreDataset:<20} {descriptor.upper():<6} "
                f"{metricas['accuracy']:>8.2f}% "
                f"{metricas['precision']:>9.2f}% "
                f"{metricas['recall']:>7.2f}% "
                f"{metricas['f1Score']:>7.2f}%"
            )

    print(f"{'='*70}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    rutaProyecto = r"./"

    resultadosTodos = evaluarTodosLosModelos(rutaProyecto)
    mostrarResumenFinal(resultadosTodos)

    print("\nEvaluación completada. Guardados en carpeta 'resultados/'.")