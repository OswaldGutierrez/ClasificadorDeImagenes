import tkinter as tk
from tkinter import filedialog, messagebox
import pickle
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageTk
from skimage.feature import hog, local_binary_pattern

# ============================================================
# Configuración general
# ============================================================

RUTA_PROYECTO = r"D:\Proyectos\ClasificacionDeImagenes"

TAMANO_IMAGEN        = (128, 128)
TAMANO_KERNEL_GAUSS  = (5, 5)
SIGMA_GAUSSIANO      = 0

HOG_PIXELES_POR_CELDA = (8, 8)
HOG_CELDAS_POR_BLOQUE = (2, 2)
HOG_ORIENTACIONES     = 9

LBP_RADIO   = 1
LBP_PUNTOS  = 8 * LBP_RADIO
LBP_METODO  = "uniform"

DATASETS = {
    "gatosPerros": ["gatos", "perros"],
    "kiwisBananas": ["kiwis", "bananas"]
}

DESCRIPTORES = ["hog", "lbp"]

COLOR_FONDO       = "#1e1e2e"
COLOR_PANEL       = "#2a2a3e"
COLOR_ACENTO      = "#7c6af7"
COLOR_ACENTO2     = "#56cfb2"
COLOR_TEXTO       = "#e0e0f0"
COLOR_TEXTO_GRIS  = "#888aaa"
COLOR_BOTON       = "#7c6af7"
COLOR_BOTON_HOVER = "#5a4fd4"
COLOR_EXITO       = "#56cfb2"
COLOR_ERROR       = "#f77c7c"


# ============================================================
# Preprocesamiento y extracción de características
# ============================================================

def preprocesarImagen(rutaImagen):
    imagen = cv2.imread(str(rutaImagen))
    if imagen is None:
        return None
    imagen = cv2.resize(imagen, TAMANO_IMAGEN, interpolation=cv2.INTER_AREA)
    imagen = cv2.GaussianBlur(imagen, TAMANO_KERNEL_GAUSS, SIGMA_GAUSSIANO)
    return imagen


def extraerHog(imagen):
    imagenGris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    return hog(
        imagenGris,
        orientations=HOG_ORIENTACIONES,
        pixels_per_cell=HOG_PIXELES_POR_CELDA,
        cells_per_block=HOG_CELDAS_POR_BLOQUE,
        block_norm="L2-Hys",
        visualize=False
    )


def extraerLbp(imagen):
    imagenGris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    lbpImagen  = local_binary_pattern(imagenGris, P=LBP_PUNTOS, R=LBP_RADIO, method=LBP_METODO)
    numPatrones = LBP_PUNTOS + 2
    histograma, _ = np.histogram(lbpImagen.ravel(), bins=numPatrones, range=(0, numPatrones), density=True)
    return histograma


# ============================================================
# Carga de modelos
# ============================================================

def cargarTodosLosModelos(rutaProyecto):
    modelos = {}
    for nombreDataset in DATASETS:
        modelos[nombreDataset] = {}
        for descriptor in DESCRIPTORES:
            rutaArchivo = (
                Path(rutaProyecto)
                / "modelos"
                / nombreDataset
                / f"svm_{descriptor}.pkl"
            )
            if rutaArchivo.exists():
                with open(rutaArchivo, "rb") as f:
                    modelos[nombreDataset][descriptor] = pickle.load(f)
            else:
                modelos[nombreDataset][descriptor] = None
    return modelos


# ============================================================
# Interfaz gráfica
# ============================================================

class InterfazClasificador:

    def __init__(self, raiz):
        self.raiz     = raiz
        self.modelos  = cargarTodosLosModelos(RUTA_PROYECTO)
        self.rutaImagenActual = None

        self.raiz.title("Clasificador de Imágenes")
        self.raiz.configure(bg=COLOR_FONDO)
        self.raiz.resizable(False, False)

        self.construirInterfaz()
        self.centrarVentana(900, 620)

    def centrarVentana(self, ancho, alto):
        self.raiz.update_idletasks()
        x = (self.raiz.winfo_screenwidth()  - ancho) // 2
        y = (self.raiz.winfo_screenheight() - alto)  // 2
        self.raiz.geometry(f"{ancho}x{alto}+{x}+{y}")

    def construirInterfaz(self):
        # Título
        frameTitulo = tk.Frame(self.raiz, bg=COLOR_FONDO)
        frameTitulo.pack(fill="x", padx=30, pady=(25, 10))

        tk.Label(
            frameTitulo, text="Clasificador de Imágenes",
            bg=COLOR_FONDO, fg=COLOR_ACENTO,
            font=("Segoe UI", 22, "bold")
        ).pack(side="left")

        tk.Label(
            frameTitulo, text="HOG + LBP  |  SVM",
            bg=COLOR_FONDO, fg=COLOR_TEXTO_GRIS,
            font=("Segoe UI", 11)
        ).pack(side="left", padx=(14, 0), pady=(8, 0))

        # Contenedor principal
        frameContenido = tk.Frame(self.raiz, bg=COLOR_FONDO)
        frameContenido.pack(fill="both", expand=True, padx=30, pady=5)

        # Panel izquierdo — imagen
        self.framePanelIzq = tk.Frame(frameContenido, bg=COLOR_PANEL, bd=0, relief="flat")
        self.framePanelIzq.pack(side="left", fill="both", expand=True, padx=(0, 12))

        self.labelImagen = tk.Label(
            self.framePanelIzq,
            text="No hay imagen cargada",
            bg=COLOR_PANEL, fg=COLOR_TEXTO_GRIS,
            font=("Segoe UI", 11)
        )
        self.labelImagen.pack(padx=16, pady=16, ipadx=160, ipady=130)

        self.labelNombreArchivo = tk.Label(
            self.framePanelIzq, text="",
            bg=COLOR_PANEL, fg=COLOR_TEXTO_GRIS,
            font=("Segoe UI", 9), wraplength=320
        )
        self.labelNombreArchivo.pack(pady=(0, 10))

        # Panel derecho — controles y resultados
        framePanelDer = tk.Frame(frameContenido, bg=COLOR_FONDO)
        framePanelDer.pack(side="right", fill="y", padx=(0, 0))

        # Sección: selección de dataset
        self.construirSeccion(framePanelDer, "Dataset")

        self.varDataset = tk.StringVar(value="gatosPerros")
        for nombreDataset, clases in DATASETS.items():
            etiqueta = f"{nombreDataset}  ({' vs '.join(clases)})"
            tk.Radiobutton(
                framePanelDer, text=etiqueta,
                variable=self.varDataset, value=nombreDataset,
                bg=COLOR_FONDO, fg=COLOR_TEXTO,
                selectcolor=COLOR_PANEL,
                activebackground=COLOR_FONDO,
                font=("Segoe UI", 10),
                command=self.limpiarResultados
            ).pack(anchor="w", padx=8, pady=2)

        # Sección: selección de descriptor
        self.construirSeccion(framePanelDer, "Descriptor")

        self.varDescriptor = tk.StringVar(value="hog")
        for descriptor in DESCRIPTORES:
            tk.Radiobutton(
                framePanelDer, text=descriptor.upper(),
                variable=self.varDescriptor, value=descriptor,
                bg=COLOR_FONDO, fg=COLOR_TEXTO,
                selectcolor=COLOR_PANEL,
                activebackground=COLOR_FONDO,
                font=("Segoe UI", 10),
                command=self.limpiarResultados
            ).pack(anchor="w", padx=8, pady=2)

        # Botones
        tk.Frame(framePanelDer, bg=COLOR_FONDO, height=12).pack()

        self.botonCargar = self.crearBoton(
            framePanelDer, "Cargar imagen", self.cargarImagen
        )
        self.botonCargar.pack(fill="x", pady=4)

        self.botonClasificar = self.crearBoton(
            framePanelDer, "Clasificar", self.clasificarImagen,
            color=COLOR_ACENTO2
        )
        self.botonClasificar.pack(fill="x", pady=4)
        self.botonClasificar.config(state="disabled")

        # Sección: resultados
        self.construirSeccion(framePanelDer, "Resultado")

        self.frameResultado = tk.Frame(framePanelDer, bg=COLOR_PANEL, bd=0)
        self.frameResultado.pack(fill="x", pady=4)

        self.labelClase = tk.Label(
            self.frameResultado, text="—",
            bg=COLOR_PANEL, fg=COLOR_TEXTO,
            font=("Segoe UI", 20, "bold")
        )
        self.labelClase.pack(pady=(12, 2))

        self.labelConfianza = tk.Label(
            self.frameResultado, text="",
            bg=COLOR_PANEL, fg=COLOR_TEXTO_GRIS,
            font=("Segoe UI", 10)
        )
        self.labelConfianza.pack(pady=(0, 12))

        # Barra de confianza
        self.canvasConfianza = tk.Canvas(
            self.frameResultado, bg=COLOR_PANEL,
            height=14, width=260, highlightthickness=0
        )
        self.canvasConfianza.pack(pady=(0, 16))

    def construirSeccion(self, padre, titulo):
        tk.Label(
            padre, text=titulo.upper(),
            bg=COLOR_FONDO, fg=COLOR_ACENTO,
            font=("Segoe UI", 8, "bold")
        ).pack(anchor="w", pady=(14, 2))

    def crearBoton(self, padre, texto, comando, color=None):
        colorFondo = color if color else COLOR_BOTON
        boton = tk.Button(
            padre, text=texto,
            bg=colorFondo, fg="white",
            activebackground=COLOR_BOTON_HOVER,
            activeforeground="white",
            font=("Segoe UI", 10, "bold"),
            relief="flat", bd=0,
            padx=12, pady=8,
            cursor="hand2",
            command=comando
        )
        return boton

    def cargarImagen(self):
        rutaImagen = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[
                ("Imágenes", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("Todos los archivos", "*.*")
            ]
        )
        if not rutaImagen:
            return

        self.rutaImagenActual = rutaImagen
        self.limpiarResultados()

        # Cargar y redimensionar con recorte centrado
        imgPil = Image.open(rutaImagen).convert("RGB")
        
        anchoObjetivo, altoObjetivo = 360, 300

        # Escalar manteniendo proporción para que cubra el área completa
        ratioW = anchoObjetivo / imgPil.width
        ratioH = altoObjetivo  / imgPil.height
        ratio  = max(ratioW, ratioH)

        nuevoAncho = int(imgPil.width  * ratio)
        nuevoAlto  = int(imgPil.height * ratio)
        imgPil     = imgPil.resize((nuevoAncho, nuevoAlto), Image.LANCZOS)

        # Recorte centrado
        izq = (nuevoAncho - anchoObjetivo) // 2
        sup = (nuevoAlto  - altoObjetivo)  // 2
        imgPil = imgPil.crop((izq, sup, izq + anchoObjetivo, sup + altoObjetivo))

        self.fotoImagen = ImageTk.PhotoImage(imgPil)
        self.labelImagen.config(image=self.fotoImagen, text="")
        self.labelNombreArchivo.config(text=Path(rutaImagen).name)
        self.botonClasificar.config(state="normal")

    def clasificarImagen(self):
        if not self.rutaImagenActual:
            return

        nombreDataset = self.varDataset.get()
        descriptor    = self.varDescriptor.get()
        modelo        = self.modelos[nombreDataset][descriptor]

        if modelo is None:
            messagebox.showerror(
                "Modelo no encontrado",
                f"No se encontró el modelo para {nombreDataset} / {descriptor.upper()}."
            )
            return

        imagen = preprocesarImagen(self.rutaImagenActual)
        if imagen is None:
            messagebox.showerror("Error", "No se pudo cargar la imagen seleccionada.")
            return

        if descriptor == "hog":
            vector = extraerHog(imagen).reshape(1, -1)
        else:
            vector = extraerLbp(imagen).reshape(1, -1)

        prediccion   = modelo.predict(vector)[0]
        probabilidad = modelo.predict_proba(vector)[0]

        clases       = DATASETS[nombreDataset]
        clasePredicha = clases[prediccion]
        confianza    = probabilidad[prediccion] * 100

        # Mostrar resultado
        self.labelClase.config(text=clasePredicha.upper(), fg=COLOR_EXITO)
        self.labelConfianza.config(
            text=f"Confianza: {confianza:.1f}%  |  {descriptor.upper()}"
        )

        # Barra de confianza
        self.canvasConfianza.delete("all")
        anchoTotal = 260
        anchoBarra = int((confianza / 100) * anchoTotal)
        self.canvasConfianza.create_rectangle(0, 0, anchoTotal, 14, fill="#3a3a55", outline="")
        colorBarra = COLOR_EXITO if confianza >= 70 else (COLOR_ACENTO if confianza >= 50 else COLOR_ERROR)
        self.canvasConfianza.create_rectangle(0, 0, anchoBarra, 14, fill=colorBarra, outline="")

    def limpiarResultados(self):
        self.labelClase.config(text="—", fg=COLOR_TEXTO)
        self.labelConfianza.config(text="")
        self.canvasConfianza.delete("all") if hasattr(self, "canvasConfianza") else None


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    raiz = tk.Tk()
    app  = InterfazClasificador(raiz)
    raiz.mainloop()