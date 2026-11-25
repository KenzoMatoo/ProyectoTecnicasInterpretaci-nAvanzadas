#include <opencv2/opencv.hpp>
#if __has_include(<opencv2/geometry.hpp>) // solo existe en OpenCV 5
#include <opencv2/geometry.hpp>
#endif
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

using namespace cv;
using namespace std;

const int ANCHO = 700; 
const int ALTO = 900;

// Formato de la hoja: 122 preguntas (A-D) en 6 columnas de hasta 21 filas.
// Coordenadas medidas sobre la hoja ya aplanada a ANCHO x ALTO.
const int PREGUNTAS = 122;
const int OPCIONES = 4;
const int FILAS_POR_COLUMNA = 21;
const float COLUMNA_X[] = { 59.f, 170.f, 279.f, 388.5f, 499.5f, 609.5f }; // centro de la opción A
const float FILA_Y = 405.5f;    // centro de la primera fila
const float PASO_Y = 22.95f;    // distancia entre filas
const float PASO_X = 18.5f;     // distancia entre opciones
const int RADIO_BURBUJA = 9;

struct Burbuja {
    int pregunta;   // 1 a PREGUNTAS
    int opcion;     // 0 = A, 1 = B, 2 = C, 3 = D
    Point2f centro;
};

// Ordena 4 puntos como: arriba-izq, arriba-der, abajo-der, abajo-izq
vector<Point2f> ordenarPuntos(const vector<Point>& pts) {
    vector<Point2f> r(4);
    auto suma = [](const Point& p) { return p.x + p.y; };
    auto resta = [](const Point& p) { return p.y - p.x; };

    r[0] = *min_element(pts.begin(), pts.end(), [&](auto& a, auto& b) { return suma(a) < suma(b); });
    r[2] = *max_element(pts.begin(), pts.end(), [&](auto& a, auto& b) { return suma(a) < suma(b); });
    r[1] = *min_element(pts.begin(), pts.end(), [&](auto& a, auto& b) { return resta(a) < resta(b); });
    r[3] = *max_element(pts.begin(), pts.end(), [&](auto& a, auto& b) { return resta(a) < resta(b); });
    return r;
}

// Busca el contorno más grande con 4 esquinas y un área mínima
vector<Point> encontrarHoja(const Mat& bordes, double areaMinima) {
    vector<vector<Point>> contornos;
    findContours(bordes.clone(), contornos, RETR_LIST, CHAIN_APPROX_SIMPLE);

    sort(contornos.begin(), contornos.end(), [](const auto& a, const auto& b) {
        return contourArea(a) > contourArea(b);
        });

    for (const auto& c : contornos) {
        if (contourArea(c) < areaMinima) break; // los demás son más chicos
        double peri = arcLength(c, true);
        vector<Point> aprox;
        approxPolyDP(c, aprox, 0.02 * peri, true);
        if (aprox.size() == 4 && isContourConvex(aprox)) return aprox;
    }
    return {};
}

// Detecta los bordes de la imagen y busca la hoja (al menos 10% del área)
vector<Point> detectarHoja(const Mat& imagen, Mat& bordes) {
    Mat gris, difuminada;
    cvtColor(imagen, gris, COLOR_BGR2GRAY);
    GaussianBlur(gris, difuminada, Size(5, 5), 0);
    Canny(difuminada, bordes, 50, 150);
    // Cierra pequeños huecos en los bordes para que el contorno no se corte
    dilate(bordes, bordes, getStructuringElement(MORPH_RECT, Size(3, 3)));

    return encontrarHoja(bordes, 0.10 * imagen.cols * imagen.rows);
}

// Corrige la perspectiva y deja la hoja en ANCHO x ALTO
Mat aplanarHoja(const Mat& imagen, const vector<Point2f>& esquinas) {
    vector<Point2f> destino = {
        {0, 0}, {ANCHO - 1.f, 0}, {ANCHO - 1.f, ALTO - 1.f}, {0, ALTO - 1.f} };
    Mat M = getPerspectiveTransform(esquinas, destino);
    Mat plana;
    warpPerspective(imagen, plana, M, Size(ANCHO, ALTO));
    return plana;
}

// Paso 1: tinta/lápiz en blanco y papel en negro. El umbral adaptativo compara
// cada píxel con su vecindario, así tolera sombras y luz desigual.
Mat binarizarHoja(const Mat& hojaPlana) {
    Mat gris, binaria;
    cvtColor(hojaPlana, gris, COLOR_BGR2GRAY);
    GaussianBlur(gris, gris, Size(3, 3), 0);
    // El bloque (51) es mayor que una burbuja para que las rellenas queden sólidas
    adaptiveThreshold(gris, binaria, 255, ADAPTIVE_THRESH_GAUSSIAN_C,
        THRESH_BINARY_INV, 51, 15);
    return binaria;
}

// Paso 2: calcula la posición de las burbujas de todas las preguntas
vector<Burbuja> generarCuadricula() {
    vector<Burbuja> burbujas;
    burbujas.reserve(PREGUNTAS * OPCIONES);
    for (int p = 0; p < PREGUNTAS; p++) {
        int columna = p / FILAS_POR_COLUMNA;
        int fila = p % FILAS_POR_COLUMNA;
        for (int o = 0; o < OPCIONES; o++) {
            Point2f centro(COLUMNA_X[columna] + o * PASO_X, FILA_Y + fila * PASO_Y);
            burbujas.push_back({ p + 1, o, centro });
        }
    }
    return burbujas;
}

// Dibuja la cuadrícula encima de una copia de la imagen (color o binaria)
Mat dibujarCuadricula(const Mat& imagen, const vector<Burbuja>& burbujas) {
    Mat vista;
    if (imagen.channels() == 1) cvtColor(imagen, vista, COLOR_GRAY2BGR);
    else vista = imagen.clone();

    for (const auto& b : burbujas)
        circle(vista, b.centro, RADIO_BURBUJA, Scalar(0, 200, 0), 1, LINE_AA);
    return vista;
}

void guardarResultados(const Mat& hojaPlana, const Mat& binaria, const Mat& cuadricula) {
    imwrite("hoja.png", hojaPlana);
    imwrite("resultado_binaria.png", binaria);
    imwrite("resultado_cuadricula.png", cuadricula);
    cout << "Guardados: hoja.png, resultado_binaria.png, resultado_cuadricula.png\n";
}

// Modo de prueba sin cámara. Una imagen de ANCHO x ALTO se toma como hoja ya
// aplanada; cualquier otra se trata como foto y primero se busca la hoja.
int probarImagen(const string& ruta) {
    Mat imagen = imread(ruta);
    if (imagen.empty()) {
        cerr << "No se pudo leer la imagen " << ruta << ".\n";
        return 1;
    }

    Mat hojaPlana;
    if (imagen.cols == ANCHO && imagen.rows == ALTO) {
        hojaPlana = imagen;
    }
    else {
        Mat bordes;
        vector<Point> hoja = detectarHoja(imagen, bordes);
        if (hoja.empty()) {
            cerr << "No se encontró la hoja en " << ruta << ".\n";
            return 1;
        }
        hojaPlana = aplanarHoja(imagen, ordenarPuntos(hoja));
    }

    vector<Burbuja> burbujas = generarCuadricula();
    Mat binaria = binarizarHoja(hojaPlana);
    Mat cuadricula = dibujarCuadricula(hojaPlana, burbujas);

    cout << "Cuadrícula: " << PREGUNTAS << " preguntas, " << burbujas.size() << " burbujas\n";
    imwrite("resultado_binaria.png", binaria);
    imwrite("resultado_cuadricula.png", cuadricula);
    cout << "Guardados: resultado_binaria.png, resultado_cuadricula.png\n";

    // Se muestran a escala para que la hoja completa quepa en la pantalla
    for (const char* ventana : { "Binarizada", "Cuadricula" }) {
        namedWindow(ventana, WINDOW_NORMAL);
        resizeWindow(ventana, 490, 630);
    }
    imshow("Binarizada", dibujarCuadricula(binaria, burbujas));
    imshow("Cuadricula", cuadricula);
    // Lado a lado y al frente, para que no queden tapadas por otras ventanas
    moveWindow("Cuadricula", 20, 20);
    moveWindow("Binarizada", 530, 20);
    setWindowProperty("Cuadricula", WND_PROP_TOPMOST, 1);
    setWindowProperty("Binarizada", WND_PROP_TOPMOST, 1);
    cout << "Presiona una tecla sobre una ventana para salir.\n";
    waitKey(0);
    destroyAllWindows();
    return 0;
}

int main(int argc, char** argv) {
    if (argc > 2 && string(argv[1]) == "--imagen") return probarImagen(argv[2]);

    int indice = (argc > 1) ? atoi(argv[1]) : 0;
    vector<Burbuja> burbujas = generarCuadricula();

    VideoCapture cap(indice);
    if (!cap.isOpened()) {
        cerr << "No se pudo abrir la cámara " << indice << ". Prueba con otro índice.\n";
        return 1;
    }

    cout << "Cámara abierta. Teclas: q/ESC = salir, s = guardar hoja\n";

    Mat frame, bordes, hojaPlana, binaria, cuadricula;

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            cerr << "No se recibió imagen de la cámara.\n";
            break;
        }

        // 1) Preprocesar y buscar la hoja
        vector<Point> hoja = detectarHoja(frame, bordes);

        Mat vista = frame.clone();

        if (!hoja.empty()) {
            // 2) Dibujar el contorno y las esquinas sobre la imagen en vivo
            vector<vector<Point>> poly = { hoja };
            polylines(vista, poly, true, Scalar(0, 255, 0), 3);

            vector<Point2f> origen = ordenarPuntos(hoja);
            const char* etiquetas[] = { "1", "2", "3", "4" };
            for (int i = 0; i < 4; i++) {
                circle(vista, origen[i], 8, Scalar(0, 0, 255), FILLED);
                putText(vista, etiquetas[i], origen[i] + Point2f(10, -10),
                    FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2);
            }

            // 3) Corregir perspectiva, binarizar y ubicar las burbujas
            hojaPlana = aplanarHoja(frame, origen);
            binaria = binarizarHoja(hojaPlana);
            cuadricula = dibujarCuadricula(hojaPlana, burbujas);

            imshow("Hoja aplanada (cuadricula)", cuadricula);
            imshow("Hoja binarizada", binaria);
        }
        else {
            putText(vista, "Buscando hoja...", Point(20, 40),
                FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2);
        }

        imshow("Camara (contorno detectado)", vista);
        imshow("Bordes (Canny)", bordes); 

        int tecla = waitKey(1) & 0xFF;
        if (tecla == 'q' || tecla == 27) break;
        if (tecla == 's' && !hojaPlana.empty()) {
            guardarResultados(hojaPlana, binaria, cuadricula);
        }
    }

    cap.release();
    destroyAllWindows();
    return 0;
}