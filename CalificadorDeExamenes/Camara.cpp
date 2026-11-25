#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#if __has_include(<opencv2/geometry.hpp>) // solo existe en OpenCV 5
#include <opencv2/geometry.hpp>
#endif
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <fstream>
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

// Paso 3: una burbuja cuenta como marcada si al menos el 70% de un círculo de
// radio 5 alrededor de su centro quedó blanco en la imagen binarizada.
const int RADIO_MUESTRA = 5;
const double UMBRAL_MARCA = 0.70;

const char LETRAS[] = "ABCD";
const int SIN_RESPUESTA = -1;
const int MULTIPLE = -2;

struct Burbuja {
    int pregunta;   // 1 a PREGUNTAS
    int opcion;     // 0 = A, 1 = B, 2 = C, 3 = D
    Point2f centro;
};

struct Calificacion {
    int total = 0;          // preguntas que tiene la clave
    int aciertos = 0;
    int incorrectas = 0;
    int sinRespuesta = 0;
    int multiples = 0;
    double nota = 0.0;      // de 0 a 10
};

// Lo que se obtiene al procesar una hoja ya aplanada
struct Lectura {
    Mat binaria;
    vector<double> rellenos;    // fracción marcada de cada burbuja (PREGUNTAS * OPCIONES)
    vector<int> respuestas;     // opción leída por pregunta, SIN_RESPUESTA o MULTIPLE
    Calificacion calificacion;
    Mat vista;                  // hoja con la cuadrícula y la calificación dibujadas
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

// Paso 3: fracción de píxeles blancos (marcados) dentro de un círculo pequeño
double porcentajeRelleno(const Mat& binaria, Point2f centro) {
    Point c(cvRound(centro.x), cvRound(centro.y));
    Rect zona = Rect(c.x - RADIO_MUESTRA, c.y - RADIO_MUESTRA,
        2 * RADIO_MUESTRA + 1, 2 * RADIO_MUESTRA + 1) & Rect(0, 0, binaria.cols, binaria.rows);
    if (zona.empty()) return 0.0;

    Mat mascara = Mat::zeros(zona.size(), CV_8U);
    circle(mascara, c - zona.tl(), RADIO_MUESTRA, Scalar(255), FILLED);
    Mat dentro;
    bitwise_and(binaria(zona), mascara, dentro);
    int area = countNonZero(mascara);
    return area > 0 ? double(countNonZero(dentro)) / area : 0.0;
}

// Paso 3: decide qué opción marcó el alumno en cada pregunta
vector<int> leerRespuestas(const Mat& binaria, const vector<Burbuja>& burbujas,
    vector<double>& rellenos) {
    rellenos.assign(burbujas.size(), 0.0);
    for (size_t i = 0; i < burbujas.size(); i++)
        rellenos[i] = porcentajeRelleno(binaria, burbujas[i].centro);

    vector<int> respuestas(PREGUNTAS, SIN_RESPUESTA);
    for (int p = 0; p < PREGUNTAS; p++) {
        int marcadas = 0;
        for (int o = 0; o < OPCIONES; o++) {
            if (rellenos[p * OPCIONES + o] >= UMBRAL_MARCA) {
                respuestas[p] = o;
                marcadas++;
            }
        }
        if (marcadas > 1) respuestas[p] = MULTIPLE;
    }
    return respuestas;
}

// Lee la clave de un archivo de texto: una respuesta por línea ("1 A", "1. A" o "A").
// Las líneas vacías o que empiezan con # se ignoran.
bool cargarClave(const string& ruta, vector<int>& clave) {
    clave.clear();
    ifstream archivo(ruta);
    if (!archivo) return false;

    string linea;
    int numeroLinea = 0;
    while (getline(archivo, linea)) {
        numeroLinea++;
        size_t inicio = linea.find_first_not_of(" \t\r");
        if (inicio == string::npos || linea[inicio] == '#') continue;

        // La respuesta es la última palabra de la línea, sin el número ni la puntuación
        size_t fin = linea.find_last_not_of(" \t\r");
        size_t ini = linea.find_last_of(" \t", fin);
        string palabra = linea.substr(ini == string::npos ? 0 : ini + 1, fin - (ini == string::npos ? 0 : ini + 1) + 1);
        size_t letra = palabra.find_first_not_of("0123456789.):-");
        int opcion = SIN_RESPUESTA;
        if (letra != string::npos && letra + 1 == palabra.size()) {
            char m = (char)toupper((unsigned char)palabra[letra]);
            if (m >= 'A' && m <= 'D') opcion = m - 'A';
        }
        if (opcion == SIN_RESPUESTA) {
            cerr << "Clave: la linea " << numeroLinea << " no tiene una letra A-D: \"" << linea << "\"\n";
            return false;
        }
        clave.push_back(opcion);
    }
    if (clave.empty() || (int)clave.size() > PREGUNTAS) {
        cerr << "Clave: debe tener entre 1 y " << PREGUNTAS << " respuestas (tiene " << clave.size() << ").\n";
        return false;
    }
    return true;
}

// Paso 4: compara las respuestas con la clave
Calificacion calificar(const vector<int>& respuestas, const vector<int>& clave) {
    Calificacion c;
    c.total = (int)clave.size();
    for (int p = 0; p < c.total; p++) {
        if (respuestas[p] == clave[p]) c.aciertos++;
        else if (respuestas[p] == SIN_RESPUESTA) c.sinRespuesta++;
        else if (respuestas[p] == MULTIPLE) c.multiples++;
        else c.incorrectas++;
    }
    c.nota = c.total > 0 ? 10.0 * c.aciertos / c.total : 0.0;
    return c;
}

string textoRespuesta(int r) {
    if (r == SIN_RESPUESTA) return "-";
    if (r == MULTIPLE) return "*";
    return string(1, LETRAS[r]);
}

string formatoNota(double nota) {
    char texto[16];
    snprintf(texto, sizeof(texto), "%.1f", nota);
    return texto;
}

// Dibuja el resultado sobre la hoja. Con clave: verde = correcta, rojo = incorrecta,
// azul = respuesta correcta que el alumno no marcó. Sin clave: verde = marcada.
Mat dibujarResultado(const Mat& hojaPlana, const vector<Burbuja>& burbujas,
    const vector<double>& rellenos, const vector<int>& respuestas,
    const vector<int>& clave, const Calificacion& cal) {
    Mat vista = dibujarCuadricula(hojaPlana, burbujas);
    const Scalar VERDE(0, 170, 0), ROJO(0, 0, 230), AZUL(230, 120, 0);
    bool hayClave = !clave.empty();
    int preguntas = hayClave ? (int)clave.size() : PREGUNTAS;

    for (int p = 0; p < preguntas; p++) {
        for (int o = 0; o < OPCIONES; o++) {
            const Point2f& centro = burbujas[p * OPCIONES + o].centro;
            bool marcada = rellenos[p * OPCIONES + o] >= UMBRAL_MARCA;
            if (!hayClave) {
                if (marcada) circle(vista, centro, RADIO_BURBUJA + 1, VERDE, 3, LINE_AA);
            }
            else if (o == clave[p]) {
                if (respuestas[p] == o) circle(vista, centro, RADIO_BURBUJA + 1, VERDE, 3, LINE_AA);
                else circle(vista, centro, RADIO_BURBUJA + 1, AZUL, 2, LINE_AA);
            }
            else if (marcada) {
                circle(vista, centro, RADIO_BURBUJA + 1, ROJO, 3, LINE_AA);
            }
        }
    }

    // Encabezado con el resumen
    rectangle(vista, Rect(0, 0, ANCHO, 58), Scalar(255, 255, 255), FILLED);
    string linea1, linea2;
    if (hayClave) {
        linea1 = "Aciertos: " + to_string(cal.aciertos) + "/" + to_string(cal.total) +
            "   Calificacion: " + formatoNota(cal.nota) + " / 10";
        linea2 = "Verde: correcta  Rojo: incorrecta  Azul: respuesta correcta";
    }
    else {
        int contestadas = 0;
        for (int r : respuestas) if (r >= 0) contestadas++;
        linea1 = "Sin clave: " + to_string(contestadas) + " respuestas leidas";
        linea2 = "Verde: opcion marcada";
    }
    putText(vista, linea1, Point(12, 26), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 2, LINE_AA);
    putText(vista, linea2, Point(12, 49), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(60, 60, 60), 1, LINE_AA);
    return vista;
}

// Pasos 1 a 4 sobre una hoja ya aplanada
Lectura procesarHoja(const Mat& hojaPlana, const vector<Burbuja>& burbujas, const vector<int>& clave) {
    Lectura l;
    l.binaria = binarizarHoja(hojaPlana);
    l.respuestas = leerRespuestas(l.binaria, burbujas, l.rellenos);
    if (!clave.empty()) l.calificacion = calificar(l.respuestas, clave);
    l.vista = dibujarResultado(hojaPlana, burbujas, l.rellenos, l.respuestas, clave, l.calificacion);
    return l;
}

void imprimirResultado(const Lectura& l, const vector<int>& clave) {
    bool hayClave = !clave.empty();
    int preguntas = hayClave ? (int)clave.size() : PREGUNTAS;

    cout << "\nRespuestas (- = sin respuesta, * = varias marcadas";
    cout << (hayClave ? ", X(c) = incorrecta y c es la correcta):\n" : "):\n");
    for (int p = 0; p < preguntas; p++) {
        string celda = to_string(p + 1) + ": " + textoRespuesta(l.respuestas[p]);
        if (hayClave) celda += l.respuestas[p] == clave[p] ? " ok" : " X(" + textoRespuesta(clave[p]) + ")";
        cout << celda << string(celda.size() < 14 ? 14 - celda.size() : 1, ' ');
        if ((p + 1) % 6 == 0 || p + 1 == preguntas) cout << "\n";
    }

    if (hayClave) {
        const Calificacion& c = l.calificacion;
        cout << "\nAciertos: " << c.aciertos << " de " << c.total
            << " | Incorrectas: " << c.incorrectas
            << " | Sin respuesta: " << c.sinRespuesta
            << " | Varias marcadas: " << c.multiples << "\n";
        cout << "Calificacion: " << formatoNota(c.nota) << " / 10\n";
    }
    else {
        cout << "\nNo hay clave: solo se leyeron las respuestas (usa --clave archivo.txt).\n";
    }
}

void guardarCsv(const string& ruta, const Lectura& l, const vector<int>& clave) {
    ofstream csv(ruta);
    bool hayClave = !clave.empty();
    int preguntas = hayClave ? (int)clave.size() : PREGUNTAS;
    csv << (hayClave ? "pregunta,respuesta,correcta,resultado\n" : "pregunta,respuesta\n");
    for (int p = 0; p < preguntas; p++) {
        csv << p + 1 << "," << textoRespuesta(l.respuestas[p]);
        if (hayClave)
            csv << "," << textoRespuesta(clave[p]) << "," << (l.respuestas[p] == clave[p] ? "correcta" : "incorrecta");
        csv << "\n";
    }
    if (hayClave)
        csv << "aciertos," << l.calificacion.aciertos << ",total," << l.calificacion.total
            << "\ncalificacion," << formatoNota(l.calificacion.nota) << "\n";
}

void guardarResultados(const Mat& hojaPlana, const Lectura& l, const vector<int>& clave) {
    imwrite("captura_hoja.png", hojaPlana);
    imwrite("resultado_binaria.png", l.binaria);
    imwrite("resultado_calificacion.png", l.vista);
    guardarCsv("resultado.csv", l, clave);
    cout << "Guardados: captura_hoja.png, resultado_binaria.png, resultado_calificacion.png, resultado.csv\n";
}

// Se muestran a escala para que la hoja completa quepa en la pantalla
void prepararVentanasHoja() {
    for (const char* ventana : { "Resultado", "Binarizada" }) {
        namedWindow(ventana, WINDOW_NORMAL);
        resizeWindow(ventana, 490, 630);
        setWindowProperty(ventana, WND_PROP_TOPMOST, 1);
    }
    moveWindow("Resultado", 20, 20);
    moveWindow("Binarizada", 530, 20);
}

void mostrarLectura(const Lectura& l, const vector<Burbuja>& burbujas) {
    imshow("Resultado", l.vista);
    imshow("Binarizada", dibujarCuadricula(l.binaria, burbujas));
}

// Modo de prueba sin cámara. Una imagen de ANCHO x ALTO se toma como hoja ya
// aplanada; cualquier otra se trata como foto y primero se busca la hoja.
int probarImagen(const string& ruta, const vector<int>& clave) {
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
            cerr << "No se encontro la hoja en " << ruta << ".\n";
            return 1;
        }
        hojaPlana = aplanarHoja(imagen, ordenarPuntos(hoja));
    }

    vector<Burbuja> burbujas = generarCuadricula();
    Lectura l = procesarHoja(hojaPlana, burbujas, clave);
    imprimirResultado(l, clave);
    guardarResultados(hojaPlana, l, clave);

    prepararVentanasHoja();
    mostrarLectura(l, burbujas);
    cout << "Presiona una tecla sobre una ventana para salir.\n";
    waitKey(0);
    destroyAllWindows();
    return 0;
}

// Modo en vivo: cámara (índice) o un archivo de video
int modoEnVivo(VideoCapture& cap, const vector<int>& clave) {
    vector<Burbuja> burbujas = generarCuadricula();
    cout << "Teclas: s = guardar y mostrar la calificacion, q/ESC = salir\n";

    prepararVentanasHoja();
    for (const char* ventana : { "Camara", "Bordes (Canny)" }) {
        namedWindow(ventana, WINDOW_NORMAL);
        resizeWindow(ventana, 480, 360);
    }
    moveWindow("Camara", 1040, 20);
    moveWindow("Bordes (Canny)", 1040, 420);

    Mat frame, bordes, hojaPlana;
    Lectura ultima;
    bool hayLectura = false;

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            cout << "No hay mas imagenes de la camara o del video.\n";
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

            // 3) Corregir perspectiva, leer respuestas y calificar
            hojaPlana = aplanarHoja(frame, origen);
            ultima = procesarHoja(hojaPlana, burbujas, clave);
            hayLectura = true;
            mostrarLectura(ultima, burbujas);
        }
        else {
            putText(vista, "Buscando hoja...", Point(20, 40),
                FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2);
        }

        imshow("Camara", vista);
        imshow("Bordes (Canny)", bordes);

        int tecla = waitKey(1) & 0xFF;
        if (tecla == 'q' || tecla == 27) break;
        if (tecla == 's' && hayLectura) {
            imprimirResultado(ultima, clave);
            guardarResultados(hojaPlana, ultima, clave);
        }
    }

    if (hayLectura && !clave.empty())
        cout << "Ultima lectura: " << ultima.calificacion.aciertos << "/" << ultima.calificacion.total
        << " aciertos, calificacion " << formatoNota(ultima.calificacion.nota) << " / 10\n";
    cap.release();
    destroyAllWindows();
    return 0;
}

// Uso:
//   CalificadorDeExamenes.exe [indice_camara] [--clave clave.txt]
//   CalificadorDeExamenes.exe --imagen foto.png [--clave clave.txt]
//   CalificadorDeExamenes.exe --video video.avi [--clave clave.txt]
// Si no se indica --clave se usa clave.txt cuando existe.
int main(int argc, char** argv) {
    cout << unitbuf; // muestra cada mensaje en cuanto se escribe
    // Oculta los mensajes internos de OpenCV para que la consola solo muestre el resultado
    utils::logging::setLogLevel(utils::logging::LOG_LEVEL_ERROR);
    string imagen, video, rutaClave;
    int indice = 0;
    for (int i = 1; i < argc; i++) {
        string a = argv[i];
        if ((a == "--imagen" || a == "--video" || a == "--clave") && i + 1 >= argc) {
            cerr << "Falta el archivo despues de " << a << ".\n";
            return 1;
        }
        if (a == "--imagen") imagen = argv[++i];
        else if (a == "--video") video = argv[++i];
        else if (a == "--clave") rutaClave = argv[++i];
        else indice = atoi(a.c_str());
    }

    vector<int> clave;
    if (!rutaClave.empty()) {
        if (!cargarClave(rutaClave, clave)) {
            cerr << "No se pudo usar la clave " << rutaClave << ".\n";
            return 1;
        }
    }
    else if (ifstream("clave.txt")) {
        rutaClave = "clave.txt";
        if (!cargarClave(rutaClave, clave)) return 1;
    }
    if (!clave.empty())
        cout << "Clave cargada de " << rutaClave << ": " << clave.size() << " preguntas.\n";
    else
        cout << "Sin clave de respuestas: solo se leeran las respuestas.\n";

    if (!imagen.empty()) return probarImagen(imagen, clave);

    VideoCapture cap;
    if (!video.empty()) cap.open(video);
    else cap.open(indice);
    if (!cap.isOpened()) {
        if (!video.empty()) cerr << "No se pudo abrir el video " << video << ".\n";
        else cerr << "No se pudo abrir la camara " << indice << ". Prueba con otro indice.\n";
        return 1;
    }
    return modoEnVivo(cap, clave);
}
