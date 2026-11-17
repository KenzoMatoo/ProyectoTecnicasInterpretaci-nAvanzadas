#include <opencv2/opencv.hpp>
#include <opencv2/geometry.hpp>  
#include <algorithm>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

const int ANCHO = 700; 
const int ALTO = 900;

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

int main(int argc, char** argv) {
    int indice = (argc > 1) ? atoi(argv[1]) : 0;

    VideoCapture cap(indice);
    if (!cap.isOpened()) {
        cerr << "No se pudo abrir la cámara " << indice << ". Prueba con otro índice.\n";
        return 1;
    }

    cout << "Cámara abierta. Teclas: q/ESC = salir, s = guardar hoja\n";

    Mat frame, gris, difuminada, bordes, hojaPlana;

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            cerr << "No se recibió imagen de la cámara.\n";
            break;
        }

        // 1) Preprocesamiento
        cvtColor(frame, gris, COLOR_BGR2GRAY);
        GaussianBlur(gris, difuminada, Size(5, 5), 0);
        Canny(difuminada, bordes, 50, 150);
        // Cierra pequeños huecos en los bordes para que el contorno no se corte
        dilate(bordes, bordes, getStructuringElement(MORPH_RECT, Size(3, 3)));

        // 2) Buscar la hoja (al menos 10% del área de la imagen)
        double areaMinima = 0.10 * frame.cols * frame.rows;
        vector<Point> hoja = encontrarHoja(bordes, areaMinima);

        Mat vista = frame.clone();

        if (!hoja.empty()) {
            // 3) Dibujar el contorno y las esquinas sobre la imagen en vivo
            vector<vector<Point>> poly = { hoja };
            polylines(vista, poly, true, Scalar(0, 255, 0), 3);

            vector<Point2f> origen = ordenarPuntos(hoja);
            const char* etiquetas[] = { "1", "2", "3", "4" };
            for (int i = 0; i < 4; i++) {
                circle(vista, origen[i], 8, Scalar(0, 0, 255), FILLED);
                putText(vista, etiquetas[i], origen[i] + Point2f(10, -10),
                    FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2);
            }

            // 4) Corregir perspectiva
            vector<Point2f> destino = {
                {0, 0}, {ANCHO - 1.f, 0}, {ANCHO - 1.f, ALTO - 1.f}, {0, ALTO - 1.f} };
            Mat M = getPerspectiveTransform(origen, destino);
            warpPerspective(frame, hojaPlana, M, Size(ANCHO, ALTO));

            imshow("Hoja aplanada (perspectiva)", hojaPlana);
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
            imwrite("hoja.png", hojaPlana);
            cout << "Hoja guardada en hoja.png\n";
        }
    }

    cap.release();
    destroyAllWindows();
    return 0;
}