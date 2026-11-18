#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <chrono>

using namespace cv;
using namespace std;
using namespace std::chrono;

int main()
{
    Mat image = imread("imagen_1000x1000.png", IMREAD_GRAYSCALE);

    if (image.empty()) {
        cout << "No se pudo abrir la imagen." << endl;
        return -1;
    }

    cout << "Imagen cargada: " << image.cols << "x" << image.rows << " pixeles" << endl;

    Mat filtrada(image.rows, image.cols, CV_8UC1, Scalar(0));

    int kernel[3][3] = {
        {1, 1, 1},
        {1, 1, 1},
        {1, 1, 1}
    };
    int pesoTotal = 9;

    auto inicio = high_resolution_clock::now();

    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {

            if (y == 0 || x == 0 || y == image.rows - 1 || x == image.cols - 1) {
                filtrada.at<uchar>(y, x) = image.at<uchar>(y, x);
                continue;
            }

            int suma = 0;
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int pixelVecino = (int)image.at<uchar>(y + ky, x + kx);
                    int coeficiente = kernel[ky + 1][kx + 1];
                    suma += pixelVecino * coeficiente;
                }
            }

            int valor = suma / pesoTotal;
            if (valor > 255) valor = 255;
            if (valor < 0) valor = 0;

            filtrada.at<uchar>(y, x) = (uchar)valor;
        }
    }

    auto fin = high_resolution_clock::now();
    duration<double, std::milli> tiempoMs = fin - inicio;

    cout << "Tiempo de aplicar el filtro 3x3: " << tiempoMs.count() << " ms" << endl;

    imwrite("imagen_filtrada.png", filtrada);
    cout << "Imagen filtrada guardada como imagen_filtrada.png" << endl;

    namedWindow("Original", WINDOW_AUTOSIZE);
    imshow("Original", image);

    namedWindow("Filtrada 3x3", WINDOW_AUTOSIZE);
    imshow("Filtrada 3x3", filtrada);

    cout << "Presiona una tecla sobre una de las ventanas de imagen para cerrar." << endl;
    waitKey(0);
    return 0;
}
