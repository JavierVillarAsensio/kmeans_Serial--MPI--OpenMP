#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void readDataFromFile(const string& filename, Mat& matrix) {
  
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Unable to open file " << filename << endl;
        return; // Return empty matrix if file cannot be opened
    }

    // Leemos los píxeles hiperespectrales uno a uno
    string line;
    unsigned int row = 0, col = 0;
    while (getline(file, line)) {
        istringstream iss(line);
        string element;
        while (getline(iss, element, ',')) {

            float value;
            stringstream(element) >> value;
            matrix.at<float>(row, col) = value;
  

            ++col;
            if (col >= matrix.size().width) {
                col = 0;
                ++row;
            }

            if (row >= matrix.size().height)
                break;
        }
    }

    file.close();
    return;
}

int main() {
  //Cubo hiperespectral de entrada
    string filename = "pavia.txt";
    Mat data(783640,102,CV_32F); 
    //Son 1096x715 pixeles, cada uno con 102 componentes.
    cout << "Data size: " << data.size() << endl;
 
    readDataFromFile(filename,data);

    // K-means
    Mat labels, centers;
    cout << "Starting K-means" << endl;

    auto start = std::chrono::high_resolution_clock::now();
    kmeans(data, 5, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 100, 0.2), 10, KMEANS_RANDOM_CENTERS, centers);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration - seconds);

    std::cout << "OpenCV KMeans time: " << seconds.count() << " seconds and " << milliseconds.count() << " milliseconds." << std::endl;

    cout << "label size: " << labels.size() << endl;

    //Reorganizamos las etiquetas en 715 filas y 1096 columnas.
    //Recordar que los píxeles en el fichero estaban escritos en orde de
    //las columnas. Así, al hacer posteriormente la trasnpuesta obtendremos la imagen correcta
    labels = labels.reshape(1,715);
    cout << "label size after rehape: " << labels.size() << endl;
 

    cout << "centers size: " << centers.size() << endl;
  
    Mat cluster_map(715,1096, CV_8U);

    cout << "cluster map size: " << cluster_map.size() << endl;

    int index = 0;
    int prev_label = 0;

    for (int i = 0; i < cluster_map.rows; ++i) {
      for (int j = 0; j < cluster_map.cols; ++j) {  
            int l = cluster_map.at<uchar>(i, j) = labels.at<int>(i,j);
            if (prev_label != cluster_map.at<unsigned char>(i, j)){
                prev_label = cluster_map.at<uchar>(i, j);
            }
        }   
    }
    
    cout << "Reshaped matrix size: " << cluster_map.size() << endl;
   
    
    Mat result_image(715,1096, CV_8UC3);

    //  Mapa de color (B, G, R) en OpenCV
    const Vec<uchar, 3> colors[] = {
        Vec<uchar, 3>(0, 0, 255),   // Red
        Vec<uchar, 3>(0, 255, 0),   // Green
        Vec<uchar, 3>(255, 0, 0),   // Blue
        Vec<uchar, 3>(255, 255, 0), // Yellow
        Vec<uchar, 3>(255, 0, 255), // Magenta
        Vec<uchar, 3>(0, 255, 255), // Cyan
        Vec<uchar, 3>(255, 255, 255), // White
        Vec<uchar, 3>(0, 0, 0),     // Black
        Vec<uchar, 3>(128, 128, 128), // Gray
	    Vec<uchar, 3>(64, 64, 64) // Gray
    };

    // Asignar a cada pixel de la imagen de salida el color en finción de la etiqueta
    index = 0;
    for (int i = 0; i < result_image.rows; ++i) {
      for (int j = 0; j < result_image.cols; ++j) {
	uchar cluster_index = cluster_map.at<unsigned char>(index++);
	result_image.at<Vec3b>(i, j) = colors[cluster_index];
      }
    }

    //Mostrar la imagen
    Mat resultT;
    transpose(result_image,resultT); 
    imshow("Clustered Image", resultT);
    imwrite("paviakmeanscpp.jpg",resultT);
    waitKey(0);

    return 0;
}
