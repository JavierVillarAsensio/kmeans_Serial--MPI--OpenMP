#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <cfloat>
#include <chrono>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// Function to calculate Euclidean distance between two points
float euclideanDistance(const Mat& point1, const Mat& point2) {
    return norm(point1 - point2);
}

// Function to initialize cluster centers randomly
void initializeCenters(const Mat& data, int K, Mat& centers) {
    int numPoints = data.rows;
    RNG rng;
    for (int i = 0; i < K; ++i) {
        int randomIndex = rng.uniform(0, numPoints);
        data.row(randomIndex).copyTo(centers.row(i));
    }
}

void assignPointsToClusters(const Mat& data, const Mat& centers, Mat& labels) {
    for (int i = 0; i < data.rows; ++i) {
        float minDist = FLT_MAX;
        int minLabel = -1;
        for (int j = 0; j < centers.rows; ++j) {
            float dist = euclideanDistance(data.row(i), centers.row(j));
            if (dist < minDist) {
                minDist = dist;
                minLabel = j;
            }
        }
        labels.at<int>(i, 0) = minLabel;
    }
}

// Function to update cluster centers
void updateCenters(const Mat& data, const Mat& labels, int K, Mat& centers, Mat& newCenters) {
    int numFeatures = data.cols;
    vector<int> clusterCounts(K, 0);
    for (int i = 0; i < data.rows; ++i) {
        int label = labels.at<int>(i);
        newCenters.row(label) += data.row(i);
        clusterCounts[label]++;
    }
    for (int i = 0; i < K; ++i) {
        if (clusterCounts[i] > 0) {
            newCenters.row(i) /= clusterCounts[i];
        }
    }
}

// Function to perform k-means clustering
void customKmeans(InputArray data, int K, InputOutputArray bestLabels, TermCriteria criteria, OutputArray centers) {
    Mat dataMat = data.getMat();
    Mat labels(data.rows(), 1, CV_32S);
    
    // Initialize cluster centers
    Mat initialCenters(K, dataMat.cols, dataMat.type());
    Mat newCenters(K, dataMat.cols, data.type(), Scalar(0));
    initializeCenters(dataMat, K, initialCenters);

    // Perform k-means algorithm
    bool converged = false;
    int iter = 0;
    float epsilon = criteria.epsilon;
    while (!converged && iter < criteria.maxCount) {
        // Assign points to clusters
        assignPointsToClusters(dataMat, initialCenters, labels);
        // Update cluster centers
        updateCenters(dataMat, labels, K, initialCenters, newCenters);
        // Check for convergence
        float maxMovement = FLT_MIN;
        for (int i = 0; i < K; ++i) {
            float dist = euclideanDistance(newCenters.row(i), initialCenters.row(i));
            if (dist > maxMovement) {
                maxMovement = dist;
            }
        }
        if (maxMovement < epsilon) {
            converged = true;
        }
        newCenters.copyTo(initialCenters);

        iter++;
    }

    // Assign the best labels
    labels.copyTo(bestLabels);
    if (centers.needed()) {
        initialCenters.copyTo(centers);
    }
}

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
    //TEST HIPERPARAMETERS TIME
    // Define different hyperparameters for testing
    /*
    std::vector<int> num_clusters_list = {3, 5, 7, 10}; // Different number of clusters
    std::vector<int> term_criteria_count_list = {10, 50, 100}; // Different term criteria count
    std::vector<double> term_criteria_eps_list = {0.2, 0.5, 1.0}; // Different term criteria epsilon

    for (int num_clusters : num_clusters_list) {
        for (int term_criteria_count : term_criteria_count_list) {
            for (double term_criteria_eps : term_criteria_eps_list) {
                auto start = std::chrono::high_resolution_clock::now();
                customKmeans(data, num_clusters, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, term_criteria_count, term_criteria_eps), centers);
                auto end = std::chrono::high_resolution_clock::now();

                std::chrono::duration<double> duration = end - start;
                auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
                auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration - seconds);

                // Output the results
                std::cout << "Number of clusters: " << num_clusters << ", Attempts: " << term_criteria_count
                          << ", Epsilon: " << term_criteria_eps << ", Time taken: " << seconds.count() << " sec and " << milliseconds.count() << " ms. "<< std::endl;
            }
        }
    }
    */

    // TEST FUNCTION ONE TIME
    auto start = std::chrono::high_resolution_clock::now();
    customKmeans(data, 5, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 50, 0.2), centers);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;

    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration - seconds);

    std::cout << "Custom KMeans time: " << seconds.count() << " seconds and " << milliseconds.count() << " milliseconds." << std::endl;

    //Reorganizamos las etiquetas en 715 filas y 1096 columnas.
    //Recordar que los píxeles en el fichero estaban escritos en orde de
    //las columnas. Así, al hacer posteriormente la trasnpuesta obtendremos la imagen correcta
    labels = labels.reshape(1,715);
 

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

