#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

#include <vector>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

#define ROWS 1096
#define COLS 715
#define BANDS 102

#define IMAGE "pavia.txt"
#define RNGS "rng.txt"

#define CONVERGENCE 0.2
#define MAX_ITER 50
#define K 5

#define FLOAT_MAX 3.4028234663852886e+38F
#define FLOAT_MIN 1.1754943508222875e-38F

static size_t n_pixels = ROWS * COLS;
static size_t data_size = ROWS * COLS * BANDS;
static size_t centers_size = K * BANDS;

float euclideanDistance(float *data, float *centers, int pixel, int center){
    float distance = 0;
    int data_offset = pixel * BANDS;
    int center_offset = center * BANDS;

    for(int i = 0; i < BANDS; i++)
        distance += pow(data[data_offset + i] - centers[center_offset + i], 2);

    return sqrt(distance);
}

int readDataFromFile(float *data){
    int data_count = ROWS * COLS * BANDS;

    FILE *file;
    if((file = fopen(IMAGE, "r")) == NULL){
        cout << "Error. File could not be read" << endl;
        return EXIT_FAILURE;
    }
    for(int i = 0; i < data_count; i++){
        fscanf(file, "%f,", &data[i]);
    }
    fclose(file);
    return EXIT_SUCCESS;
}

void initializeCenters(float *data, float* centers) {
    int rngs[K], center, pixel;
    FILE *file;
    if((file = fopen(RNGS, "r")) == NULL){
        for(int k = 0; k < K; k++){
            rngs[k] = k*5000000;
        }
    }
    else{
        for(int k = 0; k < K; k++){
            fscanf(file, "%d,", &rngs[k]);
        }
        free(file);
    }

    for (int i = 0; i < K; i++) {
        center = BANDS * i;
        pixel = BANDS * rngs[i];
        for(int j = 0; j < BANDS; j++){
            centers[center + j] = data[pixel + j];
        }
    }
}

void assignPointsToClusters(float *data, float *centers, int *labels){
    float minDist, minLabel, dist;
    int omp_num_threads = omp_get_max_threads();
    size_t chunk_size = data_size / omp_num_threads;

    #pragma omp parallel private(minDist, minLabel, dist)
    {
        //varios hilos hacen el for con 3 variables privadas repartiendo bloques de los datos
        #pragma omp for schedule (static, chunk_size)
        for(int point_to_cluster = 0; point_to_cluster < n_pixels; point_to_cluster++){ //es hasta que sea menor de n_pixels
            minDist = FLOAT_MAX;
            minLabel = -1;
            for(int center = 0; center < K; center++){
                dist = euclideanDistance(data, centers, point_to_cluster, center);
                if(dist < minDist){
                    minDist = dist;
                    minLabel = center;
                }
            }
            labels[point_to_cluster] = minLabel;
        }
    }
}

void updateCenters(int *labels, float *new_centers, float *data){
    int clusterCounts[K], pixel_offset, center_offset, label;
    for(int ks = 0; ks < K; ks++){
        clusterCounts[ks] = 0;
    }

    float new_centers_thread[K * BANDS];
    size_t chunk_size = n_pixels / omp_get_max_threads();
    size_t centers_size = K * BANDS;

    #pragma omp parallel private(label, new_centers_thread, pixel_offset, center_offset)
    {   
        int clusterCounts_thread[K];
        for(int i = 0; i < K; i++)
            clusterCounts_thread[i] = 0;

        #pragma omp for schedule(static, chunk_size) 
        for(int px = 0; px < n_pixels; px++){ 
            pixel_offset = px * BANDS;
            label = labels[px];
            clusterCounts_thread[label]++;
            for(int bs = 0; bs < BANDS; bs++){
                new_centers[(label * BANDS) + bs] += data[pixel_offset + bs];
                
            }
        }

        #pragma omp critical
        {
            for(int i = 0; i < K; i++){     //como K es muy pequeña la sobrecarga de paralelizar es mayor a hacerlo atómico porque solo es una suma
                clusterCounts[i] += clusterCounts_thread[i];
            }
        }      

        #pragma omp barrier

        #pragma omp for //static y chunk_size = 1, solo son 5 iteraciones así que tampoco hace falta mucha distribución
        for(int ks = 0; ks < K; ks++){
            center_offset = ks * BANDS;
            if(clusterCounts[ks] > 1){
                for(int bs = 0; bs < BANDS; bs++){
                    new_centers[center_offset + bs] = new_centers[center_offset + bs] / (float)clusterCounts[ks];
                }
            }
        }
    }
}

float calculate_max_distance(float *centers, float *new_centers){
    float dist, maxMovement = FLOAT_MIN;
    #pragma omp parallel for private(dist)
    for(int ks = 0; ks < K; ks++){
        dist = euclideanDistance(new_centers, centers, ks, ks);
        #pragma omp critical 
        {
            if(dist > maxMovement)
                maxMovement = dist;
        }    
    }
    return maxMovement;
}

void create_jpg(int *labels){
    Mat result_image(715,1096, CV_8UC3);

    //  Map color (B, G, R) OpenCV
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

    for(int i = 0; i < n_pixels; i++){
        result_image.at<Vec3b>((int)(i/ROWS), i%ROWS) = colors[labels[i]];
    }

    //Show
    Mat resultT;
    transpose(result_image,resultT); 
    imshow("Clustered Image", resultT);
    imwrite("paviakmeanscpp.jpg",resultT);
    waitKey(0);
}

void kmeans(float *data, float centers[K * BANDS], int labels[ROWS * COLS]) {
    float new_centers[K * BANDS], convergence;
    float prev_max_dist = FLOAT_MAX, dist;

    bool converged = false;
    int iter = 0;
    while(!converged && iter < MAX_ITER){
        assignPointsToClusters(data, centers, labels);

        for (int i = 0; i < K * BANDS; i++)
            new_centers[i] = 0;

        updateCenters(labels, new_centers, data);

        dist = calculate_max_distance(centers, new_centers);
        convergence = 1 - (dist/prev_max_dist);
        if(convergence <= CONVERGENCE)
            converged = true;
        else{
            memcpy(centers, new_centers, centers_size * sizeof(float));
            prev_max_dist = dist;
        }
        
        iter++;
    }
}

int main(){
    cout << "Starting..." << endl;

    float *data = (float*)malloc((ROWS * COLS * BANDS) * sizeof(float));
    if (data == NULL){
        cout << "Error allocating dinamic memory. Aborting..." << endl;
        return EXIT_FAILURE;
    }

    int labels[ROWS * COLS];
    float centers[K * BANDS];
    
    cout << "Number of threads of omp: " << omp_get_max_threads() << endl;
    
    if((readDataFromFile(data)) == EXIT_FAILURE){
        free(data);
        return EXIT_FAILURE;
    }
    else
        cout << "File read." << endl;

    initializeCenters(data, centers);
    cout << "Centers initialized." << endl;

    auto start = std::chrono::high_resolution_clock::now();
    kmeans(data, centers, labels);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    // Imprimir la duración en segundos
    std::cout << "El tiempo de ejecución fue de " << duration.count() << " segundos." << std::endl;

    free(data);
    create_jpg(labels);

    return EXIT_SUCCESS;
}