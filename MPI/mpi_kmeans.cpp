#include "mpi.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <vector>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

#include <chrono>

#define ROWS 1096
#define COLS 715
#define BANDS 102

#define IMAGE "pavia.txt"
#define RNGS "rng.txt"

#define CONVERGENCE 0.2
#define MAX_ITER 50
#define K 5

#define TRUE 1
#define FALSE 0

#define MASTER 0

#define FLOAT_MAX 3.4028234663852886e+38F
#define FLOAT_MIN 1.1754943508222875e-38F

int data_per_node;
int labels_per_node;
int pixels_per_node;
int centers_size;
int n_pixels;

//dependiendo si se llama al asignar puntos a centroides o calculando la convergencia
//el parámetro pixel puede significar el pixel de la imagen o, en el caso de la convergencia,
//el centroide con el que se compara, al final es el offset desde donde empieza
float euclideanDistance(float *data, float *centers, int pixel, int center){
    float distance = 0;
    int data_offset = pixel * BANDS;
    int center_offset = center * BANDS;
    for(int i = 0; i < BANDS; i++){
        distance += pow(data[data_offset + i] - centers[center_offset + i], 2);
    }
    return sqrt(distance);
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
    }

    for (int i = 0; i < K; i++) {
        center = BANDS * i;
        pixel = BANDS * rngs[i];
        for(int j = 0; j < BANDS; j++){
            centers[center + j] = data[pixel + j];
        }
    }
}

int readDataFromFile(float *data, int data_count){
    FILE *file;
    if((file = fopen(IMAGE, "r")) == NULL){
        printf("Error. No se puede leer el archivo");
        return EXIT_FAILURE;
    }
    for(int i = 0; i < data_count; i++){
        fscanf(file, "%f,", &data[i]);
    }
    free(file);
    return EXIT_SUCCESS;
}

int main(int argc, char *argv[]) {
    int size, rank, data_count = ROWS * COLS * BANDS;
    short converged = FALSE;
    float centers[K * BANDS];

    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    float *data = (float*)malloc((ROWS * COLS * BANDS) * sizeof(float));
    int *labels = (int*)malloc((ROWS * COLS) * sizeof(int));
    float *new_centers = (float*)malloc((K * BANDS) * sizeof(float));
    float *last_new_centers = (float*)malloc(BANDS * sizeof(float));
    float *all_new_centers = (float*)malloc((K * BANDS * (size - 1)) * sizeof(float));
    
    int *labels_part = (int*)malloc(((ROWS * COLS)/size) * sizeof(int));
    float *data_part = (float*)malloc(((ROWS * COLS * BANDS)/size) * sizeof(float));
    
    //variables de tamaños
    n_pixels = ROWS * COLS;
    data_per_node = (n_pixels * BANDS)/size;
    labels_per_node = n_pixels/size;
    pixels_per_node = n_pixels/size;
    centers_size = K * BANDS;
    int last_offset = n_pixels % size;
    int last_node_pixels = pixels_per_node + last_offset;
    int last_node_start = pixels_per_node * (size - 1);

    if(last_offset > 0 && rank == (size - 1))
        pixels_per_node += last_offset;    

    //leer imagen    
    if(rank == MASTER){
        if((readDataFromFile(data, data_count)) == EXIT_FAILURE){
            free(data);
            free(labels);
            return EXIT_FAILURE;
        }
        initializeCenters(data, centers);
    }
    else {
        free(data);
        free(labels);
        free(all_new_centers);
        free(last_new_centers);
    }

    auto start = std::chrono::high_resolution_clock::now();

    //repartir la imagen
    MPI_Scatter(data, data_per_node, MPI_FLOAT, data_part, data_per_node, MPI_FLOAT, MASTER, MPI_COMM_WORLD);

    //repartir las etiquetas
    MPI_Scatter(labels, labels_per_node, MPI_INT, labels_part, labels_per_node, MPI_INT, MASTER, MPI_COMM_WORLD);

    //empieza kmeans
    int iter = 0, minLabel;
    float dist, minDist, maxMovement, prev_maxMovement = FLOAT_MAX, convergence;
    float ponder = (float)pixels_per_node/n_pixels;
    while(!converged && iter < MAX_ITER){

        //se copian los centroides en todos los procesos
        MPI_Bcast(centers, centers_size, MPI_FLOAT, MASTER, MPI_COMM_WORLD);

        //assignPointsToClusters
        for(int ptc = 0; ptc < pixels_per_node; ptc++){
            minDist = FLOAT_MAX;
            minLabel = -1;
            for(int cs = 0; cs < K; cs++){
                dist = euclideanDistance(data_part, centers, ptc, cs);
                if(dist < minDist){
                    minDist = dist;
                    minLabel = cs;
                }
            }
            labels_part[ptc] = minLabel;
        }

        //updateCenters
        int pixel_offset, clusterCounts[K], label;
        for(int ks = 0; ks < K; ks++){
            clusterCounts[ks] = 0;
        }
        for(int px = 0; px < labels_per_node; px++){
            pixel_offset = px * BANDS;
            label = labels_part[px];
            clusterCounts[label]++;
            for(int bs = 0; bs < BANDS; bs++){
                new_centers[(label * BANDS) + bs] += data_part[pixel_offset + bs];
            }
        }

        int center_offset;
        for(int ks = 0; ks < K; ks++){
            center_offset = ks * BANDS;
            if(clusterCounts[ks] > 1){
                for(int bs = 0; bs < BANDS; bs++){
                    new_centers[center_offset + bs] = new_centers[center_offset + bs] / (float)clusterCounts[ks];
                    new_centers[center_offset + bs] = new_centers[center_offset + bs] * ponder;
                }
            }
        }

        //se recogen las medias ponderadas de la actualización de centroides teniendo en cuenta si todos los procesos tienen los mismos píxeles o el último tiene más
        MPI_Gather(new_centers, centers_size, MPI_FLOAT, all_new_centers, centers_size, MPI_FLOAT, MASTER, MPI_COMM_WORLD);

        //juntamos todos los nuevos centroides ponderados en el master
        if(rank == MASTER){
            for(int ac = 0; ac < (K * BANDS); ac++){
                new_centers[ac] = 0;
            }
            for(int nodes = 0; nodes < size; nodes++){
                for(int cs = 0; cs < K; cs++){
                    center_offset = cs * BANDS;
                    for(int bs = 0; bs < BANDS; bs++){
                        new_centers[center_offset + bs] += all_new_centers[(nodes * centers_size) + center_offset + bs];
                    }
                }
            }

            //miramos si converge
            maxMovement = FLOAT_MIN;
            for(int ks = 0; ks < K; ks++){
                dist = euclideanDistance(new_centers, centers, ks, ks);
                if(dist > maxMovement)
                    maxMovement = dist;
            }
            convergence = 1 - (maxMovement/prev_maxMovement);
            if(convergence < CONVERGENCE)
                converged = TRUE;

            memcpy(centers, new_centers, centers_size * sizeof(float));
            prev_maxMovement = maxMovement;
        }

        MPI_Bcast(&converged, 1, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
        
        iter++;
    }

    MPI_Gather(labels_part, labels_per_node, MPI_INT, labels, labels_per_node, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Finalize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    // Imprimir la duración en segundos
    std::cout << "El tiempo de ejecución fue de " << duration.count() << " segundos." << std::endl;

    if(rank == MASTER){
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

        for(int i = 0; i < n_pixels; i++){
            result_image.at<Vec3b>((int)(i/ROWS), i%ROWS) = colors[labels[i]];
        }

        //Mostrar la imagen
        Mat resultT;
        transpose(result_image,resultT); 
        imshow("Clustered Image", resultT);
        imwrite("paviakmeanscpp.jpg",resultT);
        waitKey(0);
    }
    return EXIT_SUCCESS;
}