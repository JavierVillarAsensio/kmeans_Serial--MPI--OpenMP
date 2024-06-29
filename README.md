## kmeans_Serial--MPI--OpenMP
# Ejecución
Las tres versiones contienen un Makefile, en el caso en serie, como la ejecución es larga al ejecutar *make* solo compila pero no ejecuta, en las versiones MPI y OpenMP compila y ejecuta el código. El mismo archivo Makefile tiene una opción *clean* para eliminar los archivos creados en la compilación y ejecución

# Importante
Para la ejecución se necesita el archivo *pavia.txt* que no se puede subir al repositorio por su peso. \
El reporte y la ejecución en MPI hablan de dos procesos en la ejecución porque es el máximo que podíamos usar en el equipo con el que se ha desarrollado. Para cambiar el número de procesos en MPI es cambiar el valor del parámetro *-n* que hay para ejecutar el código compilado con MPI por el número de procesos que se quiera y en OpenMP, la cantidad de procesos está guardada y se puede cambiar en la variabLe de entorno $OMP_NUM_THREADS.

# Autores
Francisco Jesús Díaz Pellejero \
Javier Villar Asensio
