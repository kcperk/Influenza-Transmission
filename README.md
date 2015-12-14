# Influenza Transmission CUDA Simulation 
## By: Kevin & Tarun

###Project Description
The goal of this projection is to create an Influenza transimssion simulation using CUDA. In the case of a disease outbreak, it is necessary to simulate contagion transmission. These simulations can aid health officials and policy makers in taking the best possible steps towards reducing and controlling the pandemic. We explored utilizing GPUs to create these simulations and examine performance benefits over their CPU counterparts

###Computer Requirements
* Linux (using CUDA toolkit 4.1), EWS preferred
* GEM

###How to Install
* Clone this github repo
* Follow Compilation and Execution instructions to Compile and Execute code

###Serial Compilation:

* Execute "gcc main.c" (ignore compilations warnings they are benign)
* Execute "./a.out"

###GEM Parallel Compilation:

* Execute "make"
* Execute "qsub run.pbs"
* Wait for result
* View output text file

###Testing

* Modify "globals.h" to change test enviroment

