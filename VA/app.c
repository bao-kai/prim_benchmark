/**
 * app.c
 * VA Host Application Source File with Chunking
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <dpu.h>
#include <dpu_log.h>
#include <unistd.h>
#include <getopt.h>
#include <assert.h>

#include "../support/common.h"
#include "../support/timer.h"
#include "../support/params.h"

// Define the DPU Binary path as DPU_BINARY here
#ifndef DPU_BINARY
#define DPU_BINARY "./bin/dpu_code"
#endif

#if ENERGY
#include <dpu_probe.h>
#endif

// Define maximum MRAM size per DPU (adjust as needed)
  // 60MB

// Pointer declaration
static T* A;
static T* B;
static T* C;
static T* C2;

#define MAX_MRAM_SIZE (60 * 1024 * 1024)/sizeof(T)
// Create input arrays
static void read_input(T* A, T* B, unsigned int nr_elements) {
    srand(0);
    printf("nr_elements\t%u\t", nr_elements);
    for (unsigned long long i = 0; i < nr_elements; i++) {
        A[i] = (T) (rand());
        B[i] = (T) (rand());
    }
}

// Compute output in the host
static void vector_addition_host(T* C, T* A, T* B, unsigned int nr_elements) {
    for (unsigned long long i = 0; i < nr_elements; i++) {
        C[i] = A[i] + B[i];
    }
}

int main(int argc, char **argv) {

    struct Params p = input_params(argc, argv);

    struct dpu_set_t dpu_set, dpu;
    uint32_t nr_of_dpus;

#if ENERGY
    struct dpu_probe_t probe;
    DPU_ASSERT(dpu_probe_init("energy_probe", &probe));
#endif

    // Allocate DPUs and load binary
    DPU_ASSERT(dpu_alloc(NR_DPUS, NULL, &dpu_set));
    DPU_ASSERT(dpu_load(dpu_set, DPU_BINARY, NULL));
    DPU_ASSERT(dpu_get_nr_dpus(dpu_set, &nr_of_dpus));
    printf("Allocated %d DPU(s)\n", nr_of_dpus);
    unsigned int i = 0;
    
    const unsigned long long input_size = 2000000000;//p.exp == 0 ? p.input_size * nr_of_dpus : p.input_size;
    const unsigned long long input_size_8bytes = ((input_size * sizeof(T)) % 8) != 0 ? roundup(input_size, 8) : input_size;
    const unsigned long long input_size_dpu = divceil(input_size, nr_of_dpus);
    const unsigned long long input_size_dpu_8bytes = ((input_size_dpu * sizeof(T)) % 8) != 0 ? roundup(input_size_dpu, 8) : input_size_dpu;

    A = malloc(input_size_8bytes * sizeof(T));
    B = malloc(input_size_8bytes * sizeof(T));
    C = malloc(input_size_8bytes * sizeof(T));
    C2 = malloc(input_size_8bytes * sizeof(T));
    T *bufferA = A;
    T *bufferB = B;
    T *bufferC = C2;
    
    read_input(A, B, input_size);

    Timer timer;
    printf("NR_TASKLETS\t%d\tBL\t%d\n %d\n", NR_TASKLETS, BL, p.n_reps);
    
    const unsigned long long max_chunk_size = nr_of_dpus * MAX_MRAM_SIZE / sizeof(T);
    unsigned long long offset = 0;


    for(int rep = 0; rep < p.n_warmup + p.n_reps; rep++) {
        if(rep >= p.n_warmup)
            start(&timer, 0, rep - p.n_warmup);
        vector_addition_host(C, A, B, input_size);
        if(rep >= p.n_warmup)
            stop(&timer, 0);
        for(int x=0;x<64;x++){
            offset = 0;
            int t=0;
            while (offset < input_size) {
                
                unsigned long long chunk_size = (offset + max_chunk_size > input_size) ? (input_size - offset) : max_chunk_size;
                unsigned int chunk_size_dpu = divceil(chunk_size, nr_of_dpus);
                unsigned int chunk_size_dpu_8bytes = ((chunk_size_dpu * sizeof(T)) % 8) != 0 ? roundup(chunk_size_dpu, 8) : chunk_size_dpu;

                //printf("Processing chunk starting at offset %llu\n", offset);

                if(rep >= p.n_warmup)
                    start(&timer, 1, rep - p.n_warmup);

                unsigned int kernel = 0;
                dpu_arguments_t input_arguments[NR_DPUS];
                for(i = 0; i < nr_of_dpus; i++){
                    unsigned int this_dpu_chunk_size = (i == nr_of_dpus - 1) ? (chunk_size - chunk_size_dpu * i) : chunk_size_dpu;
                    input_arguments[i].size = this_dpu_chunk_size * sizeof(T);
                    input_arguments[i].transfer_size = chunk_size_dpu_8bytes * sizeof(T);
                    input_arguments[i].kernel = kernel;
                }


                i = 0;
                DPU_FOREACH(dpu_set, dpu, i) {
                    DPU_ASSERT(dpu_prepare_xfer(dpu, &input_arguments[i]));
                }
                DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "DPU_INPUT_ARGUMENTS", 0, sizeof(input_arguments[0]), DPU_XFER_DEFAULT));


                DPU_FOREACH(dpu_set, dpu, i) {
                    DPU_ASSERT(dpu_prepare_xfer(dpu, bufferA + offset + chunk_size_dpu_8bytes * i));
                }
                DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0, chunk_size_dpu_8bytes * sizeof(T), DPU_XFER_DEFAULT));


                DPU_FOREACH(dpu_set, dpu, i) {
                    DPU_ASSERT(dpu_prepare_xfer(dpu, bufferB + offset + chunk_size_dpu_8bytes * i));
                }
                DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, chunk_size_dpu_8bytes * sizeof(T), chunk_size_dpu_8bytes * sizeof(T), DPU_XFER_DEFAULT));

                if(rep >= p.n_warmup)
                    stop(&timer, 1);


                // Run program on DPU(s) 
                if(rep >= p.n_warmup) {
                    start(&timer, 2, rep - p.n_warmup);
    #if ENERGY
                    DPU_ASSERT(dpu_probe_start(&probe));
    #endif
                }
                DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));


                if(rep >= p.n_warmup) {
                    stop(&timer, 2);
    #if ENERGY
                    DPU_ASSERT(dpu_probe_stop(&probe));
    #endif
                }

    #if PRINT
                {
                    unsigned int each_dpu = 0;
                    printf("Display DPU Logs\n");
                    DPU_FOREACH (dpu_set, dpu) {
                        printf("DPU#%d:\n", each_dpu);
                        DPU_ASSERT(dpulog_read_for_dpu(dpu.dpu, stdout));
                        each_dpu++;
                    }
                }
    #endif



                printf("Retrieve results\n%d %d", x, t++);
                if(rep >= p.n_warmup)
                    start(&timer, 3, rep - p.n_warmup);

                i = 0;
                DPU_FOREACH(dpu_set, dpu, i) {
                    DPU_ASSERT(dpu_prepare_xfer(dpu, bufferC + offset + chunk_size_dpu_8bytes * i));
                }
                DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME, input_arguments[0].transfer_size, input_arguments[0].transfer_size, DPU_XFER_DEFAULT));


                if(rep >= p.n_warmup)
                    stop(&timer, 3);
                offset += chunk_size;

            }
        }
        
    }


    printf("CPU ");
    print(&timer, 0, p.n_reps);
    printf("CPU-DPU ");
    print(&timer, 1, p.n_reps);
    printf("DPU Kernel ");
    print(&timer, 2, p.n_reps);
    printf("DPU-CPU ");
    print(&timer, 3, p.n_reps);



#if ENERGY
    double energy;
    DPU_ASSERT(dpu_probe_get(&probe, DPU_ENERGY, DPU_AVERAGE, &energy));
    printf("DPU Energy (J): %f\t", energy);
#endif

    bool status = true;
    for (i = 0; i < input_size; i++) {
        if(C[i] != bufferC[i]){ 
            status = false;
#if PRINT
            printf("%d: %u -- %u\n", i, C[i], bufferC[i]);
#endif
        }
    }

    if (status) {
        printf("[" ANSI_COLOR_GREEN "OK" ANSI_COLOR_RESET "] Outputs are equal\n");
    } else {
        printf("[" ANSI_COLOR_RED "ERROR" ANSI_COLOR_RESET "] Outputs differ!\n");
    }


    free(A);
    free(B);
    free(C);
    free(C2);
    DPU_ASSERT(dpu_free(dpu_set));

    return status ? 0 : -1;
} 
