#include <stdlib.h>
#include <stdio.h>
#include "../../support/timer.h"
#include "gemv_utils.h"

int main(int argc, char *argv[])
{
  const size_t rows = 262144;//20480;
  const size_t cols = 65536;//8192;

  float **A, *b, *x;

  b = (float*) malloc(sizeof(float)*rows);
  x = (float*) malloc(sizeof(float)*cols);

  allocate_dense(rows, cols, &A);

  make_hilbert_mat(rows,cols, &A);

#pragma omp parallel
    {
#pragma omp for
    for (size_t i = 0; i < cols; i++) {
      x[i] = (float) i+1 ;
    }

#pragma omp for
    for (size_t i = 0; i < rows; i++) {
      b[i] = (float) 0.0;
    }
    }

  Timer timer;
  start(&timer, 0, 0);


   gemv(A, x, rows, cols, &b);
   
   stop(&timer, 0);


    printf("Kernel ");
    print(&timer, 0, 1);
    printf("\n");

#if 0
  print_vec(x, rows);
  print_mat(A, rows, cols);
  print_vec(b, rows);
#endif

  printf("sum(x) = %f, sum(Ax) = %f\n", sum_vec(x,cols), sum_vec(b,rows));
  return 0;
}

void gemv(float** A, float* x, size_t rows, size_t cols, float** b) {
  omp_set_num_threads(1);
  #pragma omp parallel for
    for (size_t i = 0; i < rows; i ++ )
      for (size_t j = 0; j < cols; j ++ ) {
        
        (*b)[i] = (*b)[i] + A[i][j]*x[j];
      }
}

void make_hilbert_mat(size_t rows, size_t cols, float*** A) {
#pragma omp parallel for
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      if (i >= rows || j >= cols) {
        printf("Error: Index out of bounds! i = %zu, j = %zu\n", i, j);
        exit(1);
      }
      (*A)[i][j] = 1.0/( (float) i + (float) j + 1.0);
    }
  }
}

float sum_vec(float* vec, size_t rows) {
  float sum = 0.0;
#pragma omp parallel for reduction(+:sum)
  for (int i = 0; i < rows; i++) sum = sum + vec[i];
  return sum;
}
