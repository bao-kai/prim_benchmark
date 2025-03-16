void allocate_dense(size_t rows,size_t  cols, float*** dense) {

  *dense = malloc(sizeof(double)*rows);
  **dense = malloc(sizeof(double)*rows*cols);

  for (size_t i=0; i < rows; i++ ) {
    (*dense)[i] = (*dense)[0] + i*cols;
  }

}

void print_mat(float** A, size_t rows, size_t cols) {
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      printf("%f ", A[i][j]);
    }
    printf("\n");
  }
}

void print_vec(float* b, size_t rows) {
  for (size_t i = 0; i < rows; i++) {
    printf("%f\n", b[i]);
  }
}

void gemv(float** A, float* x, size_t rows, size_t cols, float** b);
void make_hilbert_mat(size_t rows, size_t cols, float*** A);
float sum_vec(float* vec, size_t rows);
