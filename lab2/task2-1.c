#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N 10000
#define EPSILON 1E-7
#define TAU 1E-5
#define MAX_ITERATION_COUNT 100000

void generate_A(double* A, int size) {
#pragma omp parallel for
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; ++j) {
      A[i * size + j] = 1;
    }

    A[i * size + i] = 2;
  }
}

void generate_x(double* x, int size) {
#pragma omp parallel for
  for (int i = 0; i < size; i++) {
    x[i] = 0;
  }
}

void generate_b(double* b, int size) {
#pragma omp parallel for
  for (int i = 0; i < size; i++) {
    b[i] = N + 1;
  }
}

double calc_norm_square(const double* vector, int size) {
  double norm_square = 0.0;
#pragma omp parallel for reduction(+ : norm_square)
  for (int i = 0; i < size; ++i) {
    norm_square += vector[i] * vector[i];
  }

  return norm_square;
}

void calc_Axb(const double* A, const double* x, const double* b, double* Axb, int size) {
#pragma omp parallel for
  for (int i = 0; i < size; ++i) {
    Axb[i] = -b[i];
    for (int j = 0; j < N; ++j) {
      Axb[i] += A[i * N + j] * x[j];
    }
  }
}

void calc_next_x(const double* Axb, double* x, double tau, int size) {
  for (int i = 0; i < size; ++i) {
    x[i] -= tau * Axb[i];
  }
}

int check_print_error(double* a, const char* message) {
  if (!a) {
    fprintf(stderr, "%s", message);
    return 1;
  }
  return 0;
}

double* allocate_matrix(size_t n) {
  double* matrix = malloc(n * n * sizeof(double));
  check_print_error(matrix, "Failed to allocate memory to matrix\n");
  return matrix;
}

double* allocate_vector(size_t n) {
  double* vector = malloc(n * sizeof(double));
  check_print_error(vector, "Failed to allocate memory to vector\n");
  return vector;
}

void print_vector(const double* vector, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    printf("%lf ", vector[i]);
  }
  printf("\n");
}

int main() {
  double* x;
  double* A;
  double* Axb;
  double* b;
  size_t iters_count;
  double start_time = 0.0, finish_time = 0.0;
  double b_norm = 0.0;
  double accuracy = EPSILON + 1;

  x = allocate_vector(N);
  b = allocate_vector(N);
  A = allocate_matrix(N);
  generate_x(x, N);
  generate_b(b, N);
  generate_A(A, N);

  b_norm = sqrt(calc_norm_square(b, N));
  Axb = allocate_vector(N);

  start_time = omp_get_wtime();

  for (iters_count = 0; iters_count < MAX_ITERATION_COUNT && accuracy > EPSILON; ++iters_count) {
    calc_Axb(A, x, b, Axb, N);
    calc_next_x(Axb, x, TAU, N);
    accuracy = sqrt(calc_norm_square(Axb, N)) / b_norm;
  }

  finish_time = omp_get_wtime();

  if (iters_count == MAX_ITERATION_COUNT) {
    printf("Too many iterations\n");
  } else {
    printf("Time: %lf sec\n", finish_time - start_time);
  }

  free(x);
  free(b);
  free(A);
  free(Axb);

  return 0;
}
