#include <math.h>
#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#define DIMS_COUNT 2
#define X 0
#define Y 1

int check_print_error(double *a, const char *message) {
  if (!a) {
    fprintf(stderr, "%s", message);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

double *allocate_matrix(size_t size) {
  double *matrix = malloc(size * sizeof(double));
  check_print_error(matrix, "FAILED TO ALLOCATE MEMORY TO MATRIX\n");
  return matrix;
}

void generate_matrix(double *matrix, int column, int leading_row, int leading_column, bool onRows) {
  for (int i = 0; i < leading_row; ++i) {
    for (int j = 0; j < leading_column; ++j) {
      matrix[i * column + j] = onRows ? i : j;
    }
  }
}


/// @brief Split matrix B into blocks in B_blocks
/// @param B Matrix to split
/// @param B_block where splited matrix will be
/// @param B_block_size size of splited matrix
/// @param n_2 size of B matrix
/// @param aligned_n3 aligned size
void split_B(const double *B, double *B_block, int B_block_size, int n_2, int aligned_n3, int coords_x, MPI_Comm comm_rows, MPI_Comm comm_columns) {
  if (coords_x == 0) {
    MPI_Datatype column_not_resized_t;
    MPI_Datatype column_resized_t;

    MPI_Type_vector(n_2, B_block_size, aligned_n3, MPI_DOUBLE, &column_not_resized_t);
    MPI_Type_commit(&column_not_resized_t);

    MPI_Type_create_resized(column_not_resized_t, 0, B_block_size * sizeof(double), &column_resized_t);
    MPI_Type_commit(&column_resized_t);

    MPI_Scatter(B, 1, column_resized_t, B_block, B_block_size * n_2, MPI_DOUBLE, 0, comm_rows);

    MPI_Type_free(&column_not_resized_t);
    MPI_Type_free(&column_resized_t);
  }

  MPI_Bcast(B_block, B_block_size * n_2, MPI_DOUBLE, 0, comm_columns);
}


void split_A(const double *A, double *A_block, int A_block_size, int n_2, int coords_y, MPI_Comm comm_rows, MPI_Comm comm_columns) {
  if (coords_y == 0) {
    MPI_Scatter(A, A_block_size * n_2, MPI_DOUBLE, A_block, A_block_size * n_2, MPI_DOUBLE, 0, comm_columns);
  }

  MPI_Bcast(A_block, A_block_size * n_2, MPI_DOUBLE, 0, comm_rows);
}

void init_communicators(const int dims[DIMS_COUNT], MPI_Comm *comm_grid, MPI_Comm *comm_rows, MPI_Comm *comm_columns) {
  int reorder = 1;
  int periods[DIMS_COUNT] = {};
  int sub_dims[DIMS_COUNT] = {};

  MPI_Cart_create(MPI_COMM_WORLD, DIMS_COUNT, dims, periods, reorder, comm_grid);

  sub_dims[X] = false;
  sub_dims[Y] = true;
  MPI_Cart_sub(*comm_grid, sub_dims, comm_rows);

  sub_dims[X] = true;
  sub_dims[Y] = false;
  MPI_Cart_sub(*comm_grid, sub_dims, comm_columns);
}

void init_dims(int dims[DIMS_COUNT], int proc_count, int argc, char **argv) {
  if (argc < 3)
    MPI_Dims_create(proc_count, DIMS_COUNT, dims);
  else {
    dims[X] = atoi(argv[1]);
    dims[Y] = atoi(argv[2]);
  }
}

void multiply(const double *A_block, const double *B_block, double *C_block, int A_block_size, int B_block_size, int n_2) {
  for (int i = 0; i < A_block_size; ++i) {
    for (int j = 0; j < B_block_size; ++j) {
      C_block[i * B_block_size + j] = 0;
    }
  }

  for (int i = 0; i < A_block_size; ++i) {
    for (int j = 0; j < n_2; ++j) {
      for (int k = 0; k < B_block_size; ++k) {
        C_block[i * B_block_size + k] += A_block[i * n_2 + j] * B_block[j * B_block_size + k];
      }
    }
  }
}

void gather_C(const double *C_block, double *C, int A_block_size, int B_block_size, int aligned_n1, int aligned_n3, int proc_count, MPI_Comm comm_grid) {
  MPI_Datatype not_resized_recv_t;
  MPI_Datatype resized_recv_t;

  int dims_x = aligned_n1 / A_block_size;
  int dims_y = aligned_n3 / B_block_size;
  int *recv_counts = malloc(sizeof(int) * proc_count);
  int *displs = malloc(sizeof(int) * proc_count);

  MPI_Type_vector(A_block_size, B_block_size, aligned_n3, MPI_DOUBLE, &not_resized_recv_t);
  MPI_Type_commit(&not_resized_recv_t);

  MPI_Type_create_resized(not_resized_recv_t, 0, B_block_size * sizeof(double), &resized_recv_t);
  MPI_Type_commit(&resized_recv_t);

  for (int i = 0; i < dims_x; ++i)
    for (int j = 0; j < dims_y; ++j) {
      recv_counts[i * dims_y + j] = 1;
      displs[i * dims_y + j] = j + i * dims_y * A_block_size;
    }

  MPI_Gatherv(C_block, A_block_size * B_block_size, MPI_DOUBLE, C, recv_counts, displs, resized_recv_t, 0, comm_grid);

  MPI_Type_free(&not_resized_recv_t);
  MPI_Type_free(&resized_recv_t);
  free(recv_counts);
  free(displs);
}

bool check_C(const double *C, int column, int leading_row, int leading_column, int n_2) {
  for (int i = 0; i < leading_row; ++i)
    for (int j = 0; j < leading_column; ++j) {
      if (C[i * column + j] != (double)(i * j * n_2)) {
        return false;
      }
    }

  return true;
}

void print_matrix(const double *matrix, int column, int leading_row, int leading_column) {
  for (int i = 0; i < leading_row; i++) {
    for (int j = 0; j < leading_column; j++) {
      printf("%lf ", matrix[i * column + j]);
    }

    printf("\n");
  }
}

int main(int argc, char **argv) {
  int n_1 = 2000;
  int n_2 = 1500;
  int n_3 = 2500;
  int proc_rank;
  int proc_count;
  int aligned_n1;
  int aligned_n3;
  int A_block_size;
  int B_block_size;
  int dims[DIMS_COUNT] = {};
  int coords[DIMS_COUNT] = {};
  double start_time;
  double finish_time;
  double *A = NULL;
  double *B = NULL;
  double *C = NULL;
  double *A_block = NULL;
  double *B_block = NULL;
  double *C_block = NULL;
  MPI_Comm comm_grid;
  MPI_Comm comm_rows;
  MPI_Comm comm_columns;

  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &proc_count);

  init_dims(dims, proc_count, argc, argv);

  init_communicators(dims, &comm_grid, &comm_rows, &comm_columns);

  MPI_Cart_coords(comm_grid, proc_rank, DIMS_COUNT, coords);

  A_block_size = ceil((double)n_1 / dims[X]);
  B_block_size = ceil((double)n_3 / dims[Y]);
  aligned_n1 = A_block_size * dims[X];
  aligned_n3 = B_block_size * dims[Y];

  if (coords[X] == 0 && coords[Y] == 0) {
    A = allocate_matrix(aligned_n1 * n_2);

    B = allocate_matrix(n_2 * aligned_n3);
    C = allocate_matrix(aligned_n1 * aligned_n3);

    generate_matrix(A, n_2, n_1, n_2, true);
    generate_matrix(B, aligned_n3, n_2, n_3, false);
  }

  start_time = MPI_Wtime();

  A_block = allocate_matrix(A_block_size * n_2);
  B_block = allocate_matrix(B_block_size * n_2);
  C_block = allocate_matrix(A_block_size * B_block_size);

  split_A(A, A_block, A_block_size, n_2, coords[Y], comm_rows, comm_columns);
  split_B(B, B_block, B_block_size, n_2, aligned_n3, coords[X], comm_rows, comm_columns);

  multiply(A_block, B_block, C_block, A_block_size, B_block_size, n_2);

  gather_C(C_block, C, A_block_size, B_block_size, aligned_n1, aligned_n3, proc_count, comm_grid);

  finish_time = MPI_Wtime();

  if (coords[Y] == 0 && coords[X] == 0) {
    printf("Is matrix C correct? - %s\n", check_C(C, aligned_n3, n_1, n_3, n_2) ? "yes" : "no");
    printf("Time: %lf\n", finish_time - start_time);

    free(A);
    free(B);
    free(C);
  }

  free(A_block);
  free(B_block);
  free(C_block);
  MPI_Comm_free(&comm_grid);
  MPI_Comm_free(&comm_rows);
  MPI_Comm_free(&comm_columns);

  MPI_Finalize();

  return EXIT_SUCCESS;
}
