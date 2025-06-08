#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <openacc.h>

// DONE
// Sequential vector addition
void vecadd_oacc(double *A, double *B, double *C, const int N)
{
#pragma acc parallel loop present(A[:N],B[:N],C[:N])
	for(int i=0; i<N; i++){
		C[i] = A[i] + B[i];
	}
}

int main(int argc, char *argv[])
{
    int N;

    if (argc != 2)
    {
        printf("Usage: %s <vector size N>\n", argv[0]);
        return 1;
    }
    else
    {
        N = atoi(argv[1]);
    }
    printf("Vector size: %d\n", N);

    //
    // Memory allocation
    //
    double *A = (double *)malloc(N * sizeof(double));
    double *B = (double *)malloc(N * sizeof(double));
    double *C = (double *)malloc(N * sizeof(double));

    //
    // Initialize vectors
    //
    // DONE
	for(int i=0; i<N; i++){
		A[i] = i;
		B[i] = 2*(N-i);
	}

    //
    // Vector addition
    //
    struct timespec start, end;


	clock_gettime(CLOCK_MONOTONIC, &start);
#pragma acc data copyin(A[:N], B[:N]) copyout(C[:N])
    vecadd_oacc(A, B, C, N);

	clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1.0e9;
    printf("Elapsed time: %.9f seconds\n", elapsed);

    //
    // Validation
    //
    // DONE
    // Validate vector addition
	for(int i=0; i<N; i++){
		double diff = C[i] - (2*N-i);
		if(fabs(diff) > 1E-6){
			printf("Value exeeding tolerance at i=%d (%lf)\n",i,C[i]);
			break;
		}
	}
    //
    // Free memory
    //
    free(A);
    free(B);
    free(C);

    return 0;
}
