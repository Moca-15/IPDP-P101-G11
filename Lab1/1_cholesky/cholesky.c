#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#include "cholesky.h"

//#define DEBUG 0


void cholesky_openmp(int n) {

    int i, j, k;
    double **A, **L, **U, **B;

    double start, end;
    int cnt = 0;

    /**
     * 1. Matrix initialization for A, L, U and B
     */
    start = omp_get_wtime();

    A = (double **)malloc(n * sizeof(double *));
    L = (double **)malloc(n * sizeof(double *));
    U = (double **)malloc(n * sizeof(double *));
    B = (double **)malloc(n * sizeof(double *));

    #pragma omp parallel for
    for (i = 0; i < n; i++) {
        A[i] = (double *)malloc(n * sizeof(double));
        L[i] = (double *)malloc(n * sizeof(double));
        U[i] = (double *)malloc(n * sizeof(double));
        B[i] = (double *)malloc(n * sizeof(double));
    }


    // generate random values for the matrix in range [-1,1]
    srand(time(NULL));
    #pragma omp parallel for collapse(2)
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            A[i][j] = ((double) rand() / RAND_MAX) * 2.0 - 1.0;

    // make it positive define
    for (i = 0; i < n; i++)
        for (j = i; j < n; j++) {
            if (i == j) {
                A[i][j] += n;
            } else{
                A[i][j] += ((double) rand() / RAND_MAX) * sqrt(n);
                A[j][i] = A[i][j]; // symmetry :3
            }
        }

    // initialize L, U, and B to 0
    #pragma omp parallel for collapse(2)
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++) {
            L[i][j] = 0.0;
            U[i][j] = 0.0;
            B[i][j] = 0.0;
        }

    end = omp_get_wtime();
    printf("Initialization: %f\n", end-start);


#ifdef DEBUG
    printf("cholesky checkpoint 1\n");
#endif


    /**
     * 2. Compute Cholesky factorization for U
     */

    double sum;
    start = omp_get_wtime();

    for (i = 0; i < n; i++) {
	//diagonal
        sum = 0.0;
        for (k = 0; k < i; k++)
            sum += U[k][i] * U[k][i];

        U[i][i] = sqrt(A[i][i] - sum);

	// resta
        #pragma omp parallel for private(j, k, sum)
        for (j = i + 1; j < n; j++) {
            sum = 0.0;
            for (k = 0; k < i; k++)
                sum += U[k][j] * U[k][i];
            U[i][j] = (A[j][i] - sum) / U[i][i];
        }
    }

    end = omp_get_wtime();
    printf("Cholesky:  %.6f\n", end-start);

#ifdef DEBUG
    printf("cholesky checkpoint 2\n");
#endif


    /**
     * 3. Calculate L from U'
     */

    start = omp_get_wtime();
    // transposada
    #pragma omp parallel for
    for (i = 0; i < n; i++)
        for (j = i; j < n; j++)
            L[j][i] = U[i][j];
    end = omp_get_wtime();
    printf("L=U': %.6f\n", end-start);

#ifdef DEBUG
    printf("cholesky checkpoint 3\n");
#endif


    /**
     * 4. Compute B=LU
     */

    start = omp_get_wtime();
    #pragma omp parallel for collapse(3) private(i, j, k)
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            for (k = 0; k < n; k++)
                B[i][j] += L[i][k] * U[k][j];
    end = omp_get_wtime();
    printf("B=LU: %.6f\n", end-start);

#ifdef DEBUG
    printf("cholesky checkpoint 4\n");
#endif


    /**
     * 5. Check if all elements of A and B have a difference smaller than 0.001%
     */

    cnt=0;
    #pragma omp parallel for reduction(+:cnt)
    for(i=0; i<n; i++){
        for(j=i; j<n; j++){
            double diff = A[i][j] - B[i][j];
            if(sqrt(diff*diff) > 0.001) cnt++;
        }
    }

    if(cnt != 0) {
        printf("Matrices are not equal\n");
    } else {
        printf("Matrices are equal\n");
    }
    printf("A==B?: %d\n", cnt);



    // cleanup :P
    #pragma omp parallel for
    for (i = 0; i < n; i++) {
        free(A[i]);
        free(L[i]);
        free(U[i]);
        free(B[i]);
    }

    free(A);
    free(L);
    free(U);
    free(B);


#ifdef DEBUG
    printf("cholesky checkpoint 5: hell yea execution succesful fking finally\n");
#endif

}








void cholesky(int n) {
    int i, j, k;
    double** A;
    double** L;
    double** U;
    double** B;
    double tmp;
    double start, end;
    int cnt;

#ifdef DEBUG
    printf("cholesky prev\n");
#endif

    /**
     * 1. Matrix initialization for A, L, U and B
     */
    start = omp_get_wtime();
    A = (double **)malloc(n * sizeof(double *)); 
    L = (double **)malloc(n * sizeof(double *)); 
    U = (double **)malloc(n * sizeof(double *)); 
    B = (double **)malloc(n * sizeof(double *)); 
    
    for(i=0; i<n; i++) {
         A[i] = (double *)malloc(n * sizeof(double)); 
         L[i] = (double *)malloc(n * sizeof(double)); 
         U[i] = (double *)malloc(n * sizeof(double)); 
         B[i] = (double *)malloc(n * sizeof(double)); 
    }
    
    srand(time(NULL));
    // Generate random values for the matrix
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            A[i][j] = ((double) rand() / RAND_MAX) * 2.0 - 1.0;  // Generate values between -1 and 1
        }
    }

    // Make the matrix positive definite
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            if (i == j) {
                A[i][j] += n;
            } else {
                A[i][j] += ((double) rand() / RAND_MAX) * sqrt(n);
                A[j][i] = A[i][j];
            }
        }
    }

    for(i=0; i < n; i++) {
        for(j=0; j < n; j++) {
            L[i][j] = 0.0;
            U[i][j] = 0.0;
        }
    }
    end = omp_get_wtime();
    printf("Initialization: %f\n", end-start);


#ifdef DEBUG
    printf("cholesky checkpoint 1\n");
#endif


    /**
     * 2. Compute Cholesky factorization for U
     */
    start = omp_get_wtime();
    for(i=0; i<n; i++) {
        // Calculate diagonal elements
        tmp = 0.0;
        for(k=0;k<i;k++) {
            tmp += U[k][i]*U[k][i];
        }
        U[i][i] = sqrt(A[i][i]-tmp);
        // Calculate non-diagonal elements
        for(j=i+1;j<n;j++) {
            tmp = 0.0;
            for(k=0;k<i;k++) {
                tmp += U[k][j]*U[k][i];
            }
            U[i][j] = ( A[j][i] - tmp ) / U[i][i];
        }
    }
    end = omp_get_wtime();
    printf("Cholesky: %f\n", end-start);


#ifdef DEBUG
    printf("cholesky checkpoint 2\n");
#endif


    /**
     * 3. Calculate L from U'
     */
    start = omp_get_wtime();
    for(i=0; i<n; i++){
        for(j=i; j<n; j++){
            L[j][i] = U[i][j];
        }
    }
    end = omp_get_wtime();
    printf("L=U': %f\n", end-start);


#ifdef DEBUG
    printf("cholesky checkpoint 3\n");
#endif

    /**
     * 4. Compute B=LU
     */
    start = omp_get_wtime();
    for(i=0; i<n; i++){
        for(j=0; j<n; j++){
            for(k=0; k<n; k++) B[i][j] += L[i][k] * U[k][j];
        }
    }
    end = omp_get_wtime();
    printf("B=LU: %f\n", end-start);


#ifdef DEBUG
    printf("cholesky checkpoint 4\n");
#endif


    /**
     * 5. Check if all elements of A and B have a difference smaller than 0.001%
     */
    cnt=0;
    for(i=0; i<n; i++){
        for(j=i; j<n; j++){
            double diff = A[i][j] - B[i][j];
            if(sqrt(diff*diff) > 0.001) cnt++;
        }
    }
    if(cnt != 0) {
        printf("Matrices are not equal\n");
    } else {
        printf("Matrices are equal\n");
    }
    printf("A==B?: %d\n", cnt);

    for(i=0; i<n; i++) {
        free(A[i]);
        free(L[i]);
        free(U[i]);
        free(B[i]);
   }
   free(A);
   free(L);
   free(U);
   free(B);


#ifdef DEBUG
    printf("cholesky checkpoint 5: ended successfully\n");
#endif


}


