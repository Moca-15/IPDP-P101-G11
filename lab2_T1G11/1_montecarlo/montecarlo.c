#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[]){
int numtasks, rank;
MPI_Init (&argc, &argv);

MPI_Comm_size (MPI_COMM_WORLD, &numtasks);
MPI_Comm_rank (MPI_COMM_WORLD, &rank);

printf("Hello, world, from rank %d/%d\n",rank,numtasks);

MPI_Finalize();

return 0;

}
