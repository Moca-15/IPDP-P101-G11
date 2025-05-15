#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <time.h>
#include <mpi.h>

#include "auxiliar.h"

/// TODO
/// Reading the planes from a file for MPI
void read_planes_mpi(const char* filename, PlaneList* planes, int* N, int* M, double* x_max, double* y_max, int rank, int size, int* tile_displacements)
{

	int num_planes;
	int int_arr[3];
	double db_arr[2];


	MPI_File fh;
	MPI_Status status;
	MPI_Offset data_start = 0; // quan acaba el header i començen els planes

	// MPI escriurà el file handler a l'adreça de fh
	int err = MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
	if(err != MPI_SUCCESS){
		fprintf(stderr, "error opening file :(\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	// llegirà data del header (les 1eres línies)
	if(rank == 0) {

		int max_line_size = 256;
		char line[max_line_size];

		char c;
		int line_pos;

		for(int i = 0; i<4; i++) {
			line_pos = 0;
			while(1) {
				MPI_File_read_at(fh, data_start, &c, 1, MPI_CHAR, &status);
				data_start++;
				if(c == '\n') break;
				line[line_pos++] = c;
				if(line_pos >= max_line_size-1) break;
			}
			line[line_pos] = '\0'; 
			printf("%d : read line %s\n", rank, line);

			// línies 3 i 5
			if(i == 1) {sscanf(line, "# Map: %lf, %lf : %d %d", x_max, y_max, N, M);}
			if(i == 2) {sscanf(line, "# Number of Planes: %d", &num_planes);}
		}

		// arrays per als broadcasts
		int_arr[0] = *N; int_arr[1] = *M; int_arr[2] = num_planes;
		db_arr[0] = *x_max; db_arr[1] = *y_max;

	}

	MPI_Bcast(int_arr, 3, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(db_arr, 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&data_start, 1, MPI_OFFSET, 0, MPI_COMM_WORLD);

	printf("%d : received: N = %d : M = %d : num_planes = %d \n\t x_max = %f : y_max = %f \n\t data_start = %d\n", rank, int_arr[0], int_arr[1], int_arr[2], db_arr[0], db_arr[1], data_start);

	*N = int_arr[0];
	*M = int_arr[1];
	num_planes = int_arr[2];

	*x_max = db_arr[0];
	*y_max = db_arr[1];

	MPI_Offset file_size;
	MPI_File_get_size(fh, &file_size);

	printf("%d : file size = %d\n", rank, file_size);

	MPI_Offset plane_data_size = file_size - data_start; // mida de les dades dels avions (fitxer sense el header)

	printf("%d : data size = %d\n", rank, plane_data_size);

	int line_size = (num_planes != 0 ? plane_data_size / num_planes : 0); // mida d'una línia (totes seràn iguals). serà enter pq per cada plane hi ha 1 línia
	printf("%d : line size = %d\n", rank, line_size);


	int local_planes = num_planes / size + ((rank >= num_planes%size)? 0 : 1); // els que ha d'executar aquest rank

	printf("%d : my planes = %d\n", rank, local_planes);


	MPI_Offset displacement = data_start + rank * local_planes * line_size;

	printf("%d : displacement = %lld\n", rank, displacement);





	printf("%d : barrier reached\n", rank);

	MPI_Barrier(MPI_COMM_WORLD);

	printf("%d : barrier passed \n", rank);



	//for(int i = 0; i < local_planes; i++) {
		
	

	//}



	MPI_File_close(&fh);

}

/// TODO
/// Communicate planes using mainly Send/Recv calls with default data types
void communicate_planes_send(PlaneList* list, int N, int M, double x_max, double y_max, int rank, int size, int* tile_displacements)
{
}

/// TODO
/// Communicate planes using all to all calls with default data types
void communicate_planes_alltoall(PlaneList* list, int N, int M, double x_max, double y_max, int rank, int size, int* tile_displacements)
{
}

typedef struct {
    int    index_plane;
    double x;
    double y;
    double vx;
    double vy;
} MinPlaneToSend;

/// TODO
/// Communicate planes using all to all calls with custom data types
void communicate_planes_struct_mpi(PlaneList* list,
                               int N, int M,
                               double x_max, double y_max,
                               int rank, int size,
                               int* tile_displacements)
{
}

int main(int argc, char **argv) {
    int debug = 0;                      // 0: no debug, 1: shows all planes information during checking
    int N = 0, M = 0;                   // Grid dimensions
    double x_max = 0.0, y_max = 0.0;    // Total grid size
    int max_steps;                      // Total simulation steps
    char* input_file;                   // Input file name
    int check;                          // 0: no check, 1: check the simulation is correct

    int rank, size;

    /// TODO
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);




    int tile_displacements[size+1];
    int mode = 0;
    if (argc == 5) {
        input_file = argv[1];
        max_steps = atoi(argv[2]);
        if (max_steps <= 0) {
            fprintf(stderr, "max_steps needs to be a positive integer\n");
            return 1;
        }
        mode = atoi(argv[3]);
        if (mode > 2 || mode < 0) {
            fprintf(stderr, "mode needs to be a value between 0 and 2\n");
            return 1;
        }
        check = atoi(argv[4]);
        if (check >= 2 || check < 0) {
            fprintf(stderr, "check needs to be a 0 or 1\n");
            return 1;
        }
    }
    else {
        fprintf(stderr, "Usage: %s <filename> <max_steps> <mode> <check>\n", argv[0]);
        return 1;
    }

    PlaneList owning_planes = {NULL, NULL};
    read_planes_mpi(input_file, &owning_planes, &N, &M, &x_max, &y_max, rank, size, tile_displacements);
    PlaneList owning_planes_t0 = copy_plane_list(&owning_planes);

    //print_planes_par_debug(&owning_planes);

    double time_sim = 0., time_comm = 0, time_total=0;

    double start_time = MPI_Wtime();
    int step = 0;
    for (step = 1; step <= max_steps; step++) {
        double start = MPI_Wtime();
        PlaneNode* current = owning_planes.head;
        while (current != NULL) {
            current->x += current->vx;
            current->y += current->vy;
            current = current->next;
        }
        filter_planes(&owning_planes, x_max, y_max);
        time_sim += MPI_Wtime() - start;

        start = MPI_Wtime();
        if (mode == 0)
            communicate_planes_send(&owning_planes, N, M, x_max, y_max, rank, size, tile_displacements);
        else if (mode == 1)
            communicate_planes_alltoall(&owning_planes, N, M, x_max, y_max, rank, size, tile_displacements);
        else
            communicate_planes_struct_mpi(&owning_planes, N, M, x_max, y_max, rank, size, tile_displacements);
        time_comm += MPI_Wtime() - start;
    }
    time_total = MPI_Wtime() - start_time;

    /// TODO Check computational times


    if (rank == 0) {
        printf("Flight controller simulation: #input %s mode: %d size: %d\n", input_file, mode, size);
        printf("Time simulation:     %.2fs\n", time_sim);
        printf("Time communication:  %.2fs\n", time_comm);
        printf("Time total:          %.2fs\n", time_total);
    }

    if (check ==1)
        check_planes_mpi(&owning_planes_t0, &owning_planes, N, M, x_max, y_max, max_steps, tile_displacements, size, debug);
 
    MPI_Finalize();
    return 0;
}
