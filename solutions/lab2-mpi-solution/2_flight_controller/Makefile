CC=mpicc
CFLAGS=-O3 -march=native -lm -std=c99
OBJ=fc


all:
	$(CC) $(OBJ)_seq.c auxiliar.c -o $(OBJ)_seq $(CFLAGS)
	$(CC) $(OBJ)_mpi.c auxiliar.c -o $(OBJ)_mpi $(CFLAGS)

clean:
	rm $(OBJ)

