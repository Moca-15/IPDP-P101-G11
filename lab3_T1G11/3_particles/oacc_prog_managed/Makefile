CC = nvc
CFLAGS = -acc=gpu -Minfo=all -lm
OBJ = partis_oacc_prog_managed

all:
	$(CC) $(CFLAGS) -gpu=managed $(OBJ).c -mp -o $(OBJ)
clean:
	rm $(OBJ)
