CC = nvc
CFLAGS = -acc=gpu -Minfo=all -lm
OBJ = partis_oacc

all:
	$(CC) $(CFLAGS) -gpu=cc90 $(OBJ).c -mp  -o $(OBJ)

managed:
	$(CC) $(CFLAGS) -gpu=cc90 $(OBJ)_managed.c -mp  -o $(OBJ)_managed

unmanaged:
	$(CC) $(CFLAGS) -gpu=managed $(OBJ)_unmanaged.c -mp -o $(OBJ)_unmanaged

clean:
	rm $(OBJ)
