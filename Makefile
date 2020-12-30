

CXX = g++

CFLAGS = -Wall -Werror
CPPFLAGS = $(CFLAGS)

LDFLAGS = 


OBJ=main.o \
	# src/matrix.o \
	# src/matrix_ops.o \
	# src/node.o \
	# src/graph.o \
	

%.o: %.cpp $(DEPS)
	$(CXX) -c -o $@ $< $(CPPFLAGS)

main: $(OBJ)
	$(CXX) $(OBJ) -o main $(LDFLAGS) 

run:
	./main

clean:
	rm -f $(OBJ) main








