
INCLUDE = -I/usr/local/include/NumCpp/

CXX = g++

CFLAGS = -Wall -Werror
CPPFLAGS = $(CFLAGS) $(INCLUDE)

LDFLAGS = 


OBJ=main.o \
	src/tensor.o \
	src/node.o \
	src/graph.o \



%.o: %.cpp $(DEPS)
	$(CXX) -c -o $@ $< $(CPPFLAGS)

test: $(OBJ)
	$(CXX) $(OBJ) -o main $(LDFLAGS) 

run:
	./main

clean:
	rm -f $(OBJ) main








