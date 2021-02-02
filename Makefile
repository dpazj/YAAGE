CXX = g++

CFLAGS = -Wall -Werror
CPPFLAGS = $(CFLAGS)

LDFLAGS = 

OBJ=main.o \

%.o: %.cpp $(DEPS)
	$(CXX) -c -o $@ $< $(CPPFLAGS)

main: $(OBJ)
	$(CXX) $(OBJ) -o main $(LDFLAGS) 

run:
	./main

clean:
	rm -f $(OBJ) main
