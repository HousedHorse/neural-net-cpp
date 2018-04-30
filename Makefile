OBJ = main.o Neuron.o Network.o
all: $(OBJ)
	g++ -o neural-net $(OBJ)

main.o: main.cc
	g++ -c main.cc

Neuron.o: Neuron.cc Neuron.h
	g++ -c Neuron.cc

Network.o: Network.cc Network.h
	g++ -c Network.cc

clean:
	rm *.o neural-net
