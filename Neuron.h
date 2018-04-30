#ifndef NEURON_H
#define NEURON_H

#include <vector>
#include <cstdlib>

class Neuron;

typedef std::vector<Neuron> Layer;

typedef struct Connection{
  double weight;  // weight
  double dWeight; // change in weight
} Connection;

class Neuron {

  public:
    Neuron(unsigned numOutputs, unsigned index);
    void setOutput(double o);
    double getOutput(void) const;
    void feedForward(const Layer& prev);
    void calcOutputGradients(double target);
    void calcHiddenGradients(Layer& nextLayer);
    void updateInputWeights(Layer& prevLayer);
  private:
    static double eta;   // [0.0..1.0] training rate
    static double alpha; // [0.0..1.0] multiplier of old delta weight
    static double randomWeight(void) { return rand() / double(RAND_MAX); }
    static double activation(double x);
    static double activationDerivative(double x);
    double m_output;
    double m_gradient;
    unsigned m_index;
    std::vector<Connection> m_weights;

};
#endif
