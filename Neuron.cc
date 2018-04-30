#include <cmath>

#include "Neuron.h"

double Neuron::eta   = 0.15; // [0.0..1.0] training rate
double Neuron::alpha = 0.5;  // [0.0..1.0] multiplier of old delta weight

Neuron::Neuron(unsigned numOutputs, unsigned index) : m_index(index) {
  // create the correct number of connections
  for(unsigned i = 0; i < numOutputs; i++){
    m_weights.push_back(Connection());
    m_weights.back().weight = randomWeight();
    m_weights.back().dWeight = 0;
  }
}

void   Neuron::setOutput(double o) { m_output = o; }

double Neuron::getOutput(void) const { return m_output; }

void   Neuron::feedForward(const Layer& prev) {
  double sum = 0.0;

  // loop through all neurons in prev 0 .. num neurons, including bias
  for(unsigned i = 0; i < prev.size(); i++){
    sum += prev[i].m_output * prev[i].m_weights[m_index].weight;
  }

  m_output = activation(sum);
}

double Neuron::activation(double x) {
  // tanh with range [-1.0..1.0]
  return tanh(x);
}

double Neuron::activationDerivative(double x) {
  // tanh derivative
  // this is an approximation but it's good enough
  return 1.0 - x * x;
}

void Neuron::calcOutputGradients(double target) {
  double delta = target - m_output;

  m_gradient = delta * activationDerivative(m_output);
}

void Neuron::calcHiddenGradients(Layer& nextLayer) {
  double dWeights = 0.0;

  // sum error contributions of nodes in next layer
  for(unsigned i = 0; i < nextLayer.size() - 1; i ++){
    dWeights += m_weights[i].weight * nextLayer[i].m_gradient;
  }

  m_gradient = dWeights * activationDerivative(m_output);
}

void Neuron::updateInputWeights(Layer& prevLayer) {
  for(unsigned i = 0;  i < prevLayer.size(); i++){
    Neuron& n = prevLayer[i];
    double oldDeltaWeight = n.m_weights[m_index].dWeight;

    // individual input, magnified by gradient and learning rate
    // add momentum, a fraction of the the old weight
    double newDeltaWeight = eta * n.getOutput() * m_gradient
      + alpha * oldDeltaWeight;

    n.m_weights[m_index].dWeight = newDeltaWeight;
    n.m_weights[m_index].weight += newDeltaWeight;
  }
}
