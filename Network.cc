#include <iostream>
#include <cassert>
#include <cmath>

#include "Network.h"

using namespace std;

Network::Network(const vector<unsigned> &topology) {
  for(unsigned i = 0; i < topology.size(); i++){
    // add a layer
    m_layers.push_back(Layer());
    // number of outputs is 0 if output layer, number of non-bias neurons
    // in next layer otherwise
    unsigned numOutputs = i == topology.size() - 1 ? 0 : topology[i+1];

    // add neurons (with an extra bias for each layer)
    for(unsigned j = 0; j <= topology[i]; j++){
      m_layers.back().push_back(Neuron(numOutputs, j));
      cout << "Made a Neuron..." << endl;
    }

    // set bias node's output to 1.0
    m_layers.back().back().setOutput(1.0);
  }
}

void Network::feedForward(const std::vector<double> &inputs) {
  // we assert that our number of input values be equal to  the number
  // of non-bias neurons in the input layer
  assert(inputs.size() == m_layers[0].size() - 1);

  // assign input to input neurons
  for(unsigned i = 0; i < inputs.size(); i++){
    m_layers[0][i].setOutput(inputs[i]);
  }

  // forward propogate
  for(unsigned i = 1; i < m_layers.size(); i++){
    for(unsigned j = 0; j < m_layers[i].size() - 1; j++){
      // feed forward with the previous layer
      Layer &prevLayer = m_layers[i-1];
      m_layers[i][j].feedForward(prevLayer);
    }
  }
}

void Network::backPropogate(const std::vector<double> &targets) {
  // calculate net error (Root Mean Square of Error)
  Layer& outputLayer = m_layers.back();
  m_error = 0.0;

  // m_error will be the sum of the squares of the error on each output value
  for(unsigned i = 0; i < outputLayer.size() - 1; i++){
    double delta = targets[i] - outputLayer[i].getOutput();
    m_error += delta*delta;
  }

  // get the average of this
  m_error /= outputLayer.size() - 1;
  m_error = sqrt(m_error); // Root Mean Square of Error

  // recent average measurement
  m_recentAvgError = (m_recentAvgError * m_recentAvgSmoothingFactor + m_error)
    / (m_recentAvgSmoothingFactor + 1.0);

  // calculate output layer gradients
  for(unsigned i = 0; i < outputLayer.size() - 1; i++){
    outputLayer[i].calcOutputGradients(targets[i]);
  }

  // calculate gradients on hidden layers, starting with right most
  for(unsigned i = m_layers.size() - 2; i > 0; i--){
    Layer& hiddenLayer = m_layers[i];
    Layer& nextLayer = m_layers[i+1];

    // calculate hidden layer gradients
    for(unsigned j = 0; j < hiddenLayer.size(); j++){
      hiddenLayer[j].calcHiddenGradients(nextLayer);
    }
  }

  // for all layers from outputs to first hidden layer inclusive,
  // update conection weights
  for(unsigned i = m_layers.size() - 1; i > 0; i--){
    Layer& layer = m_layers[i];
    Layer& prevLayer = m_layers[i-1];

    for(unsigned j = 0; j < layer.size() - 1; j++){
      layer[j].updateInputWeights(prevLayer);
    }
  }
}

void Network::getResults(std::vector<double> &results) const {
  results.clear();

  for(unsigned i = 0; i < m_layers.back().size() - 1; i++){
    results.push_back(m_layers.back()[i].getOutput());
  }
}
