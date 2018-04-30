#ifndef NETWORK_H
#define NETWORK_H

#include <vector>

#include "Neuron.h"

class Network {

  public:
    Network(const std::vector<unsigned>& topology);
    void feedForward(const std::vector<double>& inputs);
    void backPropogate(const std::vector<double>& targets);
    void getResults(std::vector<double>& results) const;
  private:
    std::vector<Layer> m_layers; //layers[layer_num][neuron_num]
    double m_error;
    double m_recentAvgError;
    double m_recentAvgSmoothingFactor;

};
#endif
