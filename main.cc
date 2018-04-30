#include <iostream>
#include <vector>
#include <time.h>

using namespace std;

#include "Network.h"

int main() {
  srand(time(NULL));

  vector<unsigned> topology {3,2,1};
  vector<double>   inputs  {1, 1, 1};
  vector<double>   targets {4};
  vector<double>   results {};

  // pass in a topology structure for specifications
  Network net(topology);

  cout << "feeding forward..." << endl;
  net.feedForward(inputs);
  cout << "back propogating..." << endl;
  net.backPropogate(targets);
  cout << "getting results..." << endl;
  net.getResults(results);

  return 0;

}
