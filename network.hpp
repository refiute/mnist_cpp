#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <string>
#include <vector>
using namespace std;

class MultiClassifiedNetwork {
private:
  int depth;
  vector<int> layer_size;
  vector<vector<float>> input;  // input[n_layer][n_unit]
  vector<vector<float>> output; // output[n_layer][n_unit]
  vector<vector<float>> delta;  // delta[n_layer][n_unit]

  // weight[n_curr_layer][n_curr_unit][n_bef_unit]
  vector<vector<vector<float>>> weight;
  vector<vector<vector<float>>> error;

  float activate(float x);
  float d_activate(float x);

public:
  MultiClassifiedNetwork() {}
  MultiClassifiedNetwork(const string filename);
  MultiClassifiedNetwork(vector<int> layer_size);
  pair<float, bool> forward(vector<float> x, int t);
  void backward(int t);
  void update_weight(float eta, int minibatch_size);
  int predict();
  void save(const string filename);
};

#endif
