#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>

#include "network.hpp"

using namespace std;

float MultiClassifiedNetwork::activate(const float x) {
  return 1 / (1 + exp(-x));
}

float MultiClassifiedNetwork::d_activate(const float x) {
  float act = activate(x);
  return (1 - act) * act;
}

MultiClassifiedNetwork::MultiClassifiedNetwork(vector<int> layer_size)
    : layer_size(layer_size) {
  random_device seed_gen;
  default_random_engine engine(seed_gen());
  normal_distribution<float> dist(0.0, 1.0);

  depth = layer_size.size();
  input.resize(depth);
  output.resize(depth);
  delta.resize(depth);
  weight.resize(depth);

  for (int l = 0; l < depth; l++) {
    int size = layer_size[l];
    delta[l].resize(size);

    if (l < depth - 1)
      output[l].resize(size + 1);
    else
      output[l].resize(size);

    if (l == 0)
      continue;

    int before_size = layer_size[l - 1];
    float xavier = sqrt(before_size);

    input[l].resize(size);
    weight[l].resize(size);
    for (int m = 0; m < size; m++) {
      weight[l][m].resize(before_size + 1);
      for (int n = 0; n < before_size + 1; n++) {
        weight[l][m][n] = dist(engine) / xavier;
      }
    }
  }
}

pair<float, bool> MultiClassifiedNetwork::forward(vector<float> x, int t) {
  if (x.size() != layer_size[0]) {
    cerr << "input layer size is not correct" << endl;
    exit(1);
  }

  for (int i = 0; i < x.size(); i++) {
    output[0][i] = x[i];
  }
  output[0][output[0].size() - 1] = 1;

  float sum_exp = 0;
  for (int l = 1; l < depth; l++) {
    for (int j = 0; j < input[l].size(); j++) {
      input[l][j] = 0;
      for (int i = 0; i < output[l - 1].size(); i++) {
        input[l][j] += weight[l][j][i] * output[l - 1][i];
      }

      if (l < depth - 1) {
        output[l][j] = activate(input[l][j]);
      } else {
        output[l][j] = exp(input[l][j]);
        sum_exp += output[l][j];
      }
    }

    if (l < depth - 1) {
      output[l][output[l].size() - 1] = 1;
    } else {
      for (int i = 0; i < output[depth - 1].size(); i++) {
        output[depth - 1][i] /= sum_exp;
      }
    }
  }

  float loss = -1 * log(output[depth - 1][t]);
  bool is_correct = (predict() == t);
  return {loss, is_correct};
}

int MultiClassifiedNetwork::predict() {
  auto output_layer = output[depth - 1];
  auto iter = max_element(output_layer.begin(), output_layer.end());
  return distance(output_layer.begin(), iter);
}

void MultiClassifiedNetwork::backward(int t, float eta) {
  for (int i = 0; i < delta[depth - 1].size(); i++) {
    if (i == t)
      delta[depth - 1][i] = output[depth - 1][i] - 1;
    else
      delta[depth - 1][i] = output[depth - 1][i];
  }

  for (int l = depth - 2; l > 0; l--) {
    for (int i = 0; i < output[l].size(); i++) {
      float tmp = 0;
      for (int j = 0; j < input[l + 1].size(); j++) {
        tmp += delta[l + 1][j] * weight[l + 1][j][i];
      }
      delta[l][i] = tmp * d_activate(input[l][i]);
    }
  }

  for (int l = 1; l < depth; l++) {
    for (int j = 0; j < input[l].size(); j++) {
      for (int i = 0; i < output[l - 1].size(); i++) {
        weight[l][j][i] -= eta * delta[l][j] * output[l - 1][i];
      }
    }
  }
}

void MultiClassifiedNetwork::save(const string filename) {
  ofstream ofs(filename.c_str());

  if (!ofs) {
    cerr << "cannot save model file: " << filename << endl;
    exit(1);
  }

  ofs << depth << endl;
  for (int i = 0; i < layer_size.size() - 1; i++)
    ofs << layer_size[i] << " ";
  ofs << layer_size[layer_size.size() - 1] << endl;

  for (int l = 0; l < weight.size(); l++) {
    for (int i = 0; i < weight[l].size(); i++) {
      for (int j = 0; j < weight[l][i].size() - 1; j++) {
        ofs << weight[l][i][j] << " ";
      }
      ofs << weight[l][i][weight[l][i].size() - 1] << endl;
    }
  }

  ofs.close();
}

MultiClassifiedNetwork::MultiClassifiedNetwork(const char *filename) {
  ifstream ifs(filename);

  if (!ifs) {
    cerr << "cannot open model file: " << filename << endl;
    exit(1);
  }

  ifs >> depth;

  layer_size.resize(depth);
  for (int l = 0; l < depth; l++) {
    ifs >> layer_size[l];
  }

  input.resize(depth);
  output.resize(depth);
  delta.resize(depth);
  weight.resize(depth);

  for (int l = 0; l < depth; l++) {
    int size = layer_size[l];
    delta[l].resize(size);

    if (l < depth - 1)
      output[l].resize(size + 1);
    else
      output[l].resize(size);

    if (l == 0)
      continue;
    int before_size = layer_size[l - 1];

    input[l].resize(size);
    weight[l].resize(size + 1);
    for (int m = 0; m < size + 1; m++) {
      weight[l][m].resize(before_size);
      for (int n = 0; n < before_size; n++) {
        ifs >> weight[l][m][n];
      }
    }
  }

  ifs.close();
}