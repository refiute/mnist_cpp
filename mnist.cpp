#include <algorithm>
#include <array>
#include <iostream>
#include <numeric>
#include <random>
#include <tuple>
#include <tuple>

#include "data.hpp"
#include "network.hpp"

using namespace std;

const string TRAIN_IMAGE = "./data/train-images-idx3-ubyte";
const string TRAIN_LABEL = "./data/train-labels-idx1-ubyte";
const string TEST_IMAGE = "./data/t10k-images-idx3-ubyte";
const string TEST_LABEL = "./data/t10k-labels-idx1-ubyte";

const int NUM_TRAIN = 20;
const int NUM_MINIBATCH = 50;

int main(int argc, char **argv) {
  cout << "load datasets:" << endl;
  Dataset train(TRAIN_IMAGE, TRAIN_LABEL), test(TEST_IMAGE, TEST_LABEL);

  train.load_dataset();
  cout << "\t"
       << "train data size: " << train.get_size() << endl;

  test.load_dataset();
  cout << "\t"
       << "test data size: " << test.get_size() << endl;
  cout << "\t"
       << "done" << endl;

  /*
cout << "load network" << endl;
MultiClassifiedNetwork net(argv[1]);
cout << "\t" << "done" << endl;
*/

  cout << "init network:" << endl;
  vector<int> layer_size = {{28 * 28, 1000, 1000, 10}};
  MultiClassifiedNetwork net(layer_size);
  cout << "\t"
       << "done" << endl;

  float best_loss = 1e9;
  float learning_rate = 1;
  vector<int> random_idx(train.get_size());
  iota(random_idx.begin(), random_idx.end(), 0);

  for (int epoch = 0; epoch < NUM_TRAIN; epoch++) {
    cout << "epoch " << epoch + 1 << ": " << endl;

    if (epoch % 10 == 0) {
      learning_rate /= 10;
      cout << "\t"
           << "learning_rate = " << learning_rate << endl;
    }

    // train
    float train_loss = 0;
    int train_correct = 0;
    shuffle(random_idx.begin(), random_idx.end(), mt19937());

    for (int i = 0; i < train.get_size(); i++) {
      cout << "\r\t"
           << "train: " << i + 1 << "/" << train.get_size() << flush;

      float loss;
      bool is_correct;
      tie(loss, is_correct) = net.forward(train.get_image(random_idx[i]),
                                          train.get_label(random_idx[i]));
      train_loss += loss;
      train_correct += is_correct;

      net.backward(train.get_label(random_idx[i]));
      if ((i + 1) % NUM_MINIBATCH == 0)
        net.update_weight(learning_rate);
    }
    cout << endl;

    double train_accuracy = 100.0 * train_correct / train.get_size();
    cout << "\t"
         << "train_accuracy: " << train_accuracy << "(" << train_correct << "/"
         << train.get_size() << ")" << endl;
    cout << "\t"
         << "train_loss: " << train_loss << endl;

    // test
    float test_loss = 0;
    int test_correct = 0;

    for (int i = 0; i < test.get_size(); i++) {
      cout << "\r\t"
           << "test: " << i + 1 << "/" << test.get_size() << flush;

      float loss;
      bool is_correct;
      tie(loss, is_correct) = net.forward(test.get_image(i), test.get_label(i));
      test_loss += loss;
      test_correct += is_correct;
    }
    cout << endl;

    double test_accuracy = 100.0 * test_correct / test.get_size();
    cout << "\t"
         << "test_accuracy: " << test_accuracy << "(" << test_correct << "/"
         << test.get_size() << ")" << endl;
    cout << "\t"
         << "test_loss: " << test_loss << endl;

    if (best_loss > test_loss) {
      cout << "\t"
           << "best loss!" << endl;

      best_loss = test_loss;
      net.save("best.model");
    }
  }
}
