#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>
#include <tuple>
#include <unistd.h>

#include "data.hpp"
#include "network.hpp"

using namespace std;

const string TRAIN_IMAGE = "./data/train-images-idx3-ubyte";
const string TRAIN_LABEL = "./data/train-labels-idx1-ubyte";
const string TEST_IMAGE = "./data/t10k-images-idx3-ubyte";
const string TEST_LABEL = "./data/t10k-labels-idx1-ubyte";

const int NUM_TRAIN = 20;
const int NUM_MINIBATCH = 1;

int main(int argc, char **argv) {

	int num_train = NUM_TRAIN;
	int num_minibatch = NUM_MINIBATCH;
	string model_filename = "";

	int opt;
	opterr = 0;
	while((opt = getopt(argc, argv, "hb:e:m:")) != -1) {
		switch (opt) {
			case 'e':
				num_train = atoi(optarg);
				break;

			case 'b':
				num_minibatch = atoi(optarg);
				break;

			case 'm':
				model_filename = optarg;
				break;

			case 'h':
				fprintf(stdout ,"Usage: %s [-h] [-b minibatch_size] [-e n_epoch] [-m path_to_model_file]", argv[0]);
				exit(0);

			default:
				fprintf(stderr, "Usage: %s [-h] [-b minibatch_size] [-e n_epoch] [-m path_to_model_file]", argv[0]);
				exit(1);
		}
	}

  cout << "load datasets:" << endl;
  Dataset train(TRAIN_IMAGE, TRAIN_LABEL), test(TEST_IMAGE, TEST_LABEL);

  if (!train.load_dataset()) {
    cerr << "cannot load train dataset" << endl;
    exit(1);
  }
  cout << "\t"
       << "train data size: " << train.get_size() << endl;

  test.load_dataset();
  if (!train.load_dataset()) {
    cerr << "cannot load test dataset" << endl;
    exit(1);
  }
  cout << "\t"
       << "test data size: " << test.get_size() << endl;
  cout << "\t"
       << "done" << endl;


	MultiClassifiedNetwork net;
	if (model_filename.empty()) {
		cout << "init network:" << endl;
		vector<int> layer_size = {{28 * 28, 1000, 1000, 10}};
		net = MultiClassifiedNetwork(layer_size);
		cout << "\t"
			<< "done" << endl;
	} else {
		cout << "load network model" << endl;
		net = MultiClassifiedNetwork(model_filename);
		cout << "\t" << "done" << endl;
	}

  float best_loss = 1e9;
  float learning_rate = 1;
  vector<int> random_idx(train.get_size());
  iota(random_idx.begin(), random_idx.end(), 0);
	shuffle(random_idx.begin(), random_idx.end(), mt19937());

	cout << "minibatch size: " << num_minibatch << endl;
	cout << "train epoch: " << num_train << endl;
  for (int epoch = 0; epoch < num_train; epoch++) {
    cout << "epoch " << epoch + 1 << ": " << endl;

    if (epoch % 10 == 0) {
      learning_rate /= 10;
      cout << "\t"
           << "learning_rate = " << learning_rate << endl;
    }

    // train
    float train_loss = 0;
    int train_correct = 0;
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
      if ((i + 1) % num_minibatch == 0)
        net.update_weight(learning_rate, num_minibatch);
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
