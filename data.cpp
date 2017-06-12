#include "data.hpp"

#include <cstdlib>
#include <fstream>
#include <iostream>
using namespace std;

int swap_bytes(int x) {
  return ((x >> 24) & 0xff) | ((x >> 8) & 0xff00) | ((x << 8) & 0xff0000) |
         ((x << 24) & 0xff000000);
}

bool Dataset::load_images(string path) {
  ifstream ifs(path, ios::binary | ios::in);
  if (!ifs.is_open()) {
    cerr << "cannot open file: " << path << endl;
    return false;
  }
  ifs.seekg(4);

  int read_tmp = 0;
  ifs.read((char *)&read_tmp, 4);
  read_tmp = swap_bytes(read_tmp);
  if (num_data != 0 && num_data != read_tmp) {
    cerr << "data size isn't correct" << endl;
    return false;
  }
  num_data = read_tmp;

  ifs.read((char *)&read_tmp, 4);
  image_height = swap_bytes(read_tmp);
  ifs.read((char *)&read_tmp, 4);
  image_width = swap_bytes(read_tmp);

  images.resize(num_data);
  for (int n = 0; n < num_data; n++) {
    images[n].resize(image_height * image_width);
  }

  for (int n = 0; n < num_data; n++) {
    for (int h = 0; h < image_height; h++) {
      for (int w = 0; w < image_width; w++) {
        unsigned char tmp;
        ifs.read((char *)&tmp, 1);
        images[n][image_height * h + w] = (unsigned)(tmp) / 255.0;
      }
    }
  }

  ifs.close();

  return true;
}

bool Dataset::load_labels(string path) {
  ifstream ifs(path, ios::binary | ios::in);
  if (!ifs.is_open()) {
    cerr << "cannot open file: " << path << endl;
    return false;
  }

  int magic_number;
  ifs.read((char *)&magic_number, 4);

  int tmp;
  ifs.read((char *)&tmp, 4);
  tmp = swap_bytes(tmp);
  if (num_data != 0 && num_data != tmp) {
    cerr << "data size isn't correct" << endl;
    return false;
  }
  num_data = tmp;

  labels.resize(num_data);
  for (int n = 0; n < num_data; n++) {
    unsigned char cls;
    ifs.read((char *)&cls, 1);
    labels[n] = (unsigned)cls;
  }

  ifs.close();

  return true;
}

bool Dataset::load_dataset() {
  if (!load_images(image_path) || !load_labels(label_path))
    return false;

  return true;
}
