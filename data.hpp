#ifndef DATA_H
#define DATA_H

#include <map>
#include <string>
#include <vector>

using namespace std;

class Dataset {
private:
  string image_path;
  string label_path;

  int num_data;
  int image_width;
  int image_height;

  vector<vector<float>> images;
  vector<int> labels;

  bool load_images(string path);
  bool load_labels(string path);

public:
  Dataset(string image, string label) : image_path(image), label_path(label) {
    num_data = 0;
  };
  bool load_dataset();
  int get_size() { return num_data; }
  vector<float> get_image(int idx) { return images[idx]; }
  int get_label(int idx) { return labels[idx]; }
};

#endif
