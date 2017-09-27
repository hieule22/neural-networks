#include "Perceptron.h"

#include <iostream>

int main() {
  size_t numHiddenNeurons;
  std::cout << "Number of neurons in hidden layers: ";
  std::cin >> numHiddenNeurons;

  for (size_t nIterations = 1; nIterations < 30; ++nIterations) {
    MultilayerPerceptron mlp(1, 1, numHiddenNeurons);
    std::vector<MultilayerPerceptron::TrainingElement> trainingData;

    for (int i = 0; i < 21; ++i) {
      std::vector<double> input;
      std::vector<double> output;
      double x = -1.0 + i / 10.0;
      input.push_back(x);
      output.push_back(x * x + 1);
      trainingData.push_back({input, output});
    }

    mlp.setTrainingSet(trainingData);

    for (size_t i = 0; i < nIterations; ++i)
      mlp.train(0.2f);

    std::cout << nIterations << " " << mlp.train(0.2f) << std::endl;
  }

  return 0;
}
