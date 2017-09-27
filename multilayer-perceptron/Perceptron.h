#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <cstdlib>

#include <utility>
#include <vector>

/**
 * Implementation of a simple multilayer perceptron neural network with one
 * hidden layer using the sigmoid activation function.
 */
class MultilayerPerceptron {
 public:
  // Each training data element must specify the input and the expected output.
  struct TrainingElement {
    const std::vector<double> input;
    const std::vector<double> output;

    TrainingElement(std::vector<double> in, std::vector<double> out)
        : input(std::move(in)), output(std::move(out)) {}
  };

  // Constructs a multilayer perceptron with the given input/output dimensions
  // and the number of neurons in the hidden layer.
  MultilayerPerceptron(
      size_t inputDimensiion, size_t outputDimension, size_t hiddenDimension);

  // Supplies a data set to train the multilayer perceptron.
  void setTrainingSet(std::vector<TrainingElement> trainingSet);

  // Trains the perceptron with the given learning rate.
  double train(double eta);

 private:
  struct WeightMatrix {
    const size_t inputDimension;
    const size_t outputDimension;
    std::vector<double> weights;

    WeightMatrix(size_t inputDim, size_t outputDim, double initialWeightScale)
        : inputDimension(inputDim), outputDimension(outputDim) {
      for (size_t i = 0; i < inputDimension * outputDimension; ++i) {
        weights.push_back(
            2 * initialWeightScale *
            (rand() / static_cast<double>(RAND_MAX)) - initialWeightScale);
      }
    }
  };

  struct Layer {
    const size_t dimension;
    std::vector<double> input;
    std::vector<double> output;
    std::vector<double> error;

    explicit Layer(size_t dim) : dimension(dim), input(dimension, 0),
                                 output(dimension, 0), error(dimension, 0) {}
  };

  // Sigmoid activation function to normalize the input.
  double psi(double x);

  // Derivative of sigmoid activation function to back propagate errors.
  double dpsidx(double x);

  void calculateLayerInput(size_t index);
  void calculateLayerOutput(size_t index);
  void calculateLayerError(size_t index);
  void updateWeights(size_t index, double eta);
  std::vector<double> classify(const std::vector<double>& x);

  const size_t inputDimension_;
  const size_t outputDimension_;
  const size_t hiddenDimension_;

  std::vector<WeightMatrix> weights_;
  std::vector<Layer> layers_;
  std::vector<TrainingElement> trainingSet_;
};

#endif  // PERCEPTRON_H
