#include "Perceptron.h"

#include <cmath>

#define NUM_LAYERS 3

MultilayerPerceptron::MultilayerPerceptron(
    size_t inputDimension, size_t outputDimension, size_t hiddenDimension)
    : inputDimension_(inputDimension),
      outputDimension_(outputDimension),
      hiddenDimension_(hiddenDimension) {
  layers_.push_back(Layer(inputDimension_));
  layers_.push_back(Layer(hiddenDimension_));
  layers_.push_back(Layer(outputDimension_));

  for (size_t h = 0; h < NUM_LAYERS - 1; ++h) {
    size_t dim0 = layers_[h].dimension;
    size_t dim1 = layers_[h + 1].dimension;
    weights_.push_back(WeightMatrix(dim0, dim1, 1.0f));
  }
}

void MultilayerPerceptron::calculateLayerInput(size_t index) {
  if (index > 0 && index < NUM_LAYERS) {
    WeightMatrix& w = weights_[index - 1];
    for (size_t i = 0; i < layers_[index].dimension; ++i) {
      layers_[index].input[i] = 0;
      for (size_t j = 0; j < layers_[index - 1].dimension; ++j) {
        layers_[index].input[i] +=
            layers_[index - 1].output[j] * w.weights[i * w.inputDimension + j];
      }
    }
  }
}

void MultilayerPerceptron::calculateLayerOutput(size_t index) {
  for (size_t i = 0; i < layers_[index].dimension; ++i) {
    layers_[index].output[i] = psi(layers_[index].input[i]);
  }
}

void MultilayerPerceptron::calculateLayerError(size_t index) {
  WeightMatrix& w = weights_[index];
  for (size_t i = 0; i < layers_[index].dimension; ++i) {
    double sum = 0;
    for (size_t j = 0; j < layers_[index + 1].dimension; ++j) {
      sum += w.weights[j * w.inputDimension + i] * layers_[index + 1].error[j];
    }
    layers_[index].error[i] = dpsidx(layers_[index].input[i]) * sum;
  }
}

void MultilayerPerceptron::updateWeights(size_t index, double eta) {
  WeightMatrix& w = weights_[index - 1];
  for (size_t i = 0; i < w.outputDimension; ++i) {
    for (size_t j = 0; j < w.inputDimension; ++j) {
      double dw = eta *
          (layers_[index].error[i] * layers_[index - 1].output[j]);
      w.weights[i * w.inputDimension + j] += dw;
    }
  }
}

std::vector<double> MultilayerPerceptron::classify(
    const std::vector<double>& x) {
  if (x.size() == inputDimension_) {
    for (size_t i = 0; i < inputDimension_; ++i) {
      layers_[0].output[i] = x[i];
    }
    for (size_t h=1; h < NUM_LAYERS; ++h) {
      calculateLayerInput(h);
      calculateLayerOutput(h);
    }
    return layers_[NUM_LAYERS - 1].output;
  }
  return x;
}

double MultilayerPerceptron::psi(double x) {
  return 1.0f / (1 + exp(-0.5 * x));
}


double MultilayerPerceptron::dpsidx(double x) {
  return psi(x) * (1 - psi(x));
}

void MultilayerPerceptron::setTrainingSet(
    std::vector<TrainingElement> trainingSet) {
  trainingSet_ = std::move(trainingSet);
}

double MultilayerPerceptron::train(double eta) {
  double trainingSetError = 0;
  for (size_t t = 0; t < trainingSet_.size(); ++t) {
    TrainingElement& te = trainingSet_[t];
    const std::vector<double>& x = te.input;
    const std::vector<double>& y_desired = te.output;
    std::vector<double> y_actual = classify(x);

    // Calculate global error.
    double err = 0;
    for (size_t i = 0; i < y_actual.size(); ++i) {
      err += pow(y_desired[i] - y_actual[i], 2);
    }
    trainingSetError += err * err;

    // Calculate error in output layer NUM_LAYERS - 1.
    for (size_t i = 0; i < layers_[NUM_LAYERS - 1].dimension; ++i) {
      layers_[NUM_LAYERS - 1].error[i] = y_desired[i] - y_actual[i];
    }

    // Back-propagate the error.
    for (int h = NUM_LAYERS - 2; h >= 0; h--) {
      calculateLayerError(h);
    }

    for (size_t h = 1; h < NUM_LAYERS; ++h) {
      updateWeights(h, eta);
    }
  }

  return sqrt(trainingSetError);
}
