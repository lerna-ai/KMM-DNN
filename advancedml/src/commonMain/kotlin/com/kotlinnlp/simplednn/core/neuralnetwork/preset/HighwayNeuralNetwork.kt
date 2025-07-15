package com.kotlinnlp.simplednn.core.neuralnetwork.preset

import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.functionalities.initializers.GlorotInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.Initializer
import com.kotlinnlp.simplednn.core.layers.LayerInterface
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.StackedLayersParameters

/**
 * The Highway neural network factory.
 */
object HighwayNeuralNetwork {

  /**
   * @param inputSize the size of the input layer
   * @param inputType the type of the input layer (Dense -default-, Sparse, SparseBinary)
   * @param hiddenSize the size of the hidden layers
   * @param hiddenActivation the activation function of the hidden layers
   * @param numOfHighway the number of hidden highway layers (at least 1, the default)
   * @param outputSize the size of the output layer
   * @param outputActivation the activation function of the output layer
   * @param weightsInitializer the initializer of the weights (zeros if null, default: Glorot)
   * @param biasesInitializer the initializer of the biases (zeros if null, default: Glorot)
   */
  operator fun invoke(inputSize: Int,
                      inputType: LayerType.Input = LayerType.Input.Dense,
                      hiddenSize: Int,
                      hiddenActivation: ActivationFunction?,
                      numOfHighway: Int = 1,
                      outputSize: Int,
                      outputActivation: ActivationFunction?,
                      weightsInitializer: Initializer? = GlorotInitializer(),
                      biasesInitializer: Initializer? = GlorotInitializer()): StackedLayersParameters {

    require(numOfHighway >= 1) { "The number of highway layers must be >= 1." }

    val layersConfiguration = mutableListOf<LayerInterface>().apply {

      add(LayerInterface(size = inputSize, type = inputType))

      add(LayerInterface(
        size = hiddenSize,
        activationFunction = hiddenActivation,
        connectionType = LayerType.Connection.Feedforward))

      repeat(numOfHighway) {
        add(LayerInterface(
          size = hiddenSize,
          activationFunction = hiddenActivation,
          connectionType = LayerType.Connection.Highway))
      }

      add(LayerInterface(
        size = outputSize,
        activationFunction = outputActivation,
        connectionType = LayerType.Connection.Feedforward))
    }

    return StackedLayersParameters(
      layersConfiguration = *layersConfiguration.toTypedArray(),
      weightsInitializer = weightsInitializer,
      biasesInitializer = biasesInitializer)
  }
}
