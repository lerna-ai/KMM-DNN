/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.neuralnetwork.preset

import com.kotlinnlp.simplednn.core.functionalities.activations.ELU
import com.kotlinnlp.simplednn.core.functionalities.activations.Softmax
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.neuralnetwork.preset.DeltaRNN
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertEquals
import kotlin.test.assertNull

/**
 *
 */
class DeltaRNNSpec : Spek({

  describe("a DeltaRNN Neural Netowork") {

    val hiddenActivation = ELU()
    val outputActivation = Softmax()
    val network = DeltaRNN(
      inputSize = 3,
      hiddenSize = 5,
      hiddenActivation = hiddenActivation,
      outputSize = 4,
      outputActivation = outputActivation)

    context("initialization") {

      context("input layer configuration") {

        val inputLayerConfig = network.layersConfiguration[0]

        it("should have the expected size") {
          assertEquals(3, inputLayerConfig.size)
        }

        it("should have a null activation function") {
          assertNull(inputLayerConfig.activationFunction)
        }

        it("should have a null connection type") {
          assertNull(inputLayerConfig.connectionType)
        }
      }

      context("hidden layer configuration") {

        val hiddenLayerConfig = network.layersConfiguration[1]

        it("should have the expected size") {
          assertEquals(5, hiddenLayerConfig.size)
        }

        it("should have the expected activation function") {
          assertEquals(hiddenActivation, hiddenLayerConfig.activationFunction)
        }

        it("should a DeltaRNN connection type") {
          assertEquals(LayerType.Connection.DeltaRNN, hiddenLayerConfig.connectionType)
        }
      }

      context("output layer configuration") {

        val outputLayerConfig = network.layersConfiguration[2]

        it("should have the expected size") {
          assertEquals(4, outputLayerConfig.size)
        }

        it("should have the expected activation function") {
          assertEquals(outputActivation, outputLayerConfig.activationFunction)
        }

        it("should a Feedforward connection type") {
          assertEquals(LayerType.Connection.Feedforward, outputLayerConfig.connectionType)
        }
      }
    }
  }
})
