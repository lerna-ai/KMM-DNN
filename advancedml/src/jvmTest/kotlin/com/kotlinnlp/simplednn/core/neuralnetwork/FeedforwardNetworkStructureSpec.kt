/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.neuralnetwork

import com.kotlinnlp.simplednn.core.functionalities.activations.*
import com.kotlinnlp.simplednn.core.layers.LayerInterface
import com.kotlinnlp.simplednn.core.layers.LayerType.Connection
import com.kotlinnlp.simplednn.core.layers.StackedLayers
import com.kotlinnlp.simplednn.core.layers.models.feedforward.simple.FeedforwardLayer
import com.kotlinnlp.simplednn.core.layers.StackedLayersParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.core.layers.feedforward.simple.FeedforwardLayerStructureUtils
import com.kotlinnlp.simplednn.core.neuralnetwork.utils.FeedforwardNetworkStructureUtils
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 *
 */
class FeedforwardNetworkStructureSpec : Spek({

  describe("a FeedforwardStackedLayers") {

    context("invalid configurations") {

      context("initialization with null output connection types") {

        val wrongLayersConfiguration = arrayOf(
          LayerInterface(size = 4),
          LayerInterface(size = 5, activationFunction = Tanh),
          LayerInterface(size = 3, activationFunction = Softmax(), connectionType = Connection.Feedforward)
        ).toList()

        it("should throw an exception") {
          assertFailsWith<NullPointerException> {
            StackedLayers<DenseNDArray>(params = StackedLayersParameters(wrongLayersConfiguration), dropout = 0.0f)
          }
        }
      }
    }

    context("correct configuration") {

      val layersConfiguration = arrayOf(
        LayerInterface(size = 4),
        LayerInterface(size = 5, activationFunction = Tanh, connectionType = Connection.Feedforward),
        LayerInterface(size = 3, activationFunction = Softmax(), connectionType = Connection.Feedforward)
      ).toList()

      val structure = StackedLayers<DenseNDArray>(
        params = FeedforwardNetworkStructureUtils.buildParams(layersConfiguration),
        dropout = 0.0f)

      context("architecture") {

        it("should have the expected number of layers") {
          assertEquals(2, structure.layers.size)
        }

        it("should have interconnected layers") {
          for (i in 0 until structure.layers.size - 1) {
            assertEquals(structure.layers[i].outputArray, structure.layers[i + 1].inputArray)
          }
        }

        it("should contain the expected input layer") {
          assertEquals(structure.inputLayer, structure.layers[0])
        }

        it("should contain the expected output layer") {
          assertEquals(structure.outputLayer, structure.layers[1])
        }
      }

      context("layers factory") {

        it("should contain layers of the expected type") {
          structure.layers.forEach { assertTrue { it is FeedforwardLayer } }
        }
      }

      context("methods usage") {

        val features = DenseNDArrayFactory.arrayOf(floatArrayOf(-0.8f, -0.9f, -0.9f, 1.0f))
        val output = structure.forward(features)
        val expectedOutput = DenseNDArrayFactory.arrayOf(floatArrayOf(0.19f, 0.29f, 0.53f))

        it("should return the expected output after a call of the forward method") {
          assertTrue { output.equals(expectedOutput, tolerance = 0.005f) }
        }

        val outputGold = FeedforwardLayerStructureUtils.getOutputGold3()

        structure.backward(
          outputErrors = structure.outputLayer.outputArray.values.sub(outputGold),
          propagateToInput = true)

        val inputErrors = structure.inputLayer.inputArray.errors
        val expectedInputErrors = DenseNDArrayFactory.arrayOf(floatArrayOf(0.32f, -0.14f, -0.06f, 0.07f))

        it("should contain the expected input error after a call of the backward method") {
          assertTrue { inputErrors.equals(expectedInputErrors, tolerance = 0.005f) }
        }
      }
    }
  }
})
