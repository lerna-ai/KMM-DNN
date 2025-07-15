/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain contexte at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.feedforward.simple

import com.kotlinnlp.simplednn.core.arrays.DistributionArray
import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.layers.models.feedforward.simple.FeedforwardLayerParameters
import com.kotlinnlp.simplednn.core.optimizer.getErrorsOf
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertTrue

/**
 *
 */
class FeedforwardLayerStructureSpec : Spek({

  describe("a FeedForwardLayerStructure") {

    context("forward") {

      context("input size 4 and output size 5 (tanh)") {

        val layer = FeedforwardLayerStructureUtils.buildLayer45()
        layer.forward()

        it("should match the expected output values") {
          assertTrue {
            layer.outputArray.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.39693f, -0.79688f, 0.0f, 0.70137f, -0.18775f)),
              tolerance = 1.0e-05f)
          }
        }
      }

      context("input size 5 (activated with tanh) and output size 3 (softmax)") {

        val layer = FeedforwardLayerStructureUtils.buildLayer53()
        layer.inputArray.setActivation(Tanh)
        layer.inputArray.activate()
        layer.forward()

        it("should match the expected output values") {

          assertTrue {
            layer.outputArray.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.18504f, 0.29346f, 0.5215f)),
              tolerance = 1.0e-05f)
          }
        }
      }

      context("input size 5 and output size 3 (softmax)") {

        val layer = FeedforwardLayerStructureUtils.buildLayer53()
        layer.forward()

        it("should match the expected output values") {

          assertTrue {
            layer.outputArray.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.18687f, 0.28442f, 0.52871f)),
              tolerance = 1.0e-05f)
          }
        }
      }
    }

    context("forward with relevance") {

      context("dense input") {

        val layer = FeedforwardLayerStructureUtils.buildLayer53()
        val contributions = FeedforwardLayerParameters(inputSize = 5, outputSize = 3)

        layer.forward(contributions)

        it("should match the expected output values") {
          assertTrue {
            layer.outputArray.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.18687f, 0.28442f, 0.52871f)),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected contributions") {
          val wContr: DenseNDArray = contributions.unit.weights.values
          assertTrue {
            wContr.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(-0.42f, 0.54f, -0.1f, -0.8f, -0.08f),
                floatArrayOf(-0.34f, -0.46f, 0.02f, 0.44f, -0.1f),
                floatArrayOf(0.08f, 0.04f, 0.04f, 0.04f, -0.02f)
              )),
              tolerance = 1.0e-05f)
          }
        }

        layer.setOutputRelevance(DistributionArray.uniform(length = 3))
        layer.setInputRelevance(contributions)

        it("should match the expected input relevance") {
          val relevance: DenseNDArray = layer.inputArray.relevance
          assertTrue {
            relevance.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.55888f, 0.20978f, 0.09943f, 0.05652f, 0.07539f)),
              tolerance = 1.0e-05f)
          }
        }
      }
    }

    context("backward") {

      context("input size 4 and output size 5 (tanh)") {

        val layer = FeedforwardLayerStructureUtils.buildLayer45()
        val outputGold = FeedforwardLayerStructureUtils.getOutputGold5()

        layer.forward()

        layer.outputArray.assignErrors(layer.outputArray.values.sub(outputGold))
        val paramsErrors = layer.backward(propagateToInput = true)

        val params = layer.params

        it("should match the expected errors of the outputArray") {
          assertTrue {
            layer.outputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.33439f, -0.47334f, 0.4f, 0.81362f, -1.0494f)),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected errors of the biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.unit.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.33439f, -0.47334f, 0.4f, 0.81362f, -1.0494f)),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected errors of the weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.unit.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(0.26751f, 0.30095f, 0.30095f, -0.33439f),
                floatArrayOf(0.37867f, 0.42601f, 0.42601f, -0.47334f),
                floatArrayOf(-0.32f, -0.36f, -0.36f, 0.4f),
                floatArrayOf(-0.65089f, -0.73226f, -0.73226f, 0.81362f),
                floatArrayOf(0.83952f, 0.94446f, 0.94446f, -1.04940f)
              )),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected errors of the inputArray") {

          assertTrue {
            layer.inputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.0126f, -2.07296f, 1.07476f, -0.14158f)),
              tolerance = 1.0e-05f)
          }
        }
      }

      context("input size 5 (activated with tanh) and output size 3 (softmax)") {

        val layer = FeedforwardLayerStructureUtils.buildLayer53()
        val outputGold = FeedforwardLayerStructureUtils.getOutputGold3()

        layer.inputArray.setActivation(Tanh)
        layer.inputArray.activate()
        layer.forward()

        layer.outputArray.assignErrors(layer.outputArray.values.sub(outputGold))
        val paramsErrors = layer.backward(propagateToInput = true)

        val params = layer.params

        it("should match the expected errors of the outputArray") {
          assertTrue {
            layer.outputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.81496f, 0.29346f, 0.5215f)),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected errors of the biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.unit.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.81496f, 0.29346f, 0.5215f)),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected errors of the weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.unit.weights)!!.values).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(0.30964f, 0.54116f, 0.0f, -0.49254f, 0.16085f),
                floatArrayOf(-0.1115f, -0.19487f, 0.0f, 0.17736f, -0.05792f),
                floatArrayOf(-0.19814f, -0.34629f, 0.0f, 0.31518f, -0.10293f)
              )),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected errors of the inputArray") {

          assertTrue {
            layer.inputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.440004f, 0.828046f, -0.371592f, 0.991038f, 0.414023f)),
              tolerance = 1.0e-05f)
          }
        }
      }

      context("input size 5 and output size 3 (softmax)") {

        val layer = FeedforwardLayerStructureUtils.buildLayer53()
        val outputGold = FeedforwardLayerStructureUtils.getOutputGold3()

        layer.forward()

        layer.outputArray.assignErrors(layer.outputArray.values.sub(outputGold))
        val paramsErrors = layer.backward(propagateToInput = true)

        val params = layer.params

        it("should match the expected errors of the outputArray") {
          assertTrue {
            layer.outputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.81313f, 0.28442f, 0.52871f)),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected errors of the biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.unit.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.81313f, 0.28442f, 0.52871f)),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected errors of the weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.unit.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(0.32525f, 0.6505f, 0.0f, -0.56919f, 0.16263f),
                floatArrayOf(-0.11377f, -0.22753f, 0.0f, 0.19909f, -0.05688f),
                floatArrayOf(-0.21148f, -0.42297f, 0.0f, 0.37010f, -0.10574f)
              )),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected errors of the inputArray") {

          assertTrue {
            layer.inputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.4474f, 0.82115f, -0.37411f, 0.98377f, 0.41057f)),
              tolerance = 1.0e-05f)
          }
        }
      }
    }
  }
})
