/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.recurrent.lstm

import com.kotlinnlp.simplednn.core.functionalities.losses.MSECalculator
import com.kotlinnlp.simplednn.core.optimizer.getErrorsOf
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertTrue

/**
 *
 */
class LSTMLayerStructureSpec : Spek({

  describe("a LSTMLayer") {

    context("forward") {

      context("without previous state context") {

        val layer = LSTMLayerStructureUtils.buildLayer(LSTMLayersWindow.Empty)
        layer.forward()

        it("should match the expected input gate") {
          assertTrue {
            layer.inputGate.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.40f, 0.25f, 0.50f, 0.70f, 0.45f)),
              tolerance = 0.005f)
          }
        }

        it("should match the expected output gate") {
          assertTrue {
            layer.outputGate.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.85f, 0.43f, 0.12f, 0.52f, 0.24f)),
              tolerance = 0.005f)
          }
        }

        it("should match the expected forget gate") {
          assertTrue {
            layer.forgetGate.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.79f, 0.35f, 0.88f, 0.85f, 0.45f)),
              tolerance = 0.005f)
          }
        }

        it("should match the expected candidate") {
          assertTrue {
            layer.candidate.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.38f, -0.45f, -0.92f, 0.98f, -0.89f)),
              tolerance = 0.005f)
          }
        }

        it("should match the expected cell") {
          assertTrue {
            layer.cell.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.15f, -0.11f, -0.43f, 0.6f, -0.38f)),
              tolerance = 0.005f)
          }
        }

        it("should match the expected outputArray") {
          assertTrue {
            layer.outputArray.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.13f, -0.05f, -0.05f, 0.31f, -0.09f)),
              tolerance = 0.005f)
          }
        }
      }

      context("with previous state context") {

        val layer = LSTMLayerStructureUtils.buildLayer(LSTMLayersWindow.Back)
        layer.forward()

        it("should match the expected input gate") {
          assertTrue {
            layer.inputGate.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.72f, 0.25f, 0.55f, 0.82f, 0.53f)),
              tolerance = 0.005f)
          }
        }

        it("should match the expected output gate") {
          assertTrue {
            layer.outputGate.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.91f, 0.18f, 0.05f, 0.67f, 0.39f)),
              tolerance = 0.005f)
          }
        }

        it("should match the expected forget gate") {
          assertTrue {
            layer.forgetGate.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.91f, 0.62f, 0.84f, 0.91f, 0.62f)),
              tolerance = 0.005f)
          }
        }

        it("should match the expected candidate") {
          assertTrue {
            layer.candidate.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.23f, 0.33f, -0.95f, 0.99f, -0.93f)),
              tolerance = 0.005f)
          }
        }

        it("should match the expected cell") {
          assertTrue {
            layer.cell.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.51f, -0.28f, 0.31f, 0.72f, -0.41f)),
              tolerance = 0.005f)
          }
        }

        it("should match the expected outputArray") {
          assertTrue {
            layer.outputArray.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.47f, -0.05f, 0.01f, 0.48f, -0.16f)),
              tolerance = 0.005f)
          }
        }
      }

      context("with init hidden layer") {

        val contextWindow = LSTMLayersWindow.BackHidden()
        val layer = LSTMLayerStructureUtils.buildLayer(contextWindow)

        contextWindow.setRefLayer(layer)

        layer.forward()

        it("should match the expected input gate") {
          assertTrue {
            layer.inputGate.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.72f, 0.25f, 0.55f, 0.82f, 0.53f)),
              tolerance = 0.005f)
          }
        }

        it("should match the expected output gate") {
          assertTrue {
            layer.outputGate.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.91f, 0.18f, 0.05f, 0.67f, 0.39f)),
              tolerance = 0.005f)
          }
        }

        it("should match the expected forget gate") {
          assertTrue {
            layer.forgetGate.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.91f, 0.62f, 0.84f, 0.91f, 0.62f)),
              tolerance = 0.005f)
          }
        }

        it("should match the expected candidate") {
          assertTrue {
            layer.candidate.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.23f, 0.33f, -0.95f, 0.99f, -0.93f)),
              tolerance = 0.005f)
          }
        }

        it("should match the expected cell") {
          assertTrue {
            layer.cell.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.16f, 0.08f, -0.48f, 0.67f, -0.46f)),
              tolerance = 0.005f)
          }
        }

        it("should match the expected outputArray") {
          assertTrue {
            layer.outputArray.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.15f, 0.01f, -0.02f, 0.45f, -0.18f)),
              tolerance = 0.005f)
          }
        }
      }
    }

    context("backward") {

      context("without previous and next state") {

        val layer = LSTMLayerStructureUtils.buildLayer(LSTMLayersWindow.Empty)

        layer.forward()

        val errors = MSECalculator().calculateErrors(
          output = layer.outputArray.values,
          outputGold = LSTMLayerStructureUtils.getOutputGold())

        layer.outputArray.assignErrors(errors)
        val paramsErrors = layer.backward(propagateToInput = true)

        val params = layer.params

        it("should match the expected errors of the outputArray") {
          assertTrue {
            layer.outputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.7f, -0.8f, 0.1f, -1.33f, -0.54f)),
              tolerance = 0.005f)
          }
        }

        it("should match the expected errors of the cell") {
          assertTrue {
            layer.cell.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.58f, -0.34f, 0.01f, -0.44f, -0.11f)),
              tolerance = 0.005f)
          }
        }

        it("should match the expected errors of the candidate") {
          assertTrue {
            layer.candidate.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.20f, -0.07f, 0.0f, -0.01f, -0.01f)),
              tolerance = 0.005f)
          }
        }

        it("should match the expected errors of the input gate") {
          assertTrue {
            layer.inputGate.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.05f, 0.03f, 0.0f, -0.09f, 0.02f)),
              tolerance = 0.005f)
          }
        }

        it("should match the expected errors of the output gate") {
          assertTrue {
            layer.outputGate.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.01f, 0.02f, 0.0f, -0.2f, 0.04f)),
              tolerance = 0.005f)
          }
        }

        it("should match the expected errors of the forget gate") {
          assertTrue {
            layer.forgetGate.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f)),
              tolerance = 0.005f)
          }
        }

        it("should match the expected errors of the candidate biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.candidate.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.20f, -0.07f, 0.0f, -0.01f, -0.01f)),
              tolerance = 0.005f)
          }
        }

        it("should match the expected errors of the input gate biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.inputGate.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.05f, 0.03f, 0.0f, -0.09f, 0.02f)),
              tolerance = 0.005f)
          }
        }

        it("should match the expected errors of the output gate biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.outputGate.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.01f, 0.02f, 0.0f, -0.2f, 0.04f)),
              tolerance = 0.005f)
          }
        }

        it("should match the expected errors of the forget gate biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.forgetGate.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f)),
              tolerance = 0.005f)
          }
        }

        it("should match the expected errors of the candidate weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.candidate.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(0.16f, 0.18f, 0.18f, -0.2f),
                floatArrayOf(0.05f, 0.06f, 0.06f, -0.07f),
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f),
                floatArrayOf(0.01f, 0.01f, 0.01f, -0.01f),
                floatArrayOf(0.01f, 0.01f, 0.01f, -0.01f)
              )),
              tolerance = 0.005f)
          }
        }

        it("should match the expected errors of the input gate weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.inputGate.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(-0.04f, -0.05f, -0.05f, 0.05f),
                floatArrayOf(-0.02f, -0.03f, -0.03f, 0.03f),
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f),
                floatArrayOf(0.07f, 0.08f, 0.08f, -0.09f),
                floatArrayOf(-0.02f, -0.02f, -0.02f, 0.02f)
              )),
              tolerance = 0.005f)
          }
        }

        it("should match the expected errors of the output gate weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.outputGate.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(-0.01f, -0.01f, -0.01f, 0.01f),
                floatArrayOf(-0.02f, -0.02f, -0.02f, 0.02f),
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f),
                floatArrayOf(0.16f, 0.18f, 0.18f, -0.2f),
                floatArrayOf(-0.03f, -0.03f, -0.03f, 0.04f)
              )),
              tolerance = 0.005f)
          }
        }

        it("should match the expected errors of the forget gate weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.forgetGate.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f),
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f),
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f),
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f),
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f)
              )),
              tolerance = 0.005f)
          }
        }

        it("should match the expected errors of the candidate recurrent weights") {
          assertTrue {
            paramsErrors.getErrorsOf(params.candidate.recurrentWeights)!!.values.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f)
              )),
              tolerance = 0.005f)
          }
        }

        it("should match the expected errors of the input gate recurrent weights") {
          assertTrue {
            paramsErrors.getErrorsOf(params.inputGate.recurrentWeights)!!.values.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f)
              )),
              tolerance = 0.005f)
          }
        }

        it("should match the expected errors of the output gate recurrent weights") {
          assertTrue {
            paramsErrors.getErrorsOf(params.outputGate.recurrentWeights)!!.values.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f)
              )),
              tolerance = 0.005f)
          }
        }

        it("should match the expected errors of the forget gate recurrent weights") {
          assertTrue {
            paramsErrors.getErrorsOf(params.forgetGate.recurrentWeights)!!.values.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f)
              )),
              tolerance = 0.005f)
          }
        }

        it("should match the expected errors of the inputArray") {
          assertTrue {
            layer.inputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.12f, -0.14f, 0.03f, 0.02f)),
              tolerance = 0.005f)
          }
        }
      }

      context("with previous state only") {

        val layer = LSTMLayerStructureUtils.buildLayer(LSTMLayersWindow.Back)

        layer.forward()

        val errors = MSECalculator().calculateErrors(
          output = layer.outputArray.values,
          outputGold = LSTMLayerStructureUtils.getOutputGold())

        layer.outputArray.assignErrors(errors)
        val paramsErrors = layer.backward(propagateToInput = true)

        val params = layer.params

        it("should match the expected errors of the outputArray") {
          assertTrue {
            layer.outputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.104f, -0.801f, 0.165f, -1.156f, -0.609f)),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the cell") {
          assertTrue {
            layer.cell.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.07f, -0.133f, 0.007f, -0.378f, -0.198f)),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the candidate") {
          assertTrue {
            layer.candidate.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.048f, -0.03f, 0.0f, -0.006f, -0.015f)),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the input gate") {
          assertTrue {
            layer.inputGate.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.003f, -0.008f, -0.002f, -0.055f, 0.046f)),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the output gate") {
          assertTrue {
            layer.outputGate.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.004f, 0.033f, 0.002f, -0.182f, 0.059f)),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the forget gate") {
          assertTrue {
            layer.forgetGate.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.005f, 0.019f, 0.001f, -0.003f, -0.005f)),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the candidate biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.candidate.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.048f, -0.03f, 0.0f, -0.006f, -0.015f)),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the input gate biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.inputGate.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.003f, -0.008f, -0.002f, -0.055f, 0.046f)),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the output gate biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.outputGate.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.004f, 0.033f, 0.002f, -0.182f, 0.059f)),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the forget gate biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.forgetGate.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.005f, 0.019f, 0.001f, -0.003f, -0.005f)),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the candidate weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.candidate.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(0.038f, 0.043f, 0.043f, -0.048f),
                floatArrayOf(0.024f, 0.027f, 0.027f, -0.03f),
                floatArrayOf(0.00f, 0.00f, 0.00f, 0.00f),
                floatArrayOf(0.005f, 0.006f, 0.006f, -0.006f),
                floatArrayOf(0.012f, 0.013f, 0.013f, -0.015f)
              )),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the input gate weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.inputGate.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(-0.003f, -0.003f, -0.003f, 0.003f),
                floatArrayOf(0.007f, 0.007f, 0.007f, -0.008f),
                floatArrayOf(0.001f, 0.002f, 0.002f, -0.002f),
                floatArrayOf(0.044f, 0.05f, 0.05f, -0.055f),
                floatArrayOf(-0.036f, -0.041f, -0.041f, 0.046f)
              )),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the output gate weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.outputGate.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(0.003f, 0.004f, 0.004f, -0.004f),
                floatArrayOf(-0.027f, -0.03f, -0.03f, 0.033f),
                floatArrayOf(-0.002f, -0.002f, -0.002f, 0.002f),
                floatArrayOf(0.146f, 0.164f, 0.164f, -0.182f),
                floatArrayOf(-0.047f, -0.053f, -0.053f, 0.059f)
              )),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the forget gate weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.forgetGate.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(0.004f, 0.004f, 0.004f, -0.005f),
                floatArrayOf(-0.015f, -0.017f, -0.017f, 0.019f),
                floatArrayOf(-0.001f, -0.001f, -0.001f, 0.001f),
                floatArrayOf(0.003f, 0.003f, 0.003f, -0.003f),
                floatArrayOf(0.004f, 0.004f, 0.004f, -0.005f)
              )),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the candidate recurrent weights") {
          assertTrue {
            paramsErrors.getErrorsOf(params.candidate.recurrentWeights)!!.values.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(0.01f, -0.01f, 0.014f, 0.043f, 0.038f),
                floatArrayOf(0.006f, -0.006f, 0.009f, 0.027f, 0.024f),
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
                floatArrayOf(0.001f, -0.001f, 0.002f, 0.006f, 0.005f),
                floatArrayOf(0.003f, -0.003f, 0.004f, 0.013f, 0.012f)
              )),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the input gate recurrent weights") {
          assertTrue {
            paramsErrors.getErrorsOf(params.inputGate.recurrentWeights)!!.values.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(-0.001f, 0.001f, -0.001f, -0.003f, -0.003f),
                floatArrayOf(0.002f, -0.002f, 0.002f, 0.007f, 0.007f),
                floatArrayOf(0.0f, 0.0f, 0.001f, 0.002f, 0.001f),
                floatArrayOf(0.011f, -0.011f, 0.017f, 0.05f, 0.044f),
                floatArrayOf(-0.009f, 0.009f, -0.014f, -0.041f, -0.036f)
              )),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the output gate recurrent weights") {
          assertTrue {
            paramsErrors.getErrorsOf(params.outputGate.recurrentWeights)!!.values.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(0.001f, -0.001f, 0.001f, 0.004f, 0.003f),
                floatArrayOf(-0.007f, 0.007f, -0.01f, -0.03f, -0.027f),
                floatArrayOf(0.0f, 0.0f, -0.001f, -0.002f, -0.002f),
                floatArrayOf(0.036f, -0.036f, 0.055f, 0.164f, 0.146f),
                floatArrayOf(-0.012f, 0.012f, -0.018f, -0.053f, -0.047f)
              )),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the forget gate recurrent weights") {
          assertTrue {
            paramsErrors.getErrorsOf(params.forgetGate.recurrentWeights)!!.values.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(0.001f, -0.001f, 0.001f, 0.004f, 0.004f),
                floatArrayOf(-0.004f, 0.004f, -0.006f, -0.017f, -0.015f),
                floatArrayOf(0.0f, 0.0f, 0.0f, -0.001f, -0.001f),
                floatArrayOf(0.001f, -0.001f, 0.001f, 0.003f, 0.003f),
                floatArrayOf(0.001f, -0.001f, 0.001f, 0.004f, 0.004f)
              )),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the inputArray") {
          assertTrue {
            layer.inputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.106f, -0.055f, 0.002f, 0.058f)),
              tolerance = 0.0005f)
          }
        }
      }

      context("with init hidden") {

        val contextWindow = LSTMLayersWindow.BackHidden()
        val layer = LSTMLayerStructureUtils.buildLayer(contextWindow)

        contextWindow.setRefLayer(layer)

        layer.forward()

        val errors = MSECalculator().calculateErrors(
          output = layer.outputArray.values,
          outputGold = LSTMLayerStructureUtils.getOutputGold())

        layer.outputArray.assignErrors(errors)
        layer.backward(propagateToInput = true)

        it("should match the expected errors of the outputArray") {
          assertTrue {
            layer.outputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.718f, -0.735f, 0.127f, -1.187f, -0.629f)),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the inputArray") {
          assertTrue {
            layer.inputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.002f, -0.253f, -0.036f, 0.006f)),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the init hidden array") {
          assertTrue {
            contextWindow.getPrevState().getInitHiddenErrors().equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.020f, -0.051f, 0.017f, -0.264f, 0.449f)),
              tolerance = 0.0005f)
          }
        }
      }

      context("with next state only") {

        val layer = LSTMLayerStructureUtils.buildLayer(LSTMLayersWindow.Front())

        layer.forward()

        val errors = MSECalculator().calculateErrors(
          output = layer.outputArray.values,
          outputGold = LSTMLayerStructureUtils.getOutputGold())

        layer.outputArray.assignErrors(errors)
        val paramsErrors = layer.backward(propagateToInput = true)

        val params = layer.params

        it("should match the expected errors of the outputArray") {
          assertTrue {
            layer.outputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.188f, -0.849f, 1.42f, -1.998f, -2.242f)),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the cell") {
          assertTrue {
            layer.cell.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.066f, -0.683f, 1.034f, -0.346f, -0.704f)),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the candidate") {
          assertTrue {
            layer.candidate.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.023f, -0.136f, 0.081f, -0.009f, -0.068f)),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the input gate") {
          assertTrue {
            layer.inputGate.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.006f, 0.058f, -0.238f, -0.071f, 0.155f)),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the output gate") {
          assertTrue {
            layer.outputGate.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.004f, 0.024f, -0.063f, -0.299f, 0.157f)),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the forget gate") {
          assertTrue {
            layer.forgetGate.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f)),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the candidate biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.candidate.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.023f, -0.136f, 0.081f, -0.009f, -0.068f)),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the input gate biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.inputGate.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.006f, 0.058f, -0.238f, -0.071f, 0.155f)),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the output gate biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.outputGate.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.004f, 0.024f, -0.063f, -0.299f, 0.157f)),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the forget gate biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.forgetGate.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f)),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the candidate weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.candidate.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(0.018f, 0.02f, 0.02f, -0.023f),
                floatArrayOf(0.109f, 0.123f, 0.123f, -0.136f),
                floatArrayOf(-0.065f, -0.073f, -0.073f, 0.081f),
                floatArrayOf(0.007f, 0.008f, 0.008f, -0.009f),
                floatArrayOf(0.054f, 0.061f, 0.061f, -0.068f)
              )),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the input gate weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.inputGate.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(-0.005f, -0.005f, -0.005f, 0.006f),
                floatArrayOf(-0.047f, -0.053f, -0.053f, 0.058f),
                floatArrayOf(0.19f, 0.214f, 0.214f, -0.238f),
                floatArrayOf(0.057f, 0.064f, 0.064f, -0.071f),
                floatArrayOf(-0.124f, -0.139f, -0.139f, 0.155f)
              )),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the output gate weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.outputGate.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(-0.003f, -0.003f, -0.003f, 0.004f),
                floatArrayOf(-0.019f, -0.021f, -0.021f, 0.024f),
                floatArrayOf(0.05f, 0.056f, 0.056f, -0.063f),
                floatArrayOf(0.239f, 0.269f, 0.269f, -0.299f),
                floatArrayOf(-0.126f, -0.141f, -0.141f, 0.157f)
              )),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the forget gate weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.forgetGate.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f),
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f),
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f),
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f),
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f)
              )),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the candidate recurrent weights") {
          assertTrue {
            paramsErrors.getErrorsOf(params.candidate.recurrentWeights)!!.values.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f)
              )),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the input gate recurrent weights") {
          assertTrue {
            paramsErrors.getErrorsOf(params.inputGate.recurrentWeights)!!.values.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f)
              )),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the output gate recurrent weights") {
          assertTrue {
            paramsErrors.getErrorsOf(params.outputGate.recurrentWeights)!!.values.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f)
              )),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the forget gate recurrent weights") {
          assertTrue {
            paramsErrors.getErrorsOf(params.forgetGate.recurrentWeights)!!.values.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f)
              )),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the inputArray") {
          assertTrue {
            layer.inputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.023f, 0.106f, -0.06f, -0.003f)),
              tolerance = 0.0005f)
          }
        }
      }

      context("with previous and next state") {

        val layer = LSTMLayerStructureUtils.buildLayer(LSTMLayersWindow.Bilateral)

        layer.forward()

        val errors = MSECalculator().calculateErrors(
          output = layer.outputArray.values,
          outputGold = LSTMLayerStructureUtils.getOutputGold())

        layer.outputArray.assignErrors(errors)
        val paramsErrors = layer.backward(propagateToInput = true)

        val params = layer.params

        it("should match the expected errors of the outputArray") {
          assertTrue {
            layer.outputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.406f, -0.851f, 1.485f, -1.826f, -2.309f)),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the cell") {
          assertTrue {
            layer.cell.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.363f, -0.462f, 0.965f, -0.277f, -0.989f)),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the candidate") {
          assertTrue {
            layer.candidate.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.249f, -0.103f, 0.053f, -0.004f, -0.074f)),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the input gate") {
          assertTrue {
            layer.inputGate.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.016f, -0.028f, -0.227f, -0.04f, 0.228f)),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the output gate") {
          assertTrue {
            layer.outputGate.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.017f, 0.035f, 0.021f, -0.288f, 0.225f)),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the forget gate") {
          assertTrue {
            layer.forgetGate.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.024f, 0.065f, 0.13f, -0.002f, -0.023f)),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the candidate biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.candidate.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.249f, -0.103f, 0.053f, -0.004f, -0.074f)),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the input gate biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.inputGate.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.016f, -0.028f, -0.227f, -0.04f, 0.228f)),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the output gate biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.outputGate.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.017f, 0.035f, 0.021f, -0.288f, 0.225f)),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the forget gate biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.forgetGate.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.024f, 0.065f, 0.13f, -0.002f, -0.023f)),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the candidate weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.candidate.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(-0.199f, -0.224f, -0.224f, 0.249f),
                floatArrayOf(0.082f, 0.093f, 0.093f, -0.103f),
                floatArrayOf(-0.042f, -0.048f, -0.048f, 0.053f),
                floatArrayOf(0.004f, 0.004f, 0.004f, -0.004f),
                floatArrayOf(0.059f, 0.067f, 0.067f, -0.074f)
              )),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the input gate weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.inputGate.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(0.013f, 0.015f, 0.015f, -0.016f),
                floatArrayOf(0.023f, 0.026f, 0.026f, -0.028f),
                floatArrayOf(0.181f, 0.204f, 0.204f, -0.227f),
                floatArrayOf(0.032f, 0.036f, 0.036f, -0.04f),
                floatArrayOf(-0.182f, -0.205f, -0.205f, 0.228f)
              )),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the output gate weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.outputGate.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(-0.013f, -0.015f, -0.015f, 0.017f),
                floatArrayOf(-0.028f, -0.032f, -0.032f, 0.035f),
                floatArrayOf(-0.017f, -0.019f, -0.019f, 0.021f),
                floatArrayOf(0.23f, 0.259f, 0.259f, -0.288f),
                floatArrayOf(-0.18f, -0.202f, -0.202f, 0.225f)
              )),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the forget gate weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.forgetGate.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(-0.019f, -0.021f, -0.021f, 0.024f),
                floatArrayOf(-0.052f, -0.059f, -0.059f, 0.065f),
                floatArrayOf(-0.104f, -0.117f, -0.117f, 0.13f),
                floatArrayOf(0.002f, 0.002f, 0.002f, -0.002f),
                floatArrayOf(0.019f, 0.021f, 0.021f, -0.023f)
              )),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the candidate recurrent weights") {
          assertTrue {
            paramsErrors.getErrorsOf(params.candidate.recurrentWeights)!!.values.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(-0.05f, 0.05f, -0.075f, -0.224f, -0.199f),
                floatArrayOf(0.021f, -0.021f, 0.031f, 0.093f, 0.082f),
                floatArrayOf(-0.011f, 0.011f, -0.016f, -0.048f, -0.042f),
                floatArrayOf(0.001f, -0.001f, 0.001f, 0.004f, 0.004f),
                floatArrayOf(0.015f, -0.015f, 0.022f, 0.067f, 0.059f)
              )),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the input gate recurrent weights") {
          assertTrue {
            paramsErrors.getErrorsOf(params.inputGate.recurrentWeights)!!.values.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(0.003f, -0.003f, 0.005f, 0.015f, 0.013f),
                floatArrayOf(0.006f, -0.006f, 0.009f, 0.026f, 0.023f),
                floatArrayOf(0.045f, -0.045f, 0.068f, 0.204f, 0.181f),
                floatArrayOf(0.008f, -0.008f, 0.012f, 0.036f, 0.032f),
                floatArrayOf(-0.046f, 0.046f, -0.068f, -0.205f, -0.182f)
              )),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the output gate recurrent weights") {
          assertTrue {
            paramsErrors.getErrorsOf(params.outputGate.recurrentWeights)!!.values.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(-0.003f, 0.003f, -0.005f, -0.015f, -0.013f),
                floatArrayOf(-0.007f, 0.007f, -0.011f, -0.032f, -0.028f),
                floatArrayOf(-0.004f, 0.004f, -0.006f, -0.019f, -0.017f),
                floatArrayOf(0.058f, -0.058f, 0.086f, 0.259f, 0.23f),
                floatArrayOf(-0.045f, 0.045f, -0.067f, -0.202f, -0.18f)
              )),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the forget gate recurrent weights") {
          assertTrue {
            paramsErrors.getErrorsOf(params.forgetGate.recurrentWeights)!!.values.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(-0.005f, 0.005f, -0.007f, -0.021f, -0.019f),
                floatArrayOf(-0.013f, 0.013f, -0.02f, -0.059f, -0.052f),
                floatArrayOf(-0.026f, 0.026f, -0.039f, -0.117f, -0.104f),
                floatArrayOf(0.0f, 0.0f, 0.001f, 0.002f, 0.002f),
                floatArrayOf(0.005f, -0.005f, 0.007f, 0.021f, 0.019f)
              )),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the inputArray") {
          assertTrue {
            layer.inputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.042f, 0.388f, -0.243f, 0.181f)),
              tolerance = 0.0005f)
          }
        }
      }
    }
  }
})
