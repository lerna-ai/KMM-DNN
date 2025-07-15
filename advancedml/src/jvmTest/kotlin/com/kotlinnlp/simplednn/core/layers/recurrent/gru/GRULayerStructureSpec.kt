/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.recurrent.gru

import com.kotlinnlp.simplednn.core.functionalities.losses.MSECalculator
import com.kotlinnlp.simplednn.core.optimizer.getErrorsOf
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 *
 */
class GRULayerStructureSpec : Spek({

  describe("a GRULayer") {

    context("forward") {

      context("without previous state context") {

        val layer = GRULayerStructureUtils.buildLayer(GRULayersWindow.Empty)
        layer.forward()

        it("should match the expected reset gate") {
          assertTrue {
            layer.resetGate.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.40f, 0.25f, 0.50f, 0.70f, 0.45f)),
              tolerance = 0.005f)
          }
        }

        it("should match the expected partition gate") {
          assertTrue {
            layer.partitionGate.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.85f, 0.43f, 0.12f, 0.52f, 0.24f)),
              tolerance = 0.005f)
          }
        }

        it("should match the expected candidate") {
          assertTrue {
            layer.candidate.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.87f, -0.54f, 0.96f, 0.94f, -0.21f)),
              tolerance = 0.005f)
          }
        }

        it("should match the expected outputArray") {
          assertTrue {
            layer.outputArray.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.74f, -0.23f, 0.11f, 0.49f, -0.05f)),
              tolerance = 0.005f)
          }
        }
      }

      context("with previous state context") {

        val layer = GRULayerStructureUtils.buildLayer(GRULayersWindow.Back)
        layer.forward()

        it("should match the expected reset gate") {
          assertTrue {
            layer.resetGate.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.72f, 0.25f, 0.55f, 0.82f, 0.53f)),
              tolerance = 0.005f)
          }
        }

        it("should match the expected partition gate") {
          assertTrue {
            layer.partitionGate.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.91f, 0.18f, 0.05f, 0.67f, 0.39f)),
              tolerance = 0.005f)
          }
        }

        it("should match the expected candidate") {
          assertTrue {
            layer.candidate.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.96f, 0.07f, 0.92f, 0.97f, 0.33f)),
              tolerance = 0.005f)
          }
        }

        it("should match the expected outputArray") {
          assertTrue {
            layer.outputArray.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.86f, 0.18f, -0.24f, 0.36f, -0.36f)),
              tolerance = 0.005f)
          }
        }
      }
    }

    context("backward") {

      context("without previous and next state") {

        val layer = GRULayerStructureUtils.buildLayer(GRULayersWindow.Empty)

        layer.forward()

        val errors = MSECalculator().calculateErrors(
          output = layer.outputArray.values,
          outputGold = GRULayerStructureUtils.getOutputGold())

        layer.outputArray.assignErrors(errors)
        val paramsErrors = layer.backward(propagateToInput = true)

        val params = layer.params

        it("should match the expected errors of the outputArray") {
          assertTrue {
            layer.outputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.17f, -0.98f, 0.26f, -1.15f, -0.5f)),
              tolerance = 0.005f)
          }
        }

        it("should match the expected errors of the reset gate") {
          assertTrue {
            layer.resetGate.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f)),
              tolerance = 0.005f)
          }
        }

        it("should match the expected errors of the partition gate") {
          assertTrue {
            layer.partitionGate.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.02f, 0.13f, 0.03f, -0.27f, 0.02f)),
              tolerance = 0.005f)
          }
        }

        it("should match the expected errors of the candidate") {
          assertTrue {
            layer.candidate.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.04f, -0.3f, 0.0f, -0.07f, -0.12f)),
              tolerance = 0.005f)
          }
        }

        it("should match the expected errors of the reset gate biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.resetGate.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f)),
              tolerance = 0.005f)
          }
        }

        it("should match the expected errors of the partition gate biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.partitionGate.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.02f, 0.13f, 0.03f, -0.27f, 0.02f)),
              tolerance = 0.005f)
          }
        }

        it("should match the expected errors of the candidate biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.candidate.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.04f, -0.3f, 0.0f, -0.07f, -0.12f)),
              tolerance = 0.005f)
          }
        }

        it("should match the expected errors of the reset gate weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.resetGate.weights)!!.values as DenseNDArray).equals(
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

        it("should match the expected errors of the partition gate weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.partitionGate.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(-0.01f, -0.02f, -0.02f, 0.02f),
                floatArrayOf(-0.10f, -0.12f, -0.12f, 0.13f),
                floatArrayOf(-0.02f, -0.02f, -0.02f, 0.03f),
                floatArrayOf(0.22f, 0.24f, 0.24f, -0.27f),
                floatArrayOf(-0.02f, -0.02f, -0.02f, 0.02f)
              )),
              tolerance = 0.005f)
          }
        }

        it("should match the expected errors of the candidate weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.candidate.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(-0.03f, -0.03f, -0.03f, 0.04f),
                floatArrayOf(0.24f, 0.27f, 0.27f, -0.30f),
                floatArrayOf(0.00f, 0.00f, 0.00f, 0.00f),
                floatArrayOf(0.06f, 0.06f, 0.06f, -0.07f),
                floatArrayOf(0.09f, 0.10f, 0.10f, -0.12f)
              )),
              tolerance = 0.005f)
          }
        }

        it("should match the expected errors of the reset gate recurrent weights") {
          assertTrue {
            paramsErrors.getErrorsOf(params.resetGate.recurrentWeights)!!.values.equals(
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

        it("should match the expected errors of the partition gate recurrent weights") {
          assertTrue {
            paramsErrors.getErrorsOf(params.partitionGate.recurrentWeights)!!.values.equals(
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

        it("should match the expected errors of the candidate recurrent weights") {

          assertNull(paramsErrors.getErrorsOf(params.candidate.recurrentWeights))
        }

        it("should match the expected errors of the inputArray") {
          assertTrue {
            layer.inputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.53f, -0.49f, 0.18f, 0.20f)),
              tolerance = 0.005f)
          }
        }
      }

      context("with previous state only") {

        val layer = GRULayerStructureUtils.buildLayer(GRULayersWindow.Back)

        layer.forward()

        val errors = MSECalculator().calculateErrors(
          output = layer.outputArray.values,
          outputGold = GRULayerStructureUtils.getOutputGold())

        layer.outputArray.assignErrors(errors)
        val paramsErrors = layer.backward(propagateToInput = true)

        val params = layer.params

        it("should match the expected errors of the outputArray") {
          assertTrue {
            layer.outputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.29f, -0.57f, -0.09f, -1.28f, -0.81f)),
              tolerance = 0.005f)
          }
        }

        it("should match the expected errors of the reset gate") {
          assertTrue {
            layer.resetGate.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.01f, 0.0f, -0.02f, -0.03f, -0.01f)),
              tolerance = 0.005f)
          }
        }

        it("should match the expected errors of the partition gate") {
          assertTrue {
            layer.partitionGate.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.03f, 0.01f, -0.01f, -0.52f, -0.22f)),
              tolerance = 0.005f)
          }
        }

        it("should match the expected errors of the candidate") {
          assertTrue {
            layer.candidate.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.02f, -0.10f, 0.00f, -0.06f, -0.28f)),
              tolerance = 0.005f)
          }
        }

        it("should match the expected errors of the reset gate biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.resetGate.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.01f, 0.0f, -0.02f, -0.03f, -0.01f)),
              tolerance = 0.005f)
          }
        }

        it("should match the expected errors of the partition gate biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.partitionGate.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.03f, 0.01f, -0.01f, -0.52f, -0.22f)),
              tolerance = 0.005f)
          }
        }

        it("should match the expected errors of the candidate biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.candidate.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.02f, -0.10f, 0.00f, -0.06f, -0.28f)),
              tolerance = 0.005f)
          }
        }

        it("should match the expected errors of the reset gate weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.resetGate.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(0.00f, 0.01f, 0.01f, -0.01f),
                floatArrayOf(0.00f, 0.00f, 0.00f, 0.00f),
                floatArrayOf(0.01f, 0.01f, 0.01f, -0.02f),
                floatArrayOf(0.02f, 0.02f, 0.02f, -0.03f),
                floatArrayOf(0.01f, 0.01f, 0.01f, -0.01f)
              )),
              tolerance = 0.005f)
          }
        }

        it("should match the expected errors of the partition gate weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.partitionGate.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(-0.02f, -0.02f, -0.02f, 0.03f),
                floatArrayOf(-0.01f, -0.01f, -0.01f, 0.01f),
                floatArrayOf(0.0f, 0.0f, 0.0f, -0.01f),
                floatArrayOf(0.42f, 0.47f, 0.47f, -0.52f),
                floatArrayOf(0.17f, 0.2f, 0.2f, -0.22f)
              )),
              tolerance = 0.005f)
          }
        }

        it("should match the expected errors of the candidate weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.candidate.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(-0.02f, -0.02f, -0.02f, 0.02f),
                floatArrayOf(0.08f, 0.09f, 0.09f, -0.10f),
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f),
                floatArrayOf(0.05f, 0.05f, 0.05f, -0.06f),
                floatArrayOf(0.22f, 0.25f, 0.25f, -0.28f)
              )),
              tolerance = 0.005f)
          }
        }

        it("should match the expected errors of the reset gate recurrent weights") {
          assertTrue {
            paramsErrors.getErrorsOf(params.resetGate.recurrentWeights)!!.values.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.01f, 0.0f),
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.01f, 0.01f),
                floatArrayOf(0.01f, -0.01f, 0.01f, 0.02f, 0.02f),
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.01f, 0.01f)
              )),
              tolerance = 0.005f)
          }
        }

        it("should match the expected errors of the partition gate recurrent weights") {
          assertTrue {
            paramsErrors.getErrorsOf(params.partitionGate.recurrentWeights)!!.values.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(-0.01f, 0.01f, -0.01f, -0.02f, -0.02f),
                floatArrayOf(0.0f, 0.0f, 0.0f, -0.01f, -0.01f),
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
                floatArrayOf(0.1f, -0.1f, 0.16f, 0.47f, 0.42f),
                floatArrayOf(0.04f, -0.04f, 0.07f, 0.2f, 0.17f)
              )),
              tolerance = 0.005f)
          }
        }

        it("should match the expected errors of the candidate recurrent weights") {
          assertTrue {
            paramsErrors.getErrorsOf(params.candidate.recurrentWeights)!!.values.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(0.0f, 0.0f, 0.0f, -0.01f, -0.01f),
                floatArrayOf(0.01f, -0.01f, 0.02f, 0.08f, 0.04f),
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
                floatArrayOf(0.01f, 0.0f, 0.01f, 0.04f, 0.02f),
                floatArrayOf(0.04f, -0.01f, 0.05f, 0.21f, 0.12f)
              )),
              tolerance = 0.005f)
          }
        }

        it("should match the expected errors of the inputArray") {
          assertTrue {
            layer.inputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.56f, -0.83f, 0.50f, 0.55f)),
              tolerance = 0.005f)
          }
        }
      }

      context("with next state only") {

        val layer = GRULayerStructureUtils.buildLayer(GRULayersWindow.Front)

        layer.forward()

        val errors = MSECalculator().calculateErrors(
          output = layer.outputArray.values,
          outputGold = GRULayerStructureUtils.getOutputGold())

        layer.outputArray.assignErrors(errors)
        val paramsErrors = layer.backward(propagateToInput = true)

        val params = layer.params

        it("should match the expected errors of the outputArray") {
          assertTrue {
            layer.outputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.05f, -0.24f, 0.94f, -0.18f, -0.71f)),
              tolerance = 0.005f)
          }
        }

        it("should match the expected errors of the reset gate") {
          assertTrue {
            layer.resetGate.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f)),
              tolerance = 0.005f)
          }
        }

        it("should match the expected errors of the partition gate") {
          assertTrue {
            layer.partitionGate.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.01f, 0.03f, 0.09f, -0.04f, 0.03f)),
              tolerance = 0.005f)
          }
        }

        it("should match the expected errors of the candidate") {
          assertTrue {
            layer.candidate.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.01f, -0.07f, 0.01f, -0.01f, -0.17f)),
              tolerance = 0.005f)
          }
        }

        it("should match the expected errors of the reset gate biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.resetGate.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f)),
              tolerance = 0.005f)
          }
        }

        it("should match the expected errors of the partition gate biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.partitionGate.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.01f, 0.03f, 0.09f, -0.04f, 0.03f)),
              tolerance = 0.005f)
          }
        }

        it("should match the expected errors of the candidate biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.candidate.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.01f, -0.07f, 0.01f, -0.01f, -0.17f)),
              tolerance = 0.005f)
          }
        }

        it("should match the expected errors of the reset gate weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.resetGate.weights)!!.values as DenseNDArray).equals(
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

        it("should match the expected errors of the partition gate weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.partitionGate.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(0.00f, 0.00f, 0.00f, 0.01f),
                floatArrayOf(-0.03f, -0.03f, -0.03f, 0.03f),
                floatArrayOf(-0.07f, -0.08f, -0.08f, 0.09f),
                floatArrayOf(0.03f, 0.04f, 0.04f, -0.04f),
                floatArrayOf(-0.02f, -0.02f, -0.02f, 0.03f)
              )),
              tolerance = 0.005f)
          }
        }

        it("should match the expected errors of the candidate weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.candidate.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(-0.01f, -0.01f, -0.01f, 0.01f),
                floatArrayOf(0.06f, 0.07f, 0.07f, -0.07f),
                floatArrayOf(-0.01f, -0.01f, -0.01f, 0.01f),
                floatArrayOf(0.01f, 0.01f, 0.01f, -0.01f),
                floatArrayOf(0.13f, 0.15f, 0.15f, -0.17f)
              )),
              tolerance = 0.005f)
          }
        }

        it("should match the expected errors of the reset gate recurrent weights") {
          assertTrue {
            paramsErrors.getErrorsOf(params.resetGate.recurrentWeights)!!.values.equals(
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

        it("should match the expected errors of the partition gate recurrent weights") {
          assertTrue {
            paramsErrors.getErrorsOf(params.partitionGate.recurrentWeights)!!.values.equals(
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

        it("should match the expected errors of the candidate recurrent weights") {
          assertNull(paramsErrors.getErrorsOf(params.candidate.recurrentWeights))
        }

        it("should match the expected errors of the inputArray") {
          assertTrue {
            layer.inputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.08f, -0.13f, 0.11f, 0.12f)),
              tolerance = 0.005f)
          }
        }
      }

      context("with previous and next state") {

        val layer = GRULayerStructureUtils.buildLayer(GRULayersWindow.Bilateral)

        layer.forward()

        val errors = MSECalculator().calculateErrors(
          output = layer.outputArray.values,
          outputGold = GRULayerStructureUtils.getOutputGold())

        layer.outputArray.assignErrors(errors)
        val paramsErrors = layer.backward(propagateToInput = true)

        val params = layer.params
        it("should match the expected errors of the outputArray") {
          assertTrue {
            layer.outputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.17f, 0.17f, 0.58f, -0.31f, -1.02f)),
              tolerance = 0.005f)
          }
        }

        it("should match the expected errors of the reset gate") {
          assertTrue {
            layer.resetGate.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.01f, 0.0f, -0.01f, -0.03f, 0.01f)),
              tolerance = 0.005f)
          }
        }

        it("should match the expected errors of the partition gate") {
          assertTrue {
            layer.partitionGate.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.02f, 0.0f, 0.03f, -0.13f, -0.28f)),
              tolerance = 0.005f)
          }
        }

        it("should match the expected errors of the candidate") {
          assertTrue {
            layer.candidate.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.01f, 0.03f, 0.0f, -0.01f, -0.35f)),
              tolerance = 0.005f)
          }
        }

        it("should match the expected errors of the reset gate biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.resetGate.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.01f, 0.0f, -0.01f, -0.03f, 0.01f)),
              tolerance = 0.005f)
          }
        }

        it("should match the expected errors of the partition gate biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.partitionGate.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.02f, 0.0f, 0.03f, -0.13f, -0.28f)),
              tolerance = 0.005f)
          }
        }

        it("should match the expected errors of the candidate biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.candidate.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.01f, 0.03f, 0.0f, -0.01f, -0.35f)),
              tolerance = 0.005f)
          }
        }

        it("should match the expected errors of the reset gate weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.resetGate.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(0.01f, 0.01f, 0.01f, -0.01f),
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f),
                floatArrayOf(0.01f, 0.01f, 0.01f, -0.01f),
                floatArrayOf(0.02f, 0.02f, 0.02f, -0.03f),
                floatArrayOf(-0.01f, -0.01f, -0.01f, 0.01f)
              )),
              tolerance = 0.005f)
          }
        }

        it("should match the expected errors of the partition gate weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.partitionGate.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(-0.01f, -0.01f, -0.01f, 0.02f),
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f),
                floatArrayOf(-0.03f, -0.03f, -0.03f, 0.03f),
                floatArrayOf(0.10f, 0.11f, 0.11f, -0.13f),
                floatArrayOf(0.22f, 0.25f, 0.25f, -0.28f)
              )),
              tolerance = 0.005f)
          }
        }

        it("should match the expected errors of the candidate weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.candidate.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(-0.01f, -0.01f, -0.01f, 0.01f),
                floatArrayOf(-0.02f, -0.03f, -0.03f, 0.03f),
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f),
                floatArrayOf(0.01f, 0.01f, 0.01f, -0.01f),
                floatArrayOf(0.28f, 0.32f, 0.32f, -0.35f)
              )),
              tolerance = 0.005f)
          }
        }

        it("should match the expected errors of the reset gate recurrent weights") {
          assertTrue {
            paramsErrors.getErrorsOf(params.resetGate.recurrentWeights)!!.values.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.01f, 0.01f),
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.01f, 0.01f),
                floatArrayOf(0.01f, -0.01f, 0.01f, 0.02f, 0.02f),
                floatArrayOf(0.0f, 0.0f, 0.0f, -0.01f, -0.01f)
              )),
              tolerance = 0.005f)
          }
        }

        it("should match the expected errors of the partition gate recurrent weights") {
          assertTrue {
            paramsErrors.getErrorsOf(params.partitionGate.recurrentWeights)!!.values.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(0.0f, 0.0f, 0.0f, -0.01f, -0.01f),
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
                floatArrayOf(-0.01f, 0.01f, -0.01f, -0.03f, -0.03f),
                floatArrayOf(0.03f, -0.03f, 0.04f, 0.11f, 0.10f),
                floatArrayOf(0.06f, -0.06f, 0.08f, 0.25f, 0.22f)
              )),
              tolerance = 0.005f)
          }
        }

        it("should match the expected errors of the candidate recurrent weights") {
          assertTrue {
            paramsErrors.getErrorsOf(params.candidate.recurrentWeights)!!.values.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(0.0f, 0.0f, 0.0f, -0.01f, 0.0f),
                floatArrayOf(0.0f, 0.0f, 0.0f, -0.02f, -0.01f),
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.01f, 0.01f),
                floatArrayOf(0.05f, -0.02f, 0.06f, 0.26f, 0.15f)
              )),
              tolerance = 0.005f)
          }
        }

        it("should match the expected errors of the inputArray") {
          assertTrue {
            layer.inputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.11f, -0.46f, 0.46f, 0.53f)),
              tolerance = 0.005f)
          }
        }
      }
    }
  }
})
