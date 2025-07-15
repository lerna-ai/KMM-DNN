/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.recurrent.cfn

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
class CFNLayerStructureSpec : Spek({

  describe("a CFNLayer") {

    context("forward") {

      context("without previous state context") {

        val layer = CFNLayerStructureUtils.buildLayer(CFNLayersWindow.Empty)
        layer.forward()

        it("should match the expected input gate") {
          assertTrue {
            layer.inputGate.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.397f, 0.252f, 0.5f, 0.705f, 0.453f)),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected forget gate") {
          assertTrue {
            layer.forgetGate.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.853f, 0.433f, 0.116f, 0.52f, 0.242f)),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected candidate") {
          assertTrue {
            layer.candidate.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.675f, -0.1f, 0.762f, 0.869f, -0.804f)),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected outputArray") {
          assertTrue {
            layer.outputArray.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.268f, -0.025f, 0.381f, 0.613f, -0.364f)),
              tolerance = 0.0005f)
          }
        }
      }

      context("with previous state context") {

        val layer = CFNLayerStructureUtils.buildLayer(CFNLayersWindow.Back)
        layer.forward()

        it("should match the expected input gate") {
          assertTrue {
            layer.inputGate.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.723f, 0.25f, 0.55f, 0.821f, 0.535f)),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected forget gate") {
          assertTrue {
            layer.forgetGate.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.911f, 0.181f, 0.048f, 0.675f, 0.389f)),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected candidate") {
          assertTrue {
            layer.candidate.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.675f, -0.1f, 0.762f, 0.869f, -0.804f)),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected outputArray") {
          assertTrue {
            layer.outputArray.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.308f, 0.011f, 0.405f, 0.230f, -0.689f)),
              tolerance = 0.0005f)
          }
        }
      }
    }

    context("backward") {

      context("without previous and next state") {

        val layer = CFNLayerStructureUtils.buildLayer(CFNLayersWindow.Empty)

        layer.forward()

        val errors = MSECalculator().calculateErrors(
          output = layer.outputArray.values,
          outputGold = CFNLayerStructureUtils.getOutputGold())

        layer.outputArray.assignErrors(errors)
        val paramsErrors = layer.backward(propagateToInput = true)

        val params = layer.params

        it("should match the expected errors of the outputArray") {
          assertTrue {
            layer.outputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.302f, -0.775f, 0.531f, -1.027f, -0.814f)),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the input gate") {
          assertTrue {
            layer.inputGate.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.049f, 0.015f, 0.101f, -0.186f, 0.162f)),
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

        it("should match the expected errors of the candidate") {
          assertTrue {
            layer.candidate.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.065f, -0.193f, 0.111f, -0.177f, -0.13f)),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the input gate biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.inputGate.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.049f, 0.015f, 0.101f, -0.186f, 0.162f)),
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

        it("should match the expected errors of the input gate weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.inputGate.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(0.039f, 0.044f, 0.044f, -0.049f),
                floatArrayOf(-0.012f, -0.013f, -0.013f, 0.015f),
                floatArrayOf(-0.081f, -0.091f, -0.091f, 0.101f),
                floatArrayOf(0.149f, 0.167f, 0.167f, -0.186f),
                floatArrayOf(-0.13f, -0.146f, -0.146f, 0.162f)
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

        it("should match the expected errors of the candidate weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.candidateWeights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(0.052f, 0.059f, 0.059f, -0.065f),
                floatArrayOf(0.154f, 0.174f, 0.174f, -0.193f),
                floatArrayOf(-0.089f, -0.1f, -0.1f, 0.111f),
                floatArrayOf(0.142f, 0.159f, 0.159f, -0.177f),
                floatArrayOf(0.104f, 0.117f, 0.117f, -0.130f)
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
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.318f, 0.01f, -0.027f, 0.302f)),
              tolerance = 0.0005f)
          }
        }
      }

      context("with previous state only") {

        val layer = CFNLayerStructureUtils.buildLayer(CFNLayersWindow.Back)

        layer.forward()

        val errors = MSECalculator().calculateErrors(
          output = layer.outputArray.values,
          outputGold = CFNLayerStructureUtils.getOutputGold())

        layer.outputArray.assignErrors(errors)
        val paramsErrors = layer.backward(propagateToInput = true)

        val params = layer.params

        it("should match the expected errors of the outputArray") {
          assertTrue {
            layer.outputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.262f, -0.739f, 0.555f, -1.41f, -1.139f)),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the input gate") {
          assertTrue {
            layer.inputGate.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.035f, 0.014f, 0.105f, -0.18f, 0.228f)),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the forget gate") {
          assertTrue {
            layer.forgetGate.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.004f, -0.022f, -0.007f, 0.222f, 0.18f)),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the candidate") {
          assertTrue {
            layer.candidate.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.103f, -0.183f, 0.128f, -0.283f, -0.215f)),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the input gate biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.inputGate.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.035f, 0.014f, 0.105f, -0.18f, 0.228f)),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the forget gate biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.forgetGate.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.004f, -0.022f, -0.007f, 0.222f, 0.18f)),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the input gate weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.inputGate.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(0.028f, 0.032f, 0.032f, -0.035f),
                floatArrayOf(-0.011f, -0.012f, -0.012f, 0.014f),
                floatArrayOf(-0.084f, -0.094f, -0.094f, 0.105f),
                floatArrayOf(0.144f, 0.162f, 0.162f, -0.18f),
                floatArrayOf(-0.182f, -0.205f, -0.205f, 0.228f)
              )),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the forget gate weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.forgetGate.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(-0.003f, -0.004f, -0.004f, 0.004f),
                floatArrayOf(0.017f, 0.019f, 0.019f, -0.022f),
                floatArrayOf(0.006f, 0.007f, 0.007f, -0.007f),
                floatArrayOf(-0.177f, -0.199f, -0.199f, 0.222f),
                floatArrayOf(-0.144f, -0.162f, -0.162f, 0.18f)
              )),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the candidate weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.candidateWeights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(0.082f, 0.093f, 0.093f, -0.103f),
                floatArrayOf(0.146f, 0.164f, 0.164f, -0.183f),
                floatArrayOf(-0.102f, -0.115f, -0.115f, 0.128f),
                floatArrayOf(0.226f, 0.255f, 0.255f, -0.283f),
                floatArrayOf(0.172f, 0.194f, 0.194f, -0.215f)
              )),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the input gate recurrent weights") {
          assertTrue {
            paramsErrors.getErrorsOf(params.inputGate.recurrentWeights)!!.values.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(0.007f, -0.007f, 0.011f, 0.032f, 0.028f),
                floatArrayOf(-0.003f, 0.003f, -0.004f, -0.012f, -0.011f),
                floatArrayOf(-0.021f, 0.021f, -0.031f, -0.094f, -0.084f),
                floatArrayOf(0.036f, -0.036f, 0.054f, 0.162f, 0.144f),
                floatArrayOf(-0.046f, 0.046f, -0.068f, -0.205f, -0.182f)
              )),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the forget gate recurrent weights") {
          assertTrue {
            paramsErrors.getErrorsOf(params.forgetGate.recurrentWeights)!!.values.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(-0.001f, 0.001f, -0.001f, -0.004f, -0.003f),
                floatArrayOf(0.004f, -0.004f, 0.006f, 0.019f, 0.017f),
                floatArrayOf(0.001f, -0.001f, 0.002f, 0.007f, 0.006f),
                floatArrayOf(-0.044f, 0.044f, -0.066f, -0.199f, -0.177f),
                floatArrayOf(-0.036f, 0.036f, -0.054f, -0.162f, -0.144f)
              )),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the inputArray") {
          assertTrue {
            layer.inputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.111f, 0.37f, -0.281f, 0.126f)),
              tolerance = 0.0005f)
          }
        }
      }

      context("with next state only") {

        val layer = CFNLayerStructureUtils.buildLayer(CFNLayersWindow.Front(
          currentLayerOutput = DenseNDArrayFactory.arrayOf(floatArrayOf(0.261f, -0.025f, 0.363f, 0.546f, -0.349f))))

        layer.forward()

        val errors = MSECalculator().calculateErrors(
          output = layer.outputArray.values,
          outputGold = CFNLayerStructureUtils.getOutputGold())

        layer.outputArray.assignErrors(errors)
        val paramsErrors = layer.backward(propagateToInput = true)

        val params = layer.params

        it("should match the expected errors of the outputArray") {
          assertTrue {
            layer.outputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.451f, -0.425f, 0.97f, -1.710f, -1.016f)),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the input gate") {
          assertTrue {
            layer.inputGate.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.073f, 0.008f, 0.185f, -0.309f, 0.202f)),
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

        it("should match the expected errors of the candidate") {
          assertTrue {
            layer.candidate.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.097f, -0.106f, 0.204f, -0.295f, -0.163f)),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the input gate biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.inputGate.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.073f, 0.008f, 0.185f, -0.309f, 0.202f)),
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

        it("should match the expected errors of the input gate weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.inputGate.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(0.058f, 0.066f, 0.066f, -0.073f),
                floatArrayOf(-0.006f, -0.007f, -0.007f, 0.008f),
                floatArrayOf(-0.148f, -0.166f, -0.166f, 0.185f),
                floatArrayOf(0.2479f, 0.278f, 0.278f, -0.309f),
                floatArrayOf(-0.162f, -0.182f, -0.182f, 0.202f)
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

        it("should match the expected errors of the candidate weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.candidateWeights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(0.078f, 0.088f, 0.088f, -0.097f),
                floatArrayOf(0.085f, 0.095f, 0.095f, -0.106f),
                floatArrayOf(-0.163f, -0.183f, -0.183f, 0.204f),
                floatArrayOf(0.236f, 0.265f, 0.265f, -0.295f),
                floatArrayOf(0.130f, 0.146f, 0.146f, -0.163f)
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
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.378f, 0.135f, -0.114f, 0.372f)),
              tolerance = 0.0005f)
          }
        }
      }

      context("with previous and next state") {

        val layer = CFNLayerStructureUtils.buildLayer(CFNLayersWindow.Bilateral(
          currentLayerOutput = DenseNDArrayFactory.arrayOf(floatArrayOf(0.299f, 0.0108f, 0.384f, 0.226f, -0.597f))))

        layer.forward()

        val errors = MSECalculator().calculateErrors(
          output = layer.outputArray.values,
          outputGold = CFNLayerStructureUtils.getOutputGold())

        layer.outputArray.assignErrors(errors)
        val paramsErrors = layer.backward(propagateToInput = true)

        val params = layer.params

        it("should match the expected errors of the outputArray") {
          assertTrue {
            layer.outputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.41f, -0.389f, 0.999f, -2.232f, -1.364f)),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the input gate") {
          assertTrue {
            layer.inputGate.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.055f, 0.007f, 0.188f, -0.286f, 0.273f)),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the forget gate") {
          assertTrue {
            layer.forgetGate.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.007f, -0.011f, -0.013f, 0.351f, 0.215f)),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the candidate") {
          assertTrue {
            layer.candidate.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.161f, -0.096f, 0.231f, -0.448f, -0.258f)),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the input gate biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.inputGate.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.055f, 0.007f, 0.188f, -0.286f, 0.273f)),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the forget gate biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.forgetGate.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.007f, -0.011f, -0.013f, 0.351f, 0.215f)),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the input gate weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.inputGate.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(0.044f, 0.05f, 0.05f, -0.055f),
                floatArrayOf(-0.006f, -0.007f, -0.007f, 0.007f),
                floatArrayOf(-0.151f, -0.169f, -0.169f, 0.188f),
                floatArrayOf(0.229f, 0.257f, 0.257f, -0.286f),
                floatArrayOf(-0.218f, -0.246f, -0.246f, 0.273f)
              )),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the forget gate weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.forgetGate.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(-0.005f, -0.006f, -0.006f, 0.007f),
                floatArrayOf(0.009f, 0.01f, 0.01f, -0.011f),
                floatArrayOf(0.011f, 0.012f, 0.012f, -0.013f),
                floatArrayOf(-0.281f, -0.316f, -0.316f, 0.351f),
                floatArrayOf(-0.172f, -0.194f, -0.194f, 0.215f)
              )),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the candidate weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.candidateWeights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(0.129f, 0.145f, 0.145f, -0.161f),
                floatArrayOf(0.077f, 0.087f, 0.087f, -0.096f),
                floatArrayOf(-0.185f, -0.208f, -0.208f, 0.231f),
                floatArrayOf(0.358f, 0.403f, 0.403f, -0.448f),
                floatArrayOf(0.206f, 0.232f, 0.232f, -0.258f)
              )),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the input gate recurrent weights") {
          assertTrue {
            paramsErrors.getErrorsOf(params.inputGate.recurrentWeights)!!.values.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(0.011f, -0.011f, 0.017f, 0.05f, 0.044f),
                floatArrayOf(-0.001f, 0.001f, -0.002f, -0.007f, -0.006f),
                floatArrayOf(-0.038f, 0.038f, -0.056f, -0.169f, -0.151f),
                floatArrayOf(0.057f, -0.057f, 0.086f, 0.257f, 0.229f),
                floatArrayOf(-0.055f, 0.055f, -0.082f, -0.246f, -0.218f)
              )),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the forget gate recurrent weights") {
          assertTrue {
            paramsErrors.getErrorsOf(params.forgetGate.recurrentWeights)!!.values.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(-0.001f, 0.001f, -0.002f, -0.006f, -0.005f),
                floatArrayOf(0.002f, -0.002f, 0.003f, 0.01f, 0.009f),
                floatArrayOf(0.003f, -0.003f, 0.004f, 0.012f, 0.011f),
                floatArrayOf(-0.07f, 0.07f, -0.105f, -0.316f, -0.281f),
                floatArrayOf(-0.043f, 0.043f, -0.065f, -0.194f, -0.172f)
              )),
              tolerance = 0.0005f)
          }
        }

        it("should match the expected errors of the inputArray") {
          assertTrue {
            layer.inputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.123f, 0.625f, -0.467f, 0.104f)),
              tolerance = 0.0005f)
          }
        }
      }
    }
  }
})
