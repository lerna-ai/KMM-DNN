/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.recurrent.indrnn

import com.kotlinnlp.simplednn.core.optimizer.getErrorsOf
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertTrue

/**
 *
 */
class IndRNNLayerStructureSpec : Spek({

  describe("an IndRNNLayer") {

    context("forward") {

      context("without previous state context") {

        val layer = IndRNNLayerStructureUtils.buildLayer(IndRNNLayersWindow.Empty)
        layer.forward()

        it("should match the expected output") {
          assertTrue(layer.outputArray.values.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(-0.39693f, -0.796878f, 0.0f, 0.701374f, -0.187746f)),
            tolerance = 1.0e-06f))
        }
      }

      context("with previous state context") {

        val layer = IndRNNLayerStructureUtils.buildLayer(IndRNNLayersWindow.Back)

        layer.forward()

        it("should match the expected output") {
          assertTrue(layer.outputArray.values.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(-0.39693f, -0.842046f, 0.256335f, 0.701374f, 0.205456f)),
            tolerance = 1.0e-06f))
        }
      }
    }

    context("backward") {

      context("without next and previous state") {

        val layer = IndRNNLayerStructureUtils.buildLayer(IndRNNLayersWindow.Empty)

        layer.forward()

        layer.outputArray.assignErrors(IndRNNLayerStructureUtils.getOutputErrors())
        val paramsErrors = layer.backward(propagateToInput = true)

        val params = layer.params

        it("should match the expected errors of the output") {
          assertTrue(layer.outputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(0.480194f, 0.273739f, -0.150000f, 0.833242f, 0.434138f)),
            tolerance = 1.0e-06f))
        }

        it("should match the expected errors of the biases") {
          assertTrue(paramsErrors.getErrorsOf(params.feedforwardUnit.biases)!!.values.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(0.480194f, 0.273739f, -0.150000f, 0.833242f, 0.434138f)),
            tolerance = 1.0e-06f))
        }

        it("should match the expected errors of the weights") {
          assertTrue((paramsErrors.getErrorsOf(params.feedforwardUnit.weights)!!.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(listOf(
              floatArrayOf(-0.384155f, -0.432175f, -0.432175f, 0.480194f),
              floatArrayOf(-0.218991f, -0.246365f, -0.246365f, 0.273739f),
              floatArrayOf(0.120000f, 0.135000f, 0.135000f, -0.150000f),
              floatArrayOf(-0.666594f, -0.749918f, -0.749918f, 0.833242f),
              floatArrayOf(-0.347310f, -0.390724f, -0.390724f, 0.434138f)
            )),
            tolerance = 1.0e-06f))
        }

        it("should match the expected errors of the recurrent weights") {
          assertTrue(paramsErrors.getErrorsOf(params.recurrentWeights)!!.values.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f)),
            tolerance = 1.0e-06f))
        }

        it("should match the expected errors of the input") {
          assertTrue(layer.inputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(1.166963f, -0.032159f, -0.705678f, -0.318121f)),
            tolerance = 1.0e-06f))
        }
      }

      context("with prev state only") {

        val layer = IndRNNLayerStructureUtils.buildLayer(IndRNNLayersWindow.Back)

        layer.forward()

        layer.outputArray.assignErrors(IndRNNLayerStructureUtils.getOutputErrors())
        val paramsErrors = layer.backward(propagateToInput = true)

        val params = layer.params

        it("should match the expected errors of the output") {
          assertTrue(layer.outputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(0.480194f, 0.218219f, -0.140144f, 0.833242f, 0.431005f)),
            tolerance = 1.0e-06f))
        }

        it("should match the expected errors of the biases") {
          assertTrue(paramsErrors.getErrorsOf(params.feedforwardUnit.biases)!!.values.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(0.480194f, 0.218219f, -0.140144f, 0.833242f, 0.431005f)),
            tolerance = 1.0e-06f))
        }

        it("should match the expected errors of the weights") {
          assertTrue((paramsErrors.getErrorsOf(params.feedforwardUnit.weights)!!.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(listOf(
              floatArrayOf(-0.384155f, -0.432175f, -0.432175f, 0.480194f),
              floatArrayOf(-0.174576f, -0.196397f, -0.196397f, 0.218219f),
              floatArrayOf(0.112115f, 0.126129f, 0.126129f, -0.140144f),
              floatArrayOf(-0.666594f, -0.749918f, -0.749918f, 0.833242f),
              floatArrayOf(-0.344804f, -0.387904f, -0.387904f, 0.431005f)
            )),
            tolerance = 1.0e-06f))
        }

        it("should match the expected errors of the recurrent weights") {
          assertTrue(paramsErrors.getErrorsOf(params.recurrentWeights)!!.values.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(-0.094779f, 0.043071f, 0.040826f, -0.596849f, -0.286203f)),
            tolerance = 1.0e-06f))
        }

        it("should match the expected errors of the input") {
          assertTrue(layer.inputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(1.133745f, -0.019984f, -0.706080f, -0.271285f)),
            tolerance = 1.0e-06f))
        }
      }

      context("with next state only") {

        val layer = IndRNNLayerStructureUtils.buildLayer(IndRNNLayersWindow.Front)

        layer.forward()

        layer.outputArray.assignErrors(IndRNNLayerStructureUtils.getOutputErrors())
        val paramsErrors = layer.backward(propagateToInput = true)

        val params = layer.params

        it("should match the expected errors of the output") {
          assertTrue(layer.outputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(0.480194f, 0.248190f, 0.300000f, 0.833242f, 0.318368f)),
            tolerance = 1.0e-06f))
        }

        it("should match the expected errors of the biases") {
          assertTrue(paramsErrors.getErrorsOf(params.feedforwardUnit.biases)!!.values.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(0.480194f, 0.248190f, 0.300000f, 0.833242f, 0.318368f)),
            tolerance = 1.0e-06f))
        }

        it("should match the expected errors of the weights") {
          assertTrue((paramsErrors.getErrorsOf(params.feedforwardUnit.weights)!!.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(listOf(
              floatArrayOf(-0.384155f, -0.432175f, -0.432175f, 0.480194f),
              floatArrayOf(-0.198552f, -0.223371f, -0.223371f, 0.248190f),
              floatArrayOf(-0.240000f, -0.270000f, -0.270000f, 0.300000f),
              floatArrayOf(-0.666594f, -0.749918f, -0.749918f, 0.833242f),
              floatArrayOf(-0.254694f, -0.286531f, -0.286531f, 0.318368f)
            )),
            tolerance = 1.0e-06f))
        }

        it("should match the expected errors of the recurrent weights") {
          assertTrue(paramsErrors.getErrorsOf(params.recurrentWeights)!!.values.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f)),
            tolerance = 1.0e-06f))
        }

        it("should match the expected errors of the input") {
          assertTrue(layer.inputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(1.417771f, -0.452709f, -0.492194f, -0.165298f)),
            tolerance = 1.0e-06f))
        }
      }

      context("with next and previous state") {

        val layer = IndRNNLayerStructureUtils.buildLayer(IndRNNLayersWindow.Bilateral)

        layer.forward()

        layer.outputArray.assignErrors(IndRNNLayerStructureUtils.getOutputErrors())
        val paramsErrors = layer.backward(propagateToInput = true)

        val params = layer.params

        it("should match the expected errors of the output") {
          assertTrue(layer.outputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(0.480194f, 0.197852f, 0.280288f, 0.833242f, 0.31607f)),
            tolerance = 1.0e-06f))
        }

        it("should match the expected errors of the biases") {
          assertTrue(paramsErrors.getErrorsOf(params.feedforwardUnit.biases)!!.values.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(0.480194f, 0.197852f, 0.280288f, 0.833242f, 0.31607f)),
            tolerance = 1.0e-06f))
        }

        it("should match the expected errors of the weights") {
          assertTrue((paramsErrors.getErrorsOf(params.feedforwardUnit.weights)!!.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(listOf(
              floatArrayOf(-0.384155f, -0.432175f, -0.432175f, 0.480194f),
              floatArrayOf(-0.158282f, -0.178067f, -0.178067f, 0.197852f),
              floatArrayOf(-0.224230f, -0.252259f, -0.252259f, 0.280288f),
              floatArrayOf(-0.666594f, -0.749918f, -0.749918f, 0.833242f),
              floatArrayOf(-0.252856f, -0.284463f, -0.284463f, 0.316070f)
            )),
            tolerance = 1.0e-06f))
        }

        it("should match the expected errors of the recurrent weights") {
          assertTrue(paramsErrors.getErrorsOf(params.recurrentWeights)!!.values.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(-0.094779f, 0.039051f, -0.081651f, -0.596849f, -0.209882f)),
            tolerance = 1.0e-06f))
        }

        it("should match the expected errors of the input") {
          assertTrue(layer.inputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(1.367817f, -0.421073f, -0.501533f, -0.136723f)),
            tolerance = 1.0e-06f))
        }
      }
    }
  }
})
