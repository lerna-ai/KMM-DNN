/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.recurrent.simple

import com.kotlinnlp.simplednn.core.arrays.DistributionArray
import com.kotlinnlp.simplednn.core.layers.models.recurrent.simple.SimpleRecurrentLayerParameters
import com.kotlinnlp.simplednn.core.layers.models.recurrent.LayersWindow
import com.kotlinnlp.simplednn.core.optimizer.getErrorsOf
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.mockito.kotlin.mock
import org.mockito.kotlin.whenever
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 *
 */
class SimpleRecurrentLayerStructureSpec : Spek({

  describe("a SimpleRecurrentLayer") {

    context("forward") {

      context("without previous state context") {

        val layer = SimpleRecurrentLayerStructureUtils.buildLayer(SimpleRecurrentLayersWindow.Empty)
        layer.forward()

        it("should match the expected output") {
          assertTrue(layer.outputArray.values.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(-0.39693f, -0.79688f, 0.0f, 0.70137f, -0.18775f)),
            tolerance = 1.0e-05f))
        }
      }

      context("with previous state context") {

        val layer = SimpleRecurrentLayerStructureUtils.buildLayer(SimpleRecurrentLayersWindow.Back)
        layer.forward()

        it("should match the expected output") {
          assertTrue(layer.outputArray.values.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(0.59539f, -0.8115f, 0.17565f, 0.88075f, 0.08444f)),
            tolerance = 1.0e-05f))
        }
      }
    }

    context("forward with relevance") {

      context("without previous state context") {

        val layer = SimpleRecurrentLayerStructureUtils.buildLayer(SimpleRecurrentLayersWindow.Empty)
        val contributions = SimpleRecurrentLayerParameters(inputSize = 4, outputSize = 5)

        layer.forward(contributions)

        it("should match the expected output") {
          assertTrue(layer.outputArray.values.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(-0.39693f, -0.79688f, 0.0f, 0.70137f, -0.18775f)),
            tolerance = 1.0e-05f))
        }

        it("should match the expected contributions") {
          val wContr: DenseNDArray = contributions.unit.weights.values
          assertTrue {
            wContr.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(-0.3f, -0.44f, 0.82f, -0.5f),
                floatArrayOf(-0.56f, 0.36f, -0.09f, -0.8f),
                floatArrayOf(-0.635f, 0.555f, -0.345f, 0.425f),
                floatArrayOf(-0.44f, 1.01f, 0.2f, 0.1f),
                floatArrayOf(-0.42f, -1.0f, 0.53f, 0.7f)
              )),
              tolerance = 1.0e-05f)
          }
        }

        layer.setOutputRelevance(DistributionArray.uniform(length = 5))
        layer.setInputRelevance(contributions)

        it("should match the expected input relevance") {
          val relevance: DenseNDArray = layer.inputArray.relevance
          assertTrue {
            relevance.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-12.08396f, 12.52343f, -7.69489f, 8.25543f)),
              tolerance = 1.0e-05f)
          }
        }

        it("should throw an Exception when calculating the recurrent relevance") {
          assertFailsWith <NullPointerException> {
            layer.setRecurrentRelevance(contributions)
          }
        }
      }

      context("with previous state context") {

        val prevStateLayer = SimpleRecurrentLayersWindow.Back.getPrevState()
        val contextWindow = mock<LayersWindow>()
        val layer = SimpleRecurrentLayerStructureUtils.buildLayer(contextWindow)
        val contributions = SimpleRecurrentLayerParameters(inputSize = 4, outputSize = 5)

        whenever(contextWindow.getPrevState()).thenReturn(prevStateLayer)

        layer.forward(contributions)

        it("should match the expected output") {
          assertTrue(layer.outputArray.values.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(0.59539f, -0.8115f, 0.17565f, 0.88075f, 0.08444f)),
            tolerance = 1.0e-05f))
        }

        it("should match the expected contributions") {
          val wContr: DenseNDArray = contributions.unit.weights.values
          assertTrue {
            wContr.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(-0.35f, -0.49f, 0.77f, -0.55f),
                floatArrayOf(-0.56f, 0.36f, -0.09f, -0.8f),
                floatArrayOf(-0.5975f, 0.5925f, -0.3075f, 0.4625f),
                floatArrayOf(-0.54f, 0.91f, 0.1f, 0.0f),
                floatArrayOf(-0.37f, -0.95f, 0.58f, 0.75f)
              )),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected recurrent contributions") {
          val wRecContr: DenseNDArray = contributions.unit.recurrentWeights.values
          assertTrue {
            wRecContr.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(0.04f, 0.1979f, -0.19305f, 0.7563f, 0.50483f),
                floatArrayOf(0.13816f, -0.15790f, -0.05826f, 0.50141f, -0.46483f),
                floatArrayOf(0.14764f, 0.14764f, -0.23392f, 0.32815f, -0.36202f),
                floatArrayOf(0.08f, 0.06026f, -0.06566f, 0.22326f, 0.61123f),
                floatArrayOf(0.07843f, 0.07843f, -0.27305f, 0.03163f, 0.15921f)
              )),
              tolerance = 1.0e-05f)
          }
        }

        layer.setOutputRelevance(DistributionArray.uniform(length = 5))
        layer.setInputRelevance(contributions)
        layer.setRecurrentRelevance(contributions)

        it("should match the expected input relevance") {
          val relevance: DenseNDArray = layer.inputArray.relevance
          assertTrue {
            relevance.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-1.27469f, -0.95735f, 0.85408f, 1.65854f)),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected recurrent relevance") {
          val relevance: DenseNDArray = prevStateLayer.outputArray.relevance
          assertTrue {
            relevance.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.30048f, 0.38969f, -0.80763f, 0.54242f, 0.29448f)),
              tolerance = 1.0e-05f)
          }
        }
      }
    }

    context("backward") {

      context("without next and previous state") {

        val layer = SimpleRecurrentLayerStructureUtils.buildLayer(SimpleRecurrentLayersWindow.Empty)

        layer.forward()

        val outputGold = SimpleRecurrentLayerStructureUtils.getOutputGold()

        layer.outputArray.assignErrors(layer.outputArray.values.sub(outputGold))
        val paramsErrors = layer.backward(propagateToInput = true)

        val params = layer.params

        it("should match the expected errors of the output") {
          assertTrue(layer.outputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(-0.81459f, -0.56459f, 0.15f, -0.47689f, -0.61527f)),
            tolerance = 1.0e-05f))
        }

        it("should match the expected errors of the biases") {
          assertTrue(paramsErrors.getErrorsOf(params.unit.biases)!!.values.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(-0.81459f, -0.56459f, 0.15f, -0.47689f, -0.61527f)),
            tolerance = 1.0e-05f))
        }

        it("should match the expected errors of the weights") {
          assertTrue((paramsErrors.getErrorsOf(params.unit.weights)!!.values).equals(
            DenseNDArrayFactory.arrayOf(listOf(
              floatArrayOf(0.65167f, 0.73313f, 0.73313f, -0.81459f),
              floatArrayOf(0.45167f, 0.50813f, 0.50813f, -0.56459f),
              floatArrayOf(-0.12f, -0.135f, -0.135f, 0.15f),
              floatArrayOf(0.38151f, 0.4292f, 0.4292f, -0.47689f),
              floatArrayOf(0.49221f, 0.55374f, 0.55374f, -0.61527f)
            )),
            tolerance = 1.0e-05f))
        }

        it("should match the expected errors of the recurrent weights") {
          assertTrue(paramsErrors.getErrorsOf(params.unit.recurrentWeights)!!.values.equals(
            DenseNDArrayFactory.arrayOf(listOf(
              floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
              floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
              floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
              floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
              floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f)
            )),
            tolerance = 1.0e-05f))
        }

        it("should match the expected errors of the input") {
          assertTrue(layer.inputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(-1.32512f, -0.55398f, 1.0709f, 0.5709f)),
            tolerance = 1.0e-05f))
        }
      }

      context("with prev state only") {

        val layer = SimpleRecurrentLayerStructureUtils.buildLayer(SimpleRecurrentLayersWindow.Back)

        layer.forward()

        val outputGold = SimpleRecurrentLayerStructureUtils.getOutputGold()

        layer.outputArray.assignErrors(layer.outputArray.values.sub(outputGold))
        val paramsErrors = layer.backward(propagateToInput = true)

        val params = layer.params

        it("should match the expected errors of the output") {
          assertTrue(layer.outputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(0.01639f, -0.53319f, 0.3156f, -0.17029f, -0.36295f)),
            tolerance = 1.0e-05f))
        }

        it("should match the expected errors of the biases") {
          assertTrue(paramsErrors.getErrorsOf(params.unit.biases)!!.values.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(0.01639f, -0.53319f, 0.3156f, -0.17029f, -0.36295f)),
            tolerance = 1.0e-05f))
        }

        it("should match the expected errors of the weights") {
          assertTrue((paramsErrors.getErrorsOf(params.unit.weights)!!.values).equals(
            DenseNDArrayFactory.arrayOf(listOf(
              floatArrayOf(-0.01311f, -0.01475f, -0.01475f, 0.01639f),
              floatArrayOf(0.42655f, 0.47987f, 0.47987f, -0.53319f),
              floatArrayOf(-0.25248f, -0.28404f, -0.28404f, 0.3156f),
              floatArrayOf(0.13623f, 0.15326f, 0.15326f, -0.17029f),
              floatArrayOf(0.29036f, 0.32666f, 0.32666f, -0.36295f)
            )),
            tolerance = 1.0e-05f))
        }

        it("should match the expected errors of the recurrent weights") {
          assertTrue(paramsErrors.getErrorsOf(params.unit.recurrentWeights)!!.values.equals(
            DenseNDArrayFactory.arrayOf(listOf(
              floatArrayOf(-0.00323f, 0.00323f, -0.00477f, -0.01174f, -0.01088f),
              floatArrayOf(0.10524f, -0.10524f, 0.15533f, 0.38193f, 0.35406f),
              floatArrayOf(-0.06229f, 0.06229f, -0.09194f, -0.22606f, -0.20957f),
              floatArrayOf(0.03361f, -0.03361f, 0.04961f, 0.12198f, 0.11308f),
              floatArrayOf(0.07164f, -0.07164f, 0.10573f, 0.25998f, 0.24101f)
            )),
            tolerance = 1.0e-05f))
        }

        it("should match the expected errors of the input") {
          assertTrue(layer.inputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(-0.42553f, -0.20751f, 0.28232f, 0.30119f)),
            tolerance = 1.0e-05f))
        }
      }

      context("with next state only") {

        val layer = SimpleRecurrentLayerStructureUtils.buildLayer(SimpleRecurrentLayersWindow.Front)

        layer.forward()

        val outputGold = SimpleRecurrentLayerStructureUtils.getOutputGold()

        layer.outputArray.assignErrors(layer.outputArray.values.sub(outputGold))
        val paramsErrors = layer.backward(propagateToInput = true)

        val params = layer.params

        it("should match the expected errors of the output") {
          assertTrue(layer.outputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(-0.59555f, -0.71058f, 0.41f, -0.51754f, -1.4546f)),
            tolerance = 1.0e-05f))
        }

        it("should match the expected errors of the biases") {
          assertTrue(paramsErrors.getErrorsOf(params.unit.biases)!!.values.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(-0.59555f, -0.71058f, 0.41f, -0.51754f, -1.4546f)),
            tolerance = 1.0e-05f))
        }

        it("should match the expected errors of the weights") {
          assertTrue((paramsErrors.getErrorsOf(params.unit.weights)!!.values).equals(
            DenseNDArrayFactory.arrayOf(listOf(
              floatArrayOf(0.47644f, 0.536f, 0.536f, -0.59555f),
              floatArrayOf(0.56847f, 0.63952f, 0.63952f, -0.71058f),
              floatArrayOf(-0.328f, -0.369f, -0.369f, 0.41f),
              floatArrayOf(0.41403f, 0.46578f, 0.46578f, -0.51754f),
              floatArrayOf(1.16368f, 1.30914f, 1.30914f, -1.4546f)
            )),
            tolerance = 1.0e-05f))
        }

        it("should match the expected errors of the recurrent weights") {
          assertTrue(paramsErrors.getErrorsOf(params.unit.recurrentWeights)!!.values.equals(
            DenseNDArrayFactory.arrayOf(listOf(
              floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
              floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
              floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
              floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
              floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f)
            )),
            tolerance = 1.0e-05f))
        }

        it("should match the expected errors of the input") {
          assertTrue(layer.inputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(-1.50405f, -1.34891f, 1.5466f, 0.01887f)),
            tolerance = 1.0e-05f))
        }
      }

      context("with next and previous state") {

        val layer = SimpleRecurrentLayerStructureUtils.buildLayer(SimpleRecurrentLayersWindow.Bilateral)

        layer.forward()

        val outputGold = SimpleRecurrentLayerStructureUtils.getOutputGold()

        layer.outputArray.assignErrors(layer.outputArray.values.sub(outputGold))
        val paramsErrors = layer.backward(propagateToInput = true)

        val params = layer.params

        it("should match the expected errors of the output") {
          assertTrue(layer.outputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(0.18422f, -0.66978f, 0.56758f, -0.18823f, -1.22675f)),
            tolerance = 1.0e-05f))
        }

        it("should match the expected errors of the biases") {
          assertTrue(paramsErrors.getErrorsOf(params.unit.biases)!!.values.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(0.18422f, -0.66978f, 0.56758f, -0.18823f, -1.22675f)),
            tolerance = 1.0e-05f))
        }

        it("should match the expected errors of the weights") {
          assertTrue((paramsErrors.getErrorsOf(params.unit.weights)!!.values).equals(
            DenseNDArrayFactory.arrayOf(listOf(
              floatArrayOf(-0.14738f, -0.1658f, -0.1658f, 0.18422f),
              floatArrayOf(0.53582f, 0.60280f, 0.60280f, -0.66978f),
              floatArrayOf(-0.45406f, -0.51082f, -0.51082f, 0.56758f),
              floatArrayOf(0.15058f, 0.16941f, 0.16941f, -0.18823f),
              floatArrayOf(0.9814f, 1.10408f, 1.10408f, -1.22675f)
            )),
            tolerance = 1.0e-05f))
        }

        it("should match the expected errors of the recurrent weights") {
          assertTrue(paramsErrors.getErrorsOf(params.unit.recurrentWeights)!!.values.equals(
            DenseNDArrayFactory.arrayOf(listOf(
              floatArrayOf(-0.03636f, 0.03636f, -0.05367f, -0.13196f, -0.12233f),
              floatArrayOf(0.1322f, -0.1322f, 0.19511f, 0.47976f, 0.44476f),
              floatArrayOf(-0.11203f, 0.11203f, -0.16534f, -0.40656f, -0.37689f),
              floatArrayOf(0.03715f, -0.03715f, 0.05483f, 0.13483f, 0.12499f),
              floatArrayOf(0.24213f, -0.24213f, 0.35737f, 0.87872f, 0.81461f)
            )),
            tolerance = 1.0e-05f))
        }

        it("should match the expected errors of the input") {
          assertTrue(layer.inputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(-0.62071f, -1.07621f, 0.81464f, -0.2535f)),
            tolerance = 1.0e-05f))
        }
      }
    }
  }
})
