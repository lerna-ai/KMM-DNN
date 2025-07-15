/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.recurrent.ran

import com.kotlinnlp.simplednn.core.arrays.DistributionArray
import com.kotlinnlp.simplednn.core.layers.models.recurrent.ran.RANLayerParameters
import com.kotlinnlp.simplednn.core.functionalities.losses.MSECalculator
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
class RANLayerStructureSpec : Spek({

  describe("a RANLayer") {

    context("forward") {

      context("without previous state context") {

        val layer = RANLayerStructureUtils.buildLayer(RANLayersWindow.Empty)
        layer.forward()

        it("should match the expected input gate") {
          assertTrue {
            layer.inputGate.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.39652f, 0.25162f, 0.5f, 0.70475f, 0.45264f)),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected forget gate") {
          assertTrue {
            layer.forgetGate.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.85321f, 0.43291f, 0.11609f, 0.51999f, 0.24232f)),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected candidate") {
          assertTrue {
            layer.candidate.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(1.02f, -0.1f, 0.1f, 2.03f, -1.41f)),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected outputArray") {
          assertTrue {
            layer.outputArray.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.38375f, -0.02516f, 0.04996f, 0.8918f, -0.56369f)),
              tolerance = 1.0e-05f)
          }
        }
      }

      context("with previous state context") {

        val layer = RANLayerStructureUtils.buildLayer(RANLayersWindow.Back)
        layer.forward()

        it("should match the expected input gate") {
          assertTrue {
            layer.inputGate.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.72312f, 0.24974f, 0.54983f, 0.82054f, 0.53494f)),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected forget gate") {
          assertTrue {
            layer.forgetGate.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.91133f, 0.18094f, 0.04834f, 0.67481f, 0.38936f)),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected candidate") {
          assertTrue {
            layer.candidate.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(1.02f, -0.1f, 0.1f, 2.03f, -1.41f)),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected outputArray") {
          assertTrue {
            layer.outputArray.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.5045f, 0.01121f, 0.04046f, 0.78504f, -0.78786f)),
              tolerance = 1.0e-05f)
          }
        }
      }
    }

    context("forward with relevance") {

      context("without previous state context") {

        val layer = RANLayerStructureUtils.buildLayer(RANLayersWindow.Empty)
        val contributions = RANLayerParameters(inputSize = 4, outputSize = 5)

        layer.forward(contributions)

        it("should match the expected input gate") {
          assertTrue {
            layer.inputGate.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.39652f, 0.25162f, 0.5f, 0.70475f, 0.45264f)),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected forget gate") {
          assertTrue {
            layer.forgetGate.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.85321f, 0.43291f, 0.11609f, 0.51999f, 0.24232f)),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected candidate") {
          assertTrue {
            layer.candidate.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(1.02f, -0.1f, 0.1f, 2.03f, -1.41f)),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected outputArray") {
          assertTrue {
            layer.outputArray.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.38375f, -0.02516f, 0.04996f, 0.8918f, -0.56369f)),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected contributions of the input gate") {
          val inputGateContrib: DenseNDArray = contributions.inputGate.weights.values
          assertTrue {
            inputGateContrib.equals(
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

        it("should match the expected contributions of the candidate") {
          val candidateContrib: DenseNDArray = contributions.candidate.weights.values
          assertTrue {
            candidateContrib.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(0.85f, -0.13f, 0.05f, 0.25f),
                floatArrayOf(0.56f, -0.63f, 0.27f, -0.3f),
                floatArrayOf(-0.465f, 0.315f, -0.225f, 0.475f),
                floatArrayOf(0.975f, 0.715f, -0.635f, 0.975f),
                floatArrayOf(-0.475f, -0.795f, 0.735f, -0.875f)
              )),
              tolerance = 1.0e-05f)
          }
        }

        layer.setOutputRelevance(DistributionArray.uniform(length = 5))
        layer.propagateRelevanceToGates(contributions)
        layer.setInputRelevance(contributions)

        it("should match the expected input relevance") {
          val relevance: DenseNDArray = layer.inputArray.relevance
          assertTrue {
            relevance.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-6.80494f, 7.20431f, -4.37039f, 4.97103f)),
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

        val prevStateLayer = RANLayersWindow.Back.getPrevState()
        val contextWindow = mock<LayersWindow>()
        val layer = RANLayerStructureUtils.buildLayer(contextWindow)
        val contributions = RANLayerParameters(inputSize = 4, outputSize = 5)

        whenever(contextWindow.getPrevState()).thenReturn(prevStateLayer)

        layer.forward(contributions)

        it("should match the expected input gate") {
          assertTrue {
            layer.inputGate.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.72312f, 0.24974f, 0.54983f, 0.82054f, 0.53494f)),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected forget gate") {
          assertTrue {
            layer.forgetGate.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.91133f, 0.18094f, 0.04834f, 0.67481f, 0.38936f)),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected candidate") {
          assertTrue {
            layer.candidate.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(1.02f, -0.1f, 0.1f, 2.03f, -1.41f)),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected outputArray") {
          assertTrue {
            layer.outputArray.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.5045f, 0.01121f, 0.04046f, 0.78504f, -0.78786f)),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected contributions of the input gate") {
          val inputGateContrib: DenseNDArray = contributions.inputGate.weights.values
          assertTrue {
            inputGateContrib.equals(
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

        it("should match the expected contributions of the forget gate") {
          val forgetGateContrib: DenseNDArray = contributions.forgetGate.weights.values
          assertTrue {
            forgetGateContrib.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(0.0325f, -0.2475f, 1.0125f, 0.5125f),
                floatArrayOf(-0.5350f, 0.2050f, -0.0650f, 0.025f),
                floatArrayOf(-0.6725f, -0.8325f, 0.3375f, -0.4125f),
                floatArrayOf(0.7450f, -0.7850f, 0.2950f, -0.275f),
                floatArrayOf(0.4475f, -0.6525f, 0.4275f, -0.9125f)
              )),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected contributions of the candidate") {
          val candidateContrib: DenseNDArray = contributions.candidate.weights.values
          assertTrue {
            candidateContrib.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(0.85f, -0.13f, 0.05f, 0.25f),
                floatArrayOf(0.56f, -0.63f, 0.27f, -0.3f),
                floatArrayOf(-0.465f, 0.315f, -0.225f, 0.475f),
                floatArrayOf(0.975f, 0.715f, -0.635f, 0.975f),
                floatArrayOf(-0.475f, -0.795f, 0.735f, -0.875f)
              )),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected recurrent contributions of the input gate") {
          val inputGateContrib: DenseNDArray = contributions.inputGate.recurrentWeights.values
          assertTrue {
            inputGateContrib.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(0.04f, 0.2f, -0.2f, 0.94f, 0.60f),
                floatArrayOf(0.14f, -0.16f, -0.06f, 0.63f, -0.56f),
                floatArrayOf(0.15f, 0.15f, -0.24f, 0.42f, -0.43f),
                floatArrayOf(0.08f, 0.06f, -0.07f, 0.26f, 0.72f),
                floatArrayOf(0.08f, 0.08f, -0.28f, 0.05f, 0.20f)
              )),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected recurrent contributions of the forget gate") {
          val forgetGateContrib: DenseNDArray = contributions.forgetGate.recurrentWeights.values
          assertTrue {
            forgetGateContrib.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(0.07f, -0.03f, 0.39f, 0.18f, 0.41f),
                floatArrayOf(-0.08f, -0.16f, 0.02f, -0.7f, -0.22f),
                floatArrayOf(-0.03f, -0.27f, -0.18f, -0.99f, 0.07f),
                floatArrayOf(-0.12f, 0.06f, -0.07f, 0.38f, 0.50f),
                floatArrayOf(-0.05f, 0.01f, -0.03f, 0.72f, -0.41f)
              )),
              tolerance = 1.0e-05f)
          }
        }

        layer.setOutputRelevance(DistributionArray.uniform(length = 5))
        layer.propagateRelevanceToGates(contributions)
        layer.setInputRelevance(contributions)
        layer.setRecurrentRelevance(contributions)

        it("should match the expected relevance of the input gate") {
          val relevance: DenseNDArray = layer.inputGate.relevance
          assertTrue {
            relevance.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.13434f, -0.09416f, 0.16398f, 0.15841f, 0.07058f)),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected relevance of the forget gate") {
          val relevance: DenseNDArray = layer.forgetGate.relevance
          assertTrue {
            relevance.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.03434f, 0.19416f, -0.06398f, -0.05841f, 0.02942f)),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected relevance of the candidate") {
          val relevance: DenseNDArray = layer.candidate.relevance
          assertTrue {
            relevance.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.13434f, -0.09416f, 0.16398f, 0.15841f, 0.07058f)),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected input relevance") {
          val relevance: DenseNDArray = layer.inputArray.relevance
          assertTrue {
            relevance.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.73699f, 0.21761f, -0.13861f, 1.13203f)),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected recurrent relevance") {
          val relevance: DenseNDArray = prevStateLayer.outputArray.relevance
          assertTrue {
            relevance.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.15578f, 0.3737f, -0.40348f, 0.45246f, -0.05248f)),
              tolerance = 1.0e-05f)
          }
        }
      }
    }

    context("backward") {

      context("without previous and next state") {

        val layer = RANLayerStructureUtils.buildLayer(RANLayersWindow.Empty)

        layer.forward()

        val errors = MSECalculator().calculateErrors(
          output = layer.outputArray.values,
          outputGold = RANLayerStructureUtils.getOutputGold())

        layer.outputArray.assignErrors(errors)
        val paramsErrors = layer.backward(propagateToInput = true)

        val params = layer.params

        it("should match the expected errors of the outputArray") {
          assertTrue {
            layer.outputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.15882f, -0.77467f, 0.19946f, -0.15316f, -0.69159f)),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected errors of the input gate") {
          assertTrue {
            layer.inputGate.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.03877f, 0.01459f, 0.00499f, -0.06469f, 0.2416f)),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected errors of the forget gate") {
          assertTrue {
            layer.forgetGate.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f)),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected errors of the candidate") {
          assertTrue {
            layer.candidate.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.06298f, -0.19492f, 0.09973f, -0.10794f, -0.31304f)),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected errors of the input gate biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.inputGate.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.03877f, 0.01459f, 0.00499f, -0.06469f, 0.2416f)),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected errors of the forget gate biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.forgetGate.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f)),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected errors of the candidate biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.candidate.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.06298f, -0.19492f, 0.09973f, -0.10794f, -0.31304f)),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected errors of the input gate weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.inputGate.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(0.03101f, 0.03489f, 0.03489f, -0.03877f),
                floatArrayOf(-0.01167f, -0.01313f, -0.01313f, 0.01459f),
                floatArrayOf(-0.00399f, -0.00449f, -0.00449f, 0.00499f),
                floatArrayOf(0.05175f, 0.05822f, 0.05822f, -0.06469f),
                floatArrayOf(-0.19328f, -0.21744f, -0.21744f, 0.2416f)
              )),
              tolerance = 1.0e-05f)
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
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected errors of the candidate weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.candidate.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(0.05038f, 0.05668f, 0.05668f, -0.06298f),
                floatArrayOf(0.15594f, 0.17543f, 0.17543f, -0.19492f),
                floatArrayOf(-0.07978f, -0.08976f, -0.08976f, 0.09973f),
                floatArrayOf(0.08635f, 0.09714f, 0.09714f, -0.10794f),
                floatArrayOf(0.25044f, 0.28174f, 0.28174f, -0.31304f)
              )),
              tolerance = 1.0e-05f)
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
              tolerance = 1.0e-05f)
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
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected errors of the inputArray") {
          assertTrue {
            layer.inputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.21996f, -0.12731f, 0.10792f, 0.49361f)),
              tolerance = 1.0e-05f)
          }
        }
      }

      context("with previous state only") {

        val layer = RANLayerStructureUtils.buildLayer(RANLayersWindow.Back)

        layer.forward()

        val errors = MSECalculator().calculateErrors(
          output = layer.outputArray.values,
          outputGold = RANLayerStructureUtils.getOutputGold())

        layer.outputArray.assignErrors(errors)
        val paramsErrors = layer.backward(propagateToInput = true)

        val params = layer.params

        it("should match the expected errors of the outputArray") {
          assertTrue {
            layer.outputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.04883f, -0.73869f, 0.19015f, -0.32806f, -0.46949f)),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected errors of the input gate") {
          assertTrue {
            layer.inputGate.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.00997f, 0.01384f, 0.00471f, -0.09807f, 0.16469f)),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected errors of the forget gate") {
          assertTrue {
            layer.forgetGate.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.00078f, -0.02161f, -0.00255f, 0.05157f, 0.07412f)),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected errors of the candidate") {
          assertTrue {
            layer.candidate.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.03531f, -0.18448f, 0.10455f, -0.26919f, -0.25115f)),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected errors of the input gate biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.inputGate.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.00997f, 0.01384f, 0.00471f, -0.09807f, 0.16469f)),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected errors of the forget gate biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.forgetGate.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.00078f, -0.02161f, -0.00255f, 0.05157f, 0.07412f)),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected errors of the candidate biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.candidate.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.03531f, -0.18448f, 0.10455f, -0.26919f, -0.25115f)),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected errors of the input gate weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.inputGate.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(0.00798f, 0.00898f, 0.00898f, -0.00997f),
                floatArrayOf(-0.01107f, -0.01246f, -0.01246f, 0.01384f),
                floatArrayOf(-0.00377f, -0.00424f, -0.00424f, 0.00471f),
                floatArrayOf(0.07845f, 0.08826f, 0.08826f, -0.09807f),
                floatArrayOf(-0.13175f, -0.14822f, -0.14822f, 0.16469f)
              )),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected errors of the forget gate weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.forgetGate.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(-0.00062f, -0.0007f, -0.0007f, 0.00078f),
                floatArrayOf(0.01729f, 0.01945f, 0.01945f, -0.02161f),
                floatArrayOf(0.00204f, 0.00229f, 0.00229f, -0.00255f),
                floatArrayOf(-0.04125f, -0.04641f, -0.04641f, 0.05157f),
                floatArrayOf(-0.0593f, -0.06671f, -0.06671f, 0.07412f)
              )),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected errors of the candidate weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.candidate.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(0.02825f, 0.03178f, 0.03178f, -0.03531f),
                floatArrayOf(0.14759f, 0.16603f, 0.16603f, -0.18448f),
                floatArrayOf(-0.08364f, -0.09409f, -0.09409f, 0.10455f),
                floatArrayOf(0.21535f, 0.24227f, 0.24227f, -0.26919f),
                floatArrayOf(0.20092f, 0.22604f, 0.22604f, -0.25115f)
              )),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected errors of the input gate recurrent weights") {
          assertTrue {
            paramsErrors.getErrorsOf(params.inputGate.recurrentWeights)!!.values.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(0.00199f, -0.00199f, 0.00299f, 0.00898f, 0.00798f),
                floatArrayOf(-0.00277f, 0.00277f, -0.00415f, -0.01246f, -0.01107f),
                floatArrayOf(-0.00094f, 0.00094f, -0.00141f, -0.00424f, -0.00377f),
                floatArrayOf(0.01961f, -0.01961f, 0.02942f, 0.08826f, 0.07845f),
                floatArrayOf(-0.03294f, 0.03294f, -0.04941f, -0.14822f, -0.13175f)
              )),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected errors of the forget gate recurrent weights") {
          assertTrue {
            paramsErrors.getErrorsOf(params.forgetGate.recurrentWeights)!!.values.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(-0.00016f, 0.00016f, -0.00023f, -0.0007f, -0.00062f),
                floatArrayOf(0.00432f, -0.00432f, 0.00648f, 0.01945f, 0.01729f),
                floatArrayOf(0.00051f, -0.00051f, 0.00076f, 0.00229f, 0.00204f),
                floatArrayOf(-0.01031f, 0.01031f, -0.01547f, -0.04641f, -0.04125f),
                floatArrayOf(-0.01482f, 0.01482f, -0.02224f, -0.06671f, -0.0593f)
              )),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected errors of the inputArray") {
          assertTrue {
            layer.inputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.21972f, 0.09327f, -0.127f, 0.17217f)),
              tolerance = 1.0e-05f)
          }
        }
      }

      context("with next state only") {

        val layer = RANLayerStructureUtils.buildLayer(RANLayersWindow.Front)

        layer.forward()

        val errors = MSECalculator().calculateErrors(
          output = layer.outputArray.values,
          outputGold = RANLayerStructureUtils.getOutputGold())

        layer.outputArray.assignErrors(errors)
        val paramsErrors = layer.backward(propagateToInput = true)

        val params = layer.params

        it("should match the expected errors of the outputArray") {
          assertTrue {
            layer.outputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.30882f, -0.42467f, 0.59946f, -1.00316f, -0.88159f)),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected errors of the input gate") {
          assertTrue {
            layer.inputGate.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.07538f, 0.008f, 0.01499f, -0.42373f, 0.30797f)),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected errors of the forget gate") {
          assertTrue {
            layer.forgetGate.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f)),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected errors of the candidate") {
          assertTrue {
            layer.candidate.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.12245f, -0.10685f, 0.29973f, -0.70697f, -0.39905f)),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected errors of the input gate biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.inputGate.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.07538f, 0.008f, 0.01499f, -0.42373f, 0.30797f)),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected errors of the forget gate biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.forgetGate.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f)),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected errors of the candidate biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.candidate.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.12245f, -0.10685f, 0.29973f, -0.70697f, -0.39905f)),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected errors of the input gate weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.inputGate.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(0.0603f, 0.06784f, 0.06784f, -0.07538f),
                floatArrayOf(-0.0064f, -0.0072f, -0.0072f, 0.00800f),
                floatArrayOf(-0.01199f, -0.01349f, -0.01349f, 0.01499f),
                floatArrayOf(0.33899f, 0.38136f, 0.38136f, -0.42373f),
                floatArrayOf(-0.24638f, -0.27718f, -0.27718f, 0.30797f)
              )),
              tolerance = 1.0e-05f)
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
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected errors of the candidate weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.candidate.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(0.09796f, 0.11021f, 0.11021f, -0.12245f),
                floatArrayOf(0.08548f, 0.09617f, 0.09617f, -0.10685f),
                floatArrayOf(-0.23978f, -0.26976f, -0.26976f, 0.29973f),
                floatArrayOf(0.56558f, 0.63627f, 0.63627f, -0.70697f),
                floatArrayOf(0.31924f, 0.35914f, 0.35914f, -0.39905f)
              )),
              tolerance = 1.0e-05f)
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
              tolerance = 1.0e-05f)
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
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected errors of the inputArray") {
          assertTrue {
            layer.inputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.55722f, 0.45624f, -0.39506f, 0.30611f)),
              tolerance = 1.0e-05f)
          }
        }
      }

      context("with previous and next state") {

        val layer = RANLayerStructureUtils.buildLayer(RANLayersWindow.Bilateral)

        layer.forward()

        val errors = MSECalculator().calculateErrors(
          output = layer.outputArray.values,
          outputGold = RANLayerStructureUtils.getOutputGold())

        layer.outputArray.assignErrors(errors)
        val paramsErrors = layer.backward(propagateToInput = true)

        val params = layer.params

        it("should match the expected errors of the outputArray") {
          assertTrue {
            layer.outputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.19883f, -0.38869f, 0.59015f, -1.17806f, -0.65949f)),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected errors of the input gate") {
          assertTrue {
            layer.inputGate.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.04061f, 0.00728f, 0.01461f, -0.35216f, 0.23134f)),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected errors of the forget gate") {
          assertTrue {
            layer.forgetGate.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.00317f, -0.01137f, -0.00791f, 0.18518f, 0.10412f)),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected errors of the candidate") {
          assertTrue {
            layer.candidate.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.14378f, -0.09707f, 0.32448f, -0.96664f, -0.35279f)),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected errors of the input gate biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.inputGate.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.04061f, 0.00728f, 0.01461f, -0.35216f, 0.23134f)),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected errors of the forget gate biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.forgetGate.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.00317f, -0.01137f, -0.00791f, 0.18518f, 0.10412f)),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected errors of the candidate biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.candidate.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.14378f, -0.09707f, 0.32448f, -0.96664f, -0.35279f)),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected errors of the input gate weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.inputGate.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(0.03248f, 0.03655f, 0.03655f, -0.04061f),
                floatArrayOf(-0.00583f, -0.00655f, -0.00655f, 0.00728f),
                floatArrayOf(-0.01169f, -0.01315f, -0.01315f, 0.01461f),
                floatArrayOf(0.28172f, 0.31694f, 0.31694f, -0.35216f),
                floatArrayOf(-0.18507f, -0.20820f, -0.20820f, 0.23134f)
              )),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected errors of the forget gate weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.forgetGate.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(-0.00254f, -0.00285f, -0.00285f, 0.00317f),
                floatArrayOf(0.00910f, 0.01023f, 0.01023f, -0.01137f),
                floatArrayOf(0.00633f, 0.00712f, 0.00712f, -0.00791f),
                floatArrayOf(-0.14814f, -0.16666f, -0.16666f, 0.18518f),
                floatArrayOf(-0.08330f, -0.09371f, -0.09371f, 0.10412f)
              )),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected errors of the candidate weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.candidate.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(0.11502f, 0.12940f, 0.12940f, -0.14378f),
                floatArrayOf(0.07766f, 0.08737f, 0.08737f, -0.09707f),
                floatArrayOf(-0.25959f, -0.29204f, -0.29204f, 0.32448f),
                floatArrayOf(0.77332f, 0.86998f, 0.86998f, -0.96664f),
                floatArrayOf(0.28223f, 0.31751f, 0.31751f, -0.35279f)
              )),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected errors of the input gate recurrent weights") {
          assertTrue {
            paramsErrors.getErrorsOf(params.inputGate.recurrentWeights)!!.values.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(0.00812f, -0.00812f, 0.01218f, 0.03655f, 0.03248f),
                floatArrayOf(-0.00146f, 0.00146f, -0.00218f, -0.00655f, -0.00583f),
                floatArrayOf(-0.00292f, 0.00292f, -0.00438f, -0.01315f, -0.01169f),
                floatArrayOf(0.07043f, -0.07043f, 0.10565f, 0.31694f, 0.28172f),
                floatArrayOf(-0.04627f, 0.04627f, -0.06940f, -0.20820f, -0.18507f)
              )),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected errors of the forget gate recurrent weights") {
          assertTrue {
            paramsErrors.getErrorsOf(params.forgetGate.recurrentWeights)!!.values.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(-0.00063f, 0.00063f, -0.00095f, -0.00285f, -0.00254f),
                floatArrayOf(0.00227f, -0.00227f, 0.00341f, 0.01023f, 0.00910f),
                floatArrayOf(0.00158f, -0.00158f, 0.00237f, 0.00712f, 0.00633f),
                floatArrayOf(-0.03704f, 0.03704f, -0.05555f, -0.16666f, -0.14814f),
                floatArrayOf(-0.02082f, 0.02082f, -0.03124f, -0.09371f, -0.08330f)
              )),
              tolerance = 1.0e-05f)
          }
        }

        it("should match the expected errors of the inputArray") {
          assertTrue {
            layer.inputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.65243f, 0.74348f, -0.76607f, -0.15266f)),
              tolerance = 1.0e-05f)
          }
        }
      }
    }
  }
})
