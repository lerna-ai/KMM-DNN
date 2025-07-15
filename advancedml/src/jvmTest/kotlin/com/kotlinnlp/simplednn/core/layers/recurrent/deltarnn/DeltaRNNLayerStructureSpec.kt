/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.recurrent.deltarnn

import com.kotlinnlp.simplednn.core.arrays.DistributionArray
import com.kotlinnlp.simplednn.core.layers.models.recurrent.LayersWindow
import com.kotlinnlp.simplednn.core.layers.models.recurrent.deltarnn.DeltaRNNLayerParameters
import com.kotlinnlp.simplednn.core.optimizer.getErrorsOf
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.mockito.kotlin.mock
import org.mockito.kotlin.whenever
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 *
 */
class DeltaRNNLayerStructureSpec : Spek({

  describe("a DeltaRNNLayer") {

    context("forward") {

      context("without previous state context") {

        val layer = DeltaRNNLayerStructureUtils.buildLayer(DeltaLayersWindow.Empty)
        layer.forward()

        it("should match the expected candidate") {
          assertTrue {
            layer.candidate.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.568971f, 0.410323f, -0.39693f, 0.648091f, -0.449441f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected partition array") {
          assertTrue {
            layer.partition.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.519989f, 0.169384f, 0.668188f, 0.325195f, 0.601088f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected output") {
          assertTrue {
            layer.outputArray.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.287518f, 0.06939f, -0.259175f, 0.20769f, -0.263768f)),
              tolerance = 1.0e-06f)
          }
        }
      }

      context("with previous state context") {

        val layer = DeltaRNNLayerStructureUtils.buildLayer(DeltaLayersWindow.Back)
        layer.forward()

        it("should match the expected candidate") {
          assertTrue {
            layer.candidate.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.576403f, 0.40594f, -0.222741f, 0.36182f, -0.42253f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected partition array") {
          assertTrue {
            layer.partition.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.519989f, 0.169384f, 0.668188f, 0.325195f, 0.601088f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected output") {
          assertTrue {
            layer.outputArray.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.202158f, 0.228591f, -0.240679f, -0.350224f, -0.476828f)),
              tolerance = 1.0e-06f)
          }
        }
      }
    }

    context("forward with relevance") {

      context("without previous state context") {

        val layer = DeltaRNNLayerStructureUtils.buildLayer(DeltaLayersWindow.Empty)
        val contributions = DeltaRNNLayerParameters(
          inputSize = 4,
          outputSize = 5,
          weightsInitializer = null,
          biasesInitializer = null)

        layer.forward(contributions)

        it("should match the expected candidate") {
          assertTrue {
            layer.candidate.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.568971f, 0.410323f, -0.39693f, 0.648091f, -0.449441f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected partition array") {
          assertTrue {
            layer.partition.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.519989f, 0.169384f, 0.668188f, 0.325195f, 0.601088f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected output") {
          assertTrue {
            layer.outputArray.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.287518f, 0.06939f, -0.259175f, 0.20769f, -0.263768f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected contributions of the input") {
          val inputContrib: DenseNDArray = contributions.feedforwardUnit.weights.values
          assertTrue {
            inputContrib.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(-0.4f, -0.54f, 0.72f, -0.6f),
                floatArrayOf(-0.56f, 0.36f, -0.09f, -0.8f),
                floatArrayOf(-0.56f, 0.63f, -0.27f, 0.5f),
                floatArrayOf(-0.64f, 0.81f, 0.0f, -0.1f),
                floatArrayOf(-0.32f, -0.9f, 0.63f, 0.8f)
              )),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected recurrent contributions") {
          val recContrib: DenseNDArray = contributions.recurrentUnit.weights.values
          assertTrue {
            recContrib.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f)
              )),
              tolerance = 1.0e-06f)
          }
        }

        layer.setOutputRelevance(DistributionArray.uniform(length = 5))
        layer.propagateRelevanceToGates(contributions)
        layer.setInputRelevance(contributions)

        it("should match the expected relevance of the partition array") {
          val relevance: DenseNDArray = layer.partition.relevance
          assertTrue {
            relevance.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.1f, 0.1f, 0.1f, 0.1f, 0.1f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected relevance of the candidate") {
          val relevance: DenseNDArray = layer.candidate.relevance
          assertTrue {
            relevance.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.1f, 0.1f, 0.1f, 0.1f, 0.1f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected relevance of the d1 input array") {
          val relevance: DenseNDArray = layer.relevanceSupport.d1Input.relevance
          assertTrue {
            relevance.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.1f, 0.1f, 0.1f, 0.1f, 0.1f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected input relevance") {
          val relevance: DenseNDArray = layer.inputArray.relevance
          assertTrue {
            relevance.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.058871f, -0.524906f, 1.314539f, 0.269238f)),
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

        val prevStateLayer = DeltaLayersWindow.Back.getPrevState()
        val contextWindow = mock<LayersWindow>()
        val layer = DeltaRNNLayerStructureUtils.buildLayer(contextWindow)
        val contributions = DeltaRNNLayerParameters(inputSize = 4, outputSize = 5)

        whenever(contextWindow.getPrevState()).thenReturn(prevStateLayer)

        layer.forward(contributions)

        it("should match the expected candidate") {
          assertTrue {
            layer.candidate.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.576403f, 0.40594f, -0.222741f, 0.36182f, -0.42253f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected partition array") {
          assertTrue {
            layer.partition.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.519989f, 0.169384f, 0.668188f, 0.325195f, 0.601088f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected output") {
          assertTrue {
            layer.outputArray.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.202158f, 0.228591f, -0.240679f, -0.350224f, -0.476828f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected contributions of the input") {
          val inputContrib: DenseNDArray = contributions.feedforwardUnit.weights.values
          assertTrue {
            inputContrib.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(-0.4f, -0.54f, 0.72f, -0.6f),
                floatArrayOf(-0.56f, 0.36f, -0.09f, -0.8f),
                floatArrayOf(-0.56f, 0.63f, -0.27f, 0.5f),
                floatArrayOf(-0.64f, 0.81f, 0.0f, -0.1f),
                floatArrayOf(-0.32f, -0.9f, 0.63f, 0.8f)
              )),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected recurrent contributions") {
          val recContrib: DenseNDArray = contributions.recurrentUnit.weights.values
          assertTrue {
            recContrib.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(0.0f, 0.1579f, -0.23305f, 0.716298f, 0.464826f),
                floatArrayOf(0.138163f, -0.1579f, -0.058263f, 0.501409f, -0.464826f),
                floatArrayOf(0.177638f, 0.177638f, -0.203919f, 0.358149f, -0.332018f),
                floatArrayOf(0.0f, -0.019738f, -0.145656f, 0.14326f, 0.531229f),
                floatArrayOf(0.118425f, 0.118425f, -0.23305f, 0.07163f, 0.199211f)
              )),
              tolerance = 1.0e-06f)
          }
        }

        layer.setOutputRelevance(DistributionArray.uniform(length = 5))
        layer.propagateRelevanceToGates(contributions)
        layer.setInputRelevance(contributions)
        layer.setRecurrentRelevance(contributions)

        it("should match the expected relevance of the partition array") {
          val relevance: DenseNDArray = layer.partition.relevance
          assertTrue {
            relevance.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.1f, 0.1f, 0.1f, 0.1f, 0.1f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected relevance of the candidate") {
          val relevance: DenseNDArray = layer.candidate.relevance
          assertTrue {
            relevance.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.151155f, 0.030391f, 0.06021f, -0.029987f, 0.048968f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected relevance of the d1 input array") {
          val relevance: DenseNDArray = layer.relevanceSupport.d1Input.relevance
          assertTrue {
            relevance.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.101818f, 0.031252f, 0.074148f, -0.028935f, 0.031181f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected relevance of the d1 recurrent array") {
          val relevance: DenseNDArray = layer.relevanceSupport.d1Rec.relevance
          assertTrue {
            relevance.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.05417f, 0.000358f, -0.00857f, 0.000304f, 0.018798f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected relevance of the d2 array") {
          val relevance: DenseNDArray = layer.relevanceSupport.d2.relevance
          assertTrue {
            relevance.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.103506f, -0.001219f, -0.005368f, -0.001356f, -0.001011f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected input relevance") {
          val relevance: DenseNDArray = layer.inputArray.relevance
          assertTrue {
            relevance.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.156729f, -0.419269f, 1.161412f, 0.171329f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected recurrent relevance") {
          val relevance: DenseNDArray = prevStateLayer.outputArray.relevance
          assertTrue {
            relevance.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.077057f, 0.03487f, 0.116601f, 0.037238f, 0.131607f)),
              tolerance = 1.0e-06f)
          }
        }
      }
    }

    context("backward") {

      context("without previous and next state") {

        val layer = DeltaRNNLayerStructureUtils.buildLayer(DeltaLayersWindow.Empty)

        layer.forward()

        layer.outputArray.assignErrors(layer.outputArray.values.sub(DeltaRNNLayerStructureUtils.getOutputGold()))
        val paramsErrors = layer.backward(propagateToInput = true)

        val params = layer.params

        it("should match the expected errors of the outputArray") {
          assertTrue {
            layer.outputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.25913f, -0.677332f, -0.101842f, -1.370527f, -0.664109f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the candidate array") {
          assertTrue {
            layer.candidate.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.091124f, -0.095413f, -0.057328f, -0.258489f, -0.318553f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the partition array") {
          assertTrue {
            layer.partition.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.0368f, -0.039102f, 0.008963f, -0.194915f, 0.071569f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the candidate biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.feedforwardUnit.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.091124f, -0.095413f, -0.057328f, -0.258489f, -0.318553f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the partition biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.recurrentUnit.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.0368f, -0.039102f, 0.008963f, -0.194915f, 0.071569f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the alpha array") {
          assertTrue {
            paramsErrors.getErrorsOf(params.alpha)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the beta1 array") {
          assertTrue {
            paramsErrors.getErrorsOf(params.beta1)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.074722f, 0.104f, -0.017198f, -0.018094f, -0.066896f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the beta2 array") {
          assertTrue {
            paramsErrors.getErrorsOf(params.beta2)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.feedforwardUnit.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(0.007571f, 0.008517f, 0.008517f, -0.009463f),
                floatArrayOf(0.00075f, 0.000843f, 0.000843f, -0.000937f),
                floatArrayOf(-0.025515f, -0.028704f, -0.028704f, 0.031894f),
                floatArrayOf(0.073215f, 0.082367f, 0.082367f, -0.091519f),
                floatArrayOf(-0.159192f, -0.179091f, -0.179091f, 0.19899f)
              )),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the recurrent weights") {
          assertNull(paramsErrors.getErrorsOf(params.recurrentUnit.weights))
        }

        it("should match the expected errors of the inputArray") {
          assertTrue {
            layer.inputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.023319f, 0.253729f, -0.122248f, 0.190719f)),
              tolerance = 1.0e-06f)
          }
        }
      }

      context("with previous state only") {

        val layer = DeltaRNNLayerStructureUtils.buildLayer(DeltaLayersWindow.Back)

        layer.forward()

        layer.outputArray.assignErrors(layer.outputArray.values.sub(DeltaRNNLayerStructureUtils.getOutputGold()))
        val paramsErrors = layer.backward(propagateToInput = true)

        val params = layer.params

        it("should match the expected errors of the outputArray") {
          assertTrue {
            layer.outputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.352809f, -0.494163f, -0.085426f, -1.746109f, -0.7161f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the candidate array") {
          assertTrue {
            layer.candidate.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.122505f, -0.06991f, -0.054249f, -0.493489f, -0.353592f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the partition array") {
          assertTrue {
            layer.partition.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.06814f, -0.0145f, -0.001299f, -0.413104f, -0.041468f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the candidate biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.feedforwardUnit.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.122505f, -0.06991f, -0.054249f, -0.493489f, -0.353592f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the partition biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.recurrentUnit.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.06814f, -0.0145f, -0.001299f, -0.413104f, -0.041468f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the alpha array") {
          assertTrue {
            paramsErrors.getErrorsOf(params.alpha)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.1111f, -0.003156f, -0.002889f, -0.017586f, -0.020393f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the beta1 array") {
          assertTrue {
            paramsErrors.getErrorsOf(params.beta1)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.100454f, 0.076202f, -0.016275f, -0.034544f, -0.074254f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the beta2 array") {
          assertTrue {
            paramsErrors.getErrorsOf(params.beta2)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.135487f, 0.002895f, -0.009628f, -0.251233f, -0.097111f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.feedforwardUnit.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(-0.029084f, -0.03272f, -0.03272f, 0.036355f),
                floatArrayOf(-0.010076f, -0.011335f, -0.011335f, 0.012595f),
                floatArrayOf(-0.01401f, -0.015761f, -0.015761f, 0.017512f),
                floatArrayOf(0.252961f, 0.284582f, 0.284582f, -0.316202f),
                floatArrayOf(-0.072206f, -0.081232f, -0.081232f, 0.090257f)
              )),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the recurrent weights") {
          val wRec: DenseNDArray = paramsErrors.getErrorsOf(params.recurrentUnit.weights)!!.values as DenseNDArray
          assertTrue {
            wRec.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(0.000242f, -0.000242f, 0.000357f, 0.000878f, 0.000813f),
                floatArrayOf(0.001752f, -0.001752f, 0.002586f, 0.00636f, 0.005896f),
                floatArrayOf(0.011671f, -0.011671f, 0.017226f, 0.042355f, 0.039265f),
                floatArrayOf(-0.075195f, 0.075195f, -0.110982f, -0.272891f, -0.252981f),
                floatArrayOf(0.008445f, -0.008445f, 0.012464f, 0.030647f, 0.028411f)
              )),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the inputArray") {
          assertTrue {
            layer.inputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.177606f, 0.379355f, -0.085751f, 0.080693f)),
              tolerance = 1.0e-06f)
          }
        }
      }

      context("with next state only") {

        val layer = DeltaRNNLayerStructureUtils.buildLayer(DeltaLayersWindow.Front)

        layer.forward()

        layer.outputArray.assignErrors(layer.outputArray.values.sub(DeltaRNNLayerStructureUtils.getOutputGold()))
        val paramsErrors = layer.backward(propagateToInput = true)

        val params = layer.params

        it("should match the expected errors of the outputArray") {
          assertTrue {
            layer.outputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.075296f, -0.403656f, -0.191953f, -0.383426f, -0.699093f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the candidate array") {
          assertTrue {
            layer.candidate.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.026478f, -0.056861f, -0.108053f, -0.072316f, -0.335333f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the partition array") {
          assertTrue {
            layer.partition.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.010693f, -0.023303f, 0.016893f, -0.05453f, 0.07534f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the candidate biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.feedforwardUnit.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.026478f, -0.056861f, -0.108053f, -0.072316f, -0.335333f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the partition biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.recurrentUnit.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.010693f, -0.023303f, 0.016893f, -0.05453f, 0.07534f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the alpha array") {
          assertTrue {
            paramsErrors.getErrorsOf(params.alpha)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the beta1 array") {
          assertTrue {
            paramsErrors.getErrorsOf(params.beta1)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.021712f, 0.061979f, -0.032416f, -0.005062f, -0.07042f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the beta2 array") {
          assertTrue {
            paramsErrors.getErrorsOf(params.beta2)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.feedforwardUnit.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(0.0022f, 0.002475f, 0.002475f, -0.00275f),
                floatArrayOf(0.000447f, 0.000503f, 0.000503f, -0.000558f),
                floatArrayOf(-0.048091f, -0.054102f, -0.054102f, 0.060114f),
                floatArrayOf(0.020483f, 0.023044f, 0.023044f, -0.025604f),
                floatArrayOf(-0.167578f, -0.188526f, -0.188526f, 0.209473f)
              )),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the recurrent weights") {
          assertNull(paramsErrors.getErrorsOf(params.recurrentUnit.weights))
        }

        it("should match the expected errors of the inputArray") {
          assertTrue {
            layer.inputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.10362f, 0.18901f, -0.126453f, 0.202292f)),
              tolerance = 1.0e-06f)
          }
        }
      }

      context("with previous and next state") {

        val layer = DeltaRNNLayerStructureUtils.buildLayer(DeltaLayersWindow.Bilateral)

        layer.forward()

        layer.outputArray.assignErrors(layer.outputArray.values.sub(DeltaRNNLayerStructureUtils.getOutputGold()))
        val paramsErrors = layer.backward(propagateToInput = true)

        val params = layer.params

        it("should match the expected errors of the outputArray") {
          assertTrue {
            layer.outputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.160599f, -0.233533f, -0.17643f, -0.841042f, -0.745151f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the candidate array") {
          assertTrue {
            layer.candidate.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.055764f, -0.033038f, -0.11204f, -0.237697f, -0.367937f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the partition array") {
          assertTrue {
            layer.partition.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.031017f, -0.006853f, -0.002682f, -0.198978f, -0.043151f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the candidate biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.feedforwardUnit.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.055764f, -0.033038f, -0.11204f, -0.237697f, -0.367937f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the partition biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.recurrentUnit.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.031017f, -0.006853f, -0.002682f, -0.198978f, -0.043151f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the alpha array") {
          assertTrue {
            paramsErrors.getErrorsOf(params.alpha)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.050573f, -0.001492f, -0.005966f, -0.008471f, -0.021221f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the beta1 array") {
          assertTrue {
            paramsErrors.getErrorsOf(params.beta1)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.045727f, 0.036012f, -0.033612f, -0.016639f, -0.077267f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the beta2 array") {
          assertTrue {
            paramsErrors.getErrorsOf(params.beta2)!!.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.061674f, 0.001368f, -0.019886f, -0.121011f, -0.10105f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.feedforwardUnit.weights)!!.values).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(-0.013239f, -0.014894f, -0.014894f, 0.016549f),
                floatArrayOf(-0.004762f, -0.005357f, -0.005357f, 0.005952f),
                floatArrayOf(-0.028934f, -0.032551f, -0.032551f, 0.036168f),
                floatArrayOf(0.121843f, 0.137073f, 0.137073f, -0.152304f),
                floatArrayOf(-0.075135f, -0.084527f, -0.084527f, 0.093919f)
              )),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the recurrent weights") {
          val wRec: DenseNDArray = paramsErrors.getErrorsOf(params.recurrentUnit.weights)!!.values as DenseNDArray
          assertTrue {
            wRec.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(0.00011f, -0.00011f, 0.000162f, 0.000399f, 0.000370f),
                floatArrayOf(0.000828f, -0.000828f, 0.001222f, 0.003005f, 0.002786f),
                floatArrayOf(0.024104f, -0.024104f, 0.035576f, 0.087477f, 0.081094f),
                floatArrayOf(-0.036219f, 0.036219f, -0.053457f, -0.131442f, -0.121852f),
                floatArrayOf(0.008787f, -0.008787f, 0.012969f, 0.03189f, 0.029563f)
              )),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the inputArray") {
          assertTrue {
            layer.inputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.046517f, 0.213223f, -0.067537f, 0.093758f)),
              tolerance = 1.0e-06f)
          }
        }
      }
    }
  }
})
