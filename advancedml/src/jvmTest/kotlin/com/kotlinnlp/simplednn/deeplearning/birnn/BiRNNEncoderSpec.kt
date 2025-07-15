/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.birnn

import com.kotlinnlp.simplednn.core.layers.models.recurrent.simple.SimpleRecurrentLayerParameters
import com.kotlinnlp.simplednn.core.optimizer.getErrorsOf
import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNNEncoder
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.deeplearning.birnn.utils.BiRNNEncoderUtils
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertTrue

/**
 *
 */
class BiRNNEncoderSpec : Spek({

  describe("a BiRNNEncoder") {

    val inputSequence = BiRNNEncoderUtils.buildInputSequence()
    val birnn = BiRNNEncoderUtils.buildBiRNN()
    val encoder = BiRNNEncoder<DenseNDArray>(birnn, propagateToInput = true)

    val encodedSequence = encoder.forward(inputSequence)

    it("should match the expected first output array") {
      assertTrue {
        encodedSequence[0].equals(
          DenseNDArrayFactory.arrayOf(floatArrayOf(0.187746f, -0.50052f, 0.109558f, -0.005277f, -0.084306f, -0.628766f)),
          tolerance = 1.0e-06f
        )
      }
    }

    it("should match the expected second output array") {
      assertTrue {
        encodedSequence[1].equals(
          DenseNDArrayFactory.arrayOf(floatArrayOf(-0.704648f, 0.200908f, -0.064056f, -0.329084f, -0.237601f, -0.449676f)),
          tolerance = 1.0e-06f
        )
      }
    }

    it("should match the expected third output array") {
      assertTrue {
        encodedSequence[2].equals(
          DenseNDArrayFactory.arrayOf(floatArrayOf(0.256521f, 0.725227f, 0.781582f, 0.129273f, -0.716298f, -0.263625f)),
          tolerance = 1.0e-06f
        )
      }
    }

    encoder.backward(outputErrors = BiRNNEncoderUtils.buildOutputErrorsSequence())

    val paramsErrors = encoder.getParamsErrors()

    val l2rParams = birnn.leftToRightNetwork.paramsPerLayer[0] as SimpleRecurrentLayerParameters
    val r2lParams = birnn.rightToLeftNetwork.paramsPerLayer[0] as SimpleRecurrentLayerParameters

    it("should match the expected errors of the Left-to-right biases") {
      assertTrue {
        paramsErrors.getErrorsOf(l2rParams.unit.biases)!!.values.equals(
          DenseNDArrayFactory.arrayOf(floatArrayOf(-0.213048f, 0.804082f, 1.035058f)),
          tolerance = 1.0e-06f
        )
      }
    }

    it("should match the expected errors of the Left-to-right weights ") {
      assertTrue {
        (paramsErrors.getErrorsOf(l2rParams.unit.weights)!!.values as DenseNDArray).equals(
          DenseNDArrayFactory.arrayOf(listOf(
            floatArrayOf(0.003701f, -0.32396f),
            floatArrayOf(0.525116f, 0.047213f),
            floatArrayOf(0.640192f, -0.140151f)
          )),
          tolerance = 1.0e-06f
        )
      }
    }

    it("should match the expected errors of the Left-to-right recurrent weights") {
      assertTrue {
        paramsErrors.getErrorsOf(l2rParams.unit.recurrentWeights)!!.values.equals(
          DenseNDArrayFactory.arrayOf(listOf(
            floatArrayOf(0.125452f, -0.177722f, 0.040777f),
            floatArrayOf(0.126687f, -0.258213f, 0.057472f),
            floatArrayOf(0.105992f, -0.347851f, 0.07536f)
          )),
          tolerance = 1.0e-06f
        )
      }
    }

    it("should match the expected errors of the Right-to-left biases") {
      assertTrue {
        paramsErrors.getErrorsOf(r2lParams.unit.biases)!!.values.equals(
          DenseNDArrayFactory.arrayOf(floatArrayOf(-0.117179f, 0.712793f, -0.413573f)),
          tolerance = 1.0e-06f
        )
      }
    }

    it("should match the expected errors of the Right-to-left weights") {
      assertTrue {
        (paramsErrors.getErrorsOf(r2lParams.unit.weights)!!.values as DenseNDArray).equals(
          DenseNDArrayFactory.arrayOf(listOf(
            floatArrayOf(0.43714f, 0.703645f),
            floatArrayOf(0.150406f, 0.212304f),
            floatArrayOf(-0.18375f, -0.051842f)
          )),
          tolerance = 1.0e-06f
        )
      }
    }

    it("should match the expected errors of the Right-to-left recurrent weights") {
      assertTrue {
        paramsErrors.getErrorsOf(r2lParams.unit.recurrentWeights)!!.values.equals(
          DenseNDArrayFactory.arrayOf(listOf(
            floatArrayOf(-0.087835f, -0.337705f, -0.269176f),
            floatArrayOf(-0.223279f, 0.009348f, -0.212353f),
            floatArrayOf(0.067992f, 0.121748f, 0.132418f)
          )),
          tolerance = 1.0e-06f
        )
      }
    }

    val inputErrors: List<DenseNDArray> = encoder.getInputErrors()

    it("should match the expected errors of first input array") {
      assertTrue {
        inputErrors[0].equals(
          DenseNDArrayFactory.arrayOf(floatArrayOf(1.031472f, -0.627913f)),
          tolerance = 1.0e-06f
        )
      }
    }

    it("should match the expected errors of second input array") {
      assertTrue {
        inputErrors[1].equals(
          DenseNDArrayFactory.arrayOf(floatArrayOf(-0.539497f, -0.629167f)),
          tolerance = 1.0e-06f
        )
      }
    }

    it("should match the expected errors of third input array") {
      assertTrue {
        inputErrors[2].equals(
          DenseNDArrayFactory.arrayOf(floatArrayOf(0.013097f, -0.09932f)),
          tolerance = 1.0e-06f
        )
      }
    }
  }
})
