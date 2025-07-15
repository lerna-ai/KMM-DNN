/* Copyright 2020-present Simone Cangialosi. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.attention.scaleddot

import com.kotlinnlp.simplednn.core.layers.models.attention.scaleddot.ScaledDotAttentionLayer
import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsList
import com.kotlinnlp.simplednn.core.optimizer.getErrorsOf
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.core.attention.scaleddot.ScaledDotAttentionLayerUtils
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertFails
import kotlin.test.assertTrue

/**
 *
 */
class ScaledDotAttentionLayerSpec : Spek({

  describe("a ScaledDotAttentionLayer") {

    context("wrong initialization") {

      it("should raise an Exception with an empty input sequence") {
        assertFails {
          ScaledDotAttentionLayer(
            inputArrays = mutableListOf(),
            params = ScaledDotAttentionLayerUtils.buildAttentionParams()
          )
        }
      }
    }

    context("forward") {

      val inputSequence = ScaledDotAttentionLayerUtils.buildInputSequence()
      val layer = ScaledDotAttentionLayer(
        inputArrays = inputSequence,
        params = ScaledDotAttentionLayerUtils.buildAttentionParams()
      )

      layer.forward()

      it("should match the expected queries") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(listOf(
            floatArrayOf(0.92f, 1.1f),
            floatArrayOf(0.53f, 1.04f),
            floatArrayOf(0.55f, 1.03f)
          )).equals(
            layer.queries.values,
            tolerance = 1.0e-06f)
        }
      }

      it("should match the expected keys") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(listOf(
            floatArrayOf(0.96f, 0.02f),
            floatArrayOf(0.18f, -0.12f),
            floatArrayOf(-1.0f, -0.56f)
          )).equals(
            layer.keys.values,
            tolerance = 1.0e-06f)
        }
      }

      it("should match the expected values") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(listOf(
            floatArrayOf(0.63f, 0.5f, -0.95f, 0.32f),
            floatArrayOf(-0.2f, 0.0f, -0.13f, -1.18f),
            floatArrayOf(-0.27f, -0.2f, -0.41f, 1.4f)
          )).equals(
            layer.values.values,
            tolerance = 1.0e-06f)
        }
      }

      it("should match the expected attention scores") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(listOf(
            floatArrayOf(0.582109f, 0.314298f, 0.103593f),
            floatArrayOf(0.503361f, 0.339012f, 0.157628f),
            floatArrayOf(0.506943f, 0.338013f, 0.155044f)
          )).equals(
            DenseNDArrayFactory.fromRows(layer.attentionAct),
            tolerance = 1.0e-06f)
        }
      }

      it("should match the expected output arrays") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(listOf(
            floatArrayOf(0.275899f, 0.270336f, -0.636335f, -0.039567f),
            floatArrayOf(0.206755f, 0.220155f, -0.586891f, -0.018279f),
            floatArrayOf(0.209909f, 0.222462f, -0.589105f, -0.019572f)
          )).equals(
            DenseNDArrayFactory.fromRows(layer.outputArrays.map { it.values }),
            tolerance = 1.0e-06f)
        }
      }
    }

    context("backward") {

      val inputSequence = ScaledDotAttentionLayerUtils.buildInputSequence()
      val layer = ScaledDotAttentionLayer(
        inputArrays = inputSequence,
        params = ScaledDotAttentionLayerUtils.buildAttentionParams()
      )

      layer.forward()

      layer.outputArrays.zip(ScaledDotAttentionLayerUtils.buildOutputErrors()).forEach { (array, errors) ->
        array.assignErrors(errors)
      }

      val paramsErrors: ParamsErrorsList = layer.backward(propagateToInput = true)

      it("should match the expected errors of the queries") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(listOf(
            floatArrayOf(0.166976f, 0.04868f),
            floatArrayOf(-0.157867f, -0.044738f),
            floatArrayOf(-0.118836f, -0.00275f)
          )).equals(layer.queries.errors, tolerance = 1.0e-06f)
        }
      }

      it("should match the expected errors of the keys") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(listOf(
            floatArrayOf(-0.118337f, -0.282144f),
            floatArrayOf(0.200448f, 0.381435f),
            floatArrayOf(-0.082111f, -0.099291f)
          )).equals(layer.keys.errors, tolerance = 1.0e-06f)
        }
      }

      it("should match the expected errors of the values") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(listOf(
            floatArrayOf(-0.299378f, -0.679784f, -0.557768f, -0.696967f),
            floatArrayOf(-0.254008f, -0.432802f, -0.321912f, -0.42746f),
            floatArrayOf(-0.146614f, -0.187414f, -0.12032f, -0.175574f)
          )).equals(layer.values.errors, tolerance = 1.0e-06f)
        }
      }

      it("should match the expected errors of the queries weights") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(listOf(
            floatArrayOf(-0.023508f, 0.065794f, 0.260786f, -0.049875f),
            floatArrayOf(0.002072f, 0.004134f, 0.075128f, -0.007132f)
          )).equals(paramsErrors.getErrorsOf(layer.params.queries.weights)!!.values, tolerance = 1.0e-06f)
        }
      }

      it("should match the expected errors of the keys weights") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(listOf(
            floatArrayOf(-0.077523f, 0.098533f, -0.246816f, 0.072934f),
            floatArrayOf(-0.107647f, 0.119149f, -0.520935f, 0.116003f)
          )).equals(paramsErrors.getErrorsOf(layer.params.keys.weights)!!.values, tolerance = 1.0e-06f)
        }
      }

      it("should match the expected errors of the values weights") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(listOf(
            floatArrayOf(0.202771f, -0.314063f, -0.091634f, -0.412156f),
            floatArrayOf(0.43209f, -0.685103f, -0.308845f, -0.791595f),
            floatArrayOf(0.347967f, -0.555616f, -0.276653f, -0.616254f),
            floatArrayOf(0.439844f, -0.699312f, -0.328048f, -0.795263f)
          )).equals(paramsErrors.getErrorsOf(layer.params.values.weights)!!.values, tolerance = 1.0e-06f)
        }
      }

      it("should match the expected errors of the queries biases") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(floatArrayOf(-0.109727f, 0.001192f))
            .equals(paramsErrors.getErrorsOf(layer.params.queries.biases)!!.values, tolerance = 1.0e-06f)
        }
      }

      it("should match the expected errors of the keys biases") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(floatArrayOf(0.0f, 0.0f))
            .equals(paramsErrors.getErrorsOf(layer.params.keys.biases)!!.values, tolerance = 1.0e-06f)
        }
      }

      it("should match the expected errors of the values biases") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(floatArrayOf(-0.7f, -1.3f, -1.0f, -1.3f))
            .equals(paramsErrors.getErrorsOf(layer.params.values.biases)!!.values, tolerance = 1.0e-06f)
        }
      }

      it("should match the expected errors of the first input") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(floatArrayOf(-0.057415f, 0.586003f, -0.788808f, -0.138081f))
            .equals(layer.inputArrays[0].errors, tolerance = 1.0e-06f)
        }
      }

      it("should match the expected errors of the second input") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(floatArrayOf(-0.137857f, 0.857493f, -0.429106f, -0.207021f))
            .equals(layer.inputArrays[1].errors, tolerance = 1.0e-06f)
        }
      }

      it("should match the expected errors of the third input") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(floatArrayOf(-0.127408f, -0.047507f, -0.313912f, -0.052238f))
            .equals(layer.inputArrays[2].errors, tolerance = 1.0e-06f)
        }
      }
    }

    context("forward with dropout") {

      val inputSequence = ScaledDotAttentionLayerUtils.buildInputSequence()
      val layer = ScaledDotAttentionLayer(
        inputArrays = inputSequence,
        attentionDropout = 1.0e-12f, // activate the dropout actually without dropping (very low probability)
        params = ScaledDotAttentionLayerUtils.buildAttentionParams()
      )

      layer.forward()

      it("should match the expected queries") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(listOf(
            floatArrayOf(0.92f, 1.1f),
            floatArrayOf(0.53f, 1.04f),
            floatArrayOf(0.55f, 1.03f)
          )).equals(
            layer.queries.values,
            tolerance = 1.0e-06f)
        }
      }

      it("should match the expected keys") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(listOf(
            floatArrayOf(0.96f, 0.02f),
            floatArrayOf(0.18f, -0.12f),
            floatArrayOf(-1.0f, -0.56f)
          )).equals(
            layer.keys.values,
            tolerance = 1.0e-06f)
        }
      }

      it("should match the expected values") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(listOf(
            floatArrayOf(0.63f, 0.5f, -0.95f, 0.32f),
            floatArrayOf(-0.2f, 0.0f, -0.13f, -1.18f),
            floatArrayOf(-0.27f, -0.2f, -0.41f, 1.4f)
          )).equals(
            layer.values.values,
            tolerance = 1.0e-06f)
        }
      }

      it("should match the expected attention scores") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(listOf(
            floatArrayOf(0.582109f, 0.314298f, 0.103593f),
            floatArrayOf(0.503361f, 0.339012f, 0.157628f),
            floatArrayOf(0.506943f, 0.338013f, 0.155044f)
          )).equals(
            DenseNDArrayFactory.fromRows(layer.attentionAct),
            tolerance = 1.0e-06f)
        }
      }

      it("should match the expected output arrays") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(listOf(
            floatArrayOf(0.275899f, 0.270336f, -0.636335f, -0.039567f),
            floatArrayOf(0.206755f, 0.220155f, -0.586891f, -0.018279f),
            floatArrayOf(0.209909f, 0.222462f, -0.589105f, -0.019572f)
          )).equals(
            DenseNDArrayFactory.fromRows(layer.outputArrays.map { it.values }),
            tolerance = 1.0e-06f)
        }
      }
    }

    context("backward with dropout") {

      val inputSequence = ScaledDotAttentionLayerUtils.buildInputSequence()
      val layer = ScaledDotAttentionLayer(
        inputArrays = inputSequence,
        attentionDropout = 1.0e-12f, // activate the dropout actually without dropping (very low probability)
        params = ScaledDotAttentionLayerUtils.buildAttentionParams()
      )

      layer.forward()

      layer.outputArrays.zip(ScaledDotAttentionLayerUtils.buildOutputErrors()).forEach { (array, errors) ->
        array.assignErrors(errors)
      }

      val paramsErrors: ParamsErrorsList = layer.backward(propagateToInput = true)

      it("should match the expected errors of the queries") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(listOf(
            floatArrayOf(0.166976f, 0.04868f),
            floatArrayOf(-0.157867f, -0.044738f),
            floatArrayOf(-0.118836f, -0.00275f)
          )).equals(layer.queries.errors, tolerance = 1.0e-06f)
        }
      }

      it("should match the expected errors of the keys") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(listOf(
            floatArrayOf(-0.118337f, -0.282144f),
            floatArrayOf(0.200448f, 0.381435f),
            floatArrayOf(-0.082111f, -0.099291f)
          )).equals(layer.keys.errors, tolerance = 1.0e-06f)
        }
      }

      it("should match the expected errors of the values") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(listOf(
            floatArrayOf(-0.299378f, -0.679784f, -0.557768f, -0.696967f),
            floatArrayOf(-0.254008f, -0.432802f, -0.321912f, -0.42746f),
            floatArrayOf(-0.146614f, -0.187414f, -0.12032f, -0.175574f)
          )).equals(layer.values.errors, tolerance = 1.0e-06f)
        }
      }

      it("should match the expected errors of the queries weights") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(listOf(
            floatArrayOf(-0.023508f, 0.065794f, 0.260786f, -0.049875f),
            floatArrayOf(0.002072f, 0.004134f, 0.075128f, -0.007132f)
          )).equals(paramsErrors.getErrorsOf(layer.params.queries.weights)!!.values, tolerance = 1.0e-06f)
        }
      }

      it("should match the expected errors of the keys weights") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(listOf(
            floatArrayOf(-0.077523f, 0.098533f, -0.246816f, 0.072934f),
            floatArrayOf(-0.107647f, 0.119149f, -0.520935f, 0.116003f)
          )).equals(paramsErrors.getErrorsOf(layer.params.keys.weights)!!.values, tolerance = 1.0e-06f)
        }
      }

      it("should match the expected errors of the values weights") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(listOf(
            floatArrayOf(0.202771f, -0.314063f, -0.091634f, -0.412156f),
            floatArrayOf(0.43209f, -0.685103f, -0.308845f, -0.791595f),
            floatArrayOf(0.347967f, -0.555616f, -0.276653f, -0.616254f),
            floatArrayOf(0.439844f, -0.699312f, -0.328048f, -0.795263f)
          )).equals(paramsErrors.getErrorsOf(layer.params.values.weights)!!.values, tolerance = 1.0e-06f)
        }
      }

      it("should match the expected errors of the queries biases") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(floatArrayOf(-0.109727f, 0.001192f))
            .equals(paramsErrors.getErrorsOf(layer.params.queries.biases)!!.values, tolerance = 1.0e-06f)
        }
      }

      it("should match the expected errors of the keys biases") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(floatArrayOf(0.0f, 0.0f))
            .equals(paramsErrors.getErrorsOf(layer.params.keys.biases)!!.values, tolerance = 1.0e-06f)
        }
      }

      it("should match the expected errors of the values biases") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(floatArrayOf(-0.7f, -1.3f, -1.0f, -1.3f))
            .equals(paramsErrors.getErrorsOf(layer.params.values.biases)!!.values, tolerance = 1.0e-06f)
        }
      }

      it("should match the expected errors of the first input") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(floatArrayOf(-0.057415f, 0.586003f, -0.788808f, -0.138081f))
            .equals(layer.inputArrays[0].errors, tolerance = 1.0e-06f)
        }
      }

      it("should match the expected errors of the second input") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(floatArrayOf(-0.137857f, 0.857493f, -0.429106f, -0.207021f))
            .equals(layer.inputArrays[1].errors, tolerance = 1.0e-06f)
        }
      }

      it("should match the expected errors of the third input") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(floatArrayOf(-0.127408f, -0.047507f, -0.313912f, -0.052238f))
            .equals(layer.inputArrays[2].errors, tolerance = 1.0e-06f)
        }
      }
    }
  }
})
