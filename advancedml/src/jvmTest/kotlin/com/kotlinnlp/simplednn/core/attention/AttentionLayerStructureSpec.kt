/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.attention

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.models.attention.AttentionLayer
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertFails
import kotlin.test.assertTrue

/**
 *
 */
class AttentionLayerStructureSpec : Spek({

  describe("an AttentionLayerStructure") {

    val utils = AttentionLayerUtils

    context("wrong initialization") {

      val inputSequence: List<DenseNDArray> = utils.buildInputSequence()
      val attentionSequence: List<DenseNDArray> = utils.buildAttentionSequence(inputSequence)
      val params = utils.buildAttentionParams()

      it("should raise an Exception with an empty input sequence") {
        assertFails {
          AttentionLayer(
            inputArrays = mutableListOf<AugmentedArray<DenseNDArray>>(),
            inputType = LayerType.Input.Dense,
            attentionArrays = attentionSequence.map { AugmentedArray(it) },
            params = params)
        }
      }

      it("should raise an Exception with an empty attention sequence") {
        assertFails {
          AttentionLayer(
            inputArrays = inputSequence.map { AugmentedArray(it) },
            inputType = LayerType.Input.Dense,
            attentionArrays = mutableListOf(),
            params = params)
        }
      }

      it("should raise an Exception with input and attention sequences not compatible") {
        assertFails {
          AttentionLayer(
            inputArrays = inputSequence.map { AugmentedArray(it) },
            inputType = LayerType.Input.Dense,
            attentionArrays = attentionSequence.mapIndexed { i, elm ->
              AugmentedArray(if (i == 1) DenseNDArrayFactory.arrayOf(floatArrayOf(1.0f, 0.1f, 0.3f)) else elm)
            },
            params = params)
        }
      }

      it("should raise an Exception with a attention arrays with a not expected size") {
        assertFails {
          AttentionLayer(
            inputArrays = inputSequence.map { AugmentedArray(it) },
            inputType = LayerType.Input.Dense,
            attentionArrays = attentionSequence.mapIndexed { i, elm ->
              AugmentedArray(if (i == 1) DenseNDArrayFactory.arrayOf(floatArrayOf(1.0f, 0.1f, 0.3f)) else elm)
            },
            params = params)
        }
      }
    }

    context("correct initialization") {

      val inputSequence: List<DenseNDArray> = utils.buildInputSequence()
      val attentionSequence: List<DenseNDArray> = utils.buildAttentionSequence(inputSequence)
      val structure = AttentionLayer(
        inputArrays = inputSequence.map { AugmentedArray(it) },
        inputType = LayerType.Input.Dense,
        attentionArrays = attentionSequence.map { AugmentedArray(it) },
        params = utils.buildAttentionParams())

      it("should initialize the attention matrix correctly") {
        assertTrue {
          structure.attentionMatrix.values.equals(
            DenseNDArrayFactory.arrayOf(listOf(
              attentionSequence[0].toFloatArray(),
              attentionSequence[1].toFloatArray(),
              attentionSequence[2].toFloatArray()
            )),
            tolerance = 1.0e-06f
          )
        }
      }

      structure.attentionMatrix.assignErrors(DenseNDArrayFactory.arrayOf(listOf(
        floatArrayOf(0.1f, 0.2f),
        floatArrayOf(0.3f, 0.4f),
        floatArrayOf(0.5f, 0.6f)
      )))
    }

    context("forward") {

      val inputSequence: List<DenseNDArray> = utils.buildInputSequence()
      val attentionSequence: List<DenseNDArray> = utils.buildAttentionSequence(inputSequence)
      val structure = AttentionLayer(
        inputArrays = inputSequence.map { AugmentedArray(it) },
        inputType = LayerType.Input.Dense,
        attentionArrays = attentionSequence.map { AugmentedArray(it) },
        params = utils.buildAttentionParams())

      structure.forward()

      it("should match the expected attention scores") {
        assertTrue {
          structure.attentionScores.values.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(0.304352f, 0.348001f, 0.347647f)),
            tolerance = 1.0e-06f)
        }
      }

      it("should match the expected output array") {
        assertTrue {
          structure.outputArray.values.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(-0.191447f, 0.282824f, 0.030317f, 0.530541f)),
            tolerance = 1.0e-06f)
        }
      }
    }

    context("backward") {

      val inputSequence: List<DenseNDArray> = utils.buildInputSequence()
      val attentionSequence: List<DenseNDArray> = utils.buildAttentionSequence(inputSequence)
      val structure = AttentionLayer(
        inputArrays = inputSequence.map { AugmentedArray(it) },
        inputType = LayerType.Input.Dense,
        attentionArrays = attentionSequence.map { AugmentedArray(it) },
        params = utils.buildAttentionParams())

      structure.forward()

      structure.outputArray.assignErrors(utils.buildOutputErrors())
      structure.backward(propagateToInput = true)

      val attentionErrors: List<DenseNDArray> = structure.attentionArrays.map { it.errors }

      it("should match the expected errors of the first attention array") {
        assertTrue {
          attentionErrors[0].equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(-0.027623f, -0.046039f)),
            tolerance = 1.0e-06f)
        }
      }

      it("should match the expected errors of the second attention array") {
        assertTrue {
          attentionErrors[1].equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(-0.006529f, -0.010882f)),
            tolerance = 1.0e-06f)
        }
      }

      it("should match the expected errors of the third attention array") {
        assertTrue {
          attentionErrors[2].equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(0.034152f, 0.056921f)),
            tolerance = 1.0e-06f)
        }
      }

      it("should match the expected errors of the first input") {
        assertTrue {
          structure.inputArrays[0].errors.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(-0.06087f, 0.152176f, 0.030435f, -0.152176f)),
            tolerance = 1.0e-06f)
        }
      }

      it("should match the expected errors of the second input") {
        assertTrue {
          structure.inputArrays[1].errors.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(-0.0696f, 0.174f, 0.0348f, -0.174f)),
            tolerance = 1.0e-06f)
        }
      }

      it("should match the expected errors of the third input") {
        assertTrue {
          structure.inputArrays[2].errors.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(-0.069529f, 0.173823f, 0.034765f, -0.173823f)),
            tolerance = 1.0e-06f)
        }
      }
    }
  }
})
