/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.attention

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
              AugmentedArray(if (i == 1) DenseNDArrayFactory.arrayOf(doubleArrayOf(1.0, 0.1, 0.3)) else elm)
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
              AugmentedArray(if (i == 1) DenseNDArrayFactory.arrayOf(doubleArrayOf(1.0, 0.1, 0.3)) else elm)
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
              attentionSequence[0].toDoubleArray(),
              attentionSequence[1].toDoubleArray(),
              attentionSequence[2].toDoubleArray()
            )),
            tolerance = 1.0e-06
          )
        }
      }

      structure.attentionMatrix.assignErrors(DenseNDArrayFactory.arrayOf(listOf(
        doubleArrayOf(0.1, 0.2),
        doubleArrayOf(0.3, 0.4),
        doubleArrayOf(0.5, 0.6)
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
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.304352, 0.348001, 0.347647)),
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected output array") {
        assertTrue {
          structure.outputArray.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.191447, 0.282824, 0.030317, 0.530541)),
            tolerance = 1.0e-06)
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
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.027623, -0.046039)),
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the second attention array") {
        assertTrue {
          attentionErrors[1].equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.006529, -0.010882)),
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the third attention array") {
        assertTrue {
          attentionErrors[2].equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.034152, 0.056921)),
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the first input") {
        assertTrue {
          structure.inputArrays[0].errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.06087, 0.152176, 0.030435, -0.152176)),
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the second input") {
        assertTrue {
          structure.inputArrays[1].errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.0696, 0.174, 0.0348, -0.174)),
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the third input") {
        assertTrue {
          structure.inputArrays[2].errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.069529, 0.173823, 0.034765, -0.173823)),
            tolerance = 1.0e-06)
        }
      }
    }
  }
})
