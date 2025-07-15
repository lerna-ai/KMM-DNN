/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.recurrent.lstm

import com.kotlinnlp.simplednn.core.functionalities.initializers.ConstantInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.RandomInitializer
import com.kotlinnlp.simplednn.core.functionalities.randomgenerators.RandomGenerator
import com.kotlinnlp.simplednn.core.layers.models.recurrent.lstm.LSTMLayerParameters
import org.mockito.kotlin.mock
import org.mockito.kotlin.whenever
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.*

/**
 *
 */
class LSTMLayerParametersSpec : Spek({

  describe("a LSTMLayerParameters") {

    context("initialization") {

      context("dense input") {

        var k = 0
        val initValues = floatArrayOf(
          0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f,
          0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f,
          1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f,
          1.9f, 2.0f, 2.1f, 2.2f, 2.3f, 2.4f,
          2.5f, 2.6f, 2.7f, 2.8f, 2.9f, 3.0f,
          3.1f, 3.2f, 3.3f, 3.4f, 3.5f, 3.6f,
          3.7f, 3.8f, 3.9f, 4.0f)
        val randomGenerator = mock<RandomGenerator>()
        whenever(randomGenerator.next()).thenAnswer { initValues[k++] }

        val params = LSTMLayerParameters(
          inputSize = 3,
          outputSize = 2,
          weightsInitializer = RandomInitializer(randomGenerator),
          biasesInitializer = ConstantInitializer(0.9f))

        val wIn = params.inputGate.weights.values
        val wOut = params.outputGate.weights.values
        val wFor = params.forgetGate.weights.values
        val wC = params.candidate.weights.values

        val bIn = params.inputGate.biases.values
        val bOut = params.outputGate.biases.values
        val bFor = params.forgetGate.biases.values
        val bC = params.candidate.biases.values

        val wInr = params.inputGate.recurrentWeights.values
        val wOutr = params.outputGate.recurrentWeights.values
        val wForr = params.forgetGate.recurrentWeights.values
        val wCr = params.candidate.recurrentWeights.values

        it("should contain the expected initialized weights of the input gate") {
          (0 until wIn.length).forEach { i -> assertEquals(initValues[i], wIn[i]) }
        }

        it("should contain the expected initialized weights of the output gate") {
          (0 until wOut.length).forEach { i -> assertEquals(initValues[i + 6], wOut[i]) }
        }

        it("should contain the expected initialized weights of the forget gate") {
          (0 until wFor.length).forEach { i -> assertEquals(initValues[i + 12], wFor[i]) }
        }

        it("should contain the expected initialized weights of the candidate") {
          (0 until wC.length).forEach { i -> assertEquals(initValues[i + 18], wC[i]) }
        }

        it("should contain the expected initialized biases of the input gate") {
          (0 until bIn.length).forEach { i -> assertEquals(0.9f, bIn[i]) }
        }

        it("should contain the expected initialized biases of the forget gate") {
          (0 until bFor.length).forEach { i -> assertEquals(0.9f, bFor[i]) }
        }

        it("should contain the expected initialized biases of the output gate") {
          (0 until bOut.length).forEach { i -> assertEquals(0.9f, bOut[i]) }
        }

        it("should contain the expected initialized biases of the candidate") {
          (0 until bC.length).forEach { i -> assertEquals(0.9f, bC[i]) }
        }

        it("should contain the expected initialized recurrent weights of the input gate") {
          (0 until wInr.length).forEach { i -> assertEquals(initValues[i + 24], wInr[i]) }
        }

        it("should contain the expected initialized recurrent weights of the output gate") {
          (0 until wOutr.length).forEach { i -> assertEquals(initValues[i + 28], wOutr[i]) }
        }

        it("should contain the expected initialized recurrent weights of the forget gate") {
          (0 until wForr.length).forEach { i -> assertEquals(initValues[i + 32], wForr[i]) }
        }

        it("should contain the expected initialized recurrent weights of the candidate") {
          (0 until wCr.length).forEach { i -> assertEquals(initValues[i + 36], wCr[i]) }
        }
      }
    }
  }
})
