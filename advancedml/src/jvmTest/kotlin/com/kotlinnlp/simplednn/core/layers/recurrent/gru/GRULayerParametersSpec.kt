/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.recurrent.gru

import com.kotlinnlp.simplednn.core.functionalities.initializers.ConstantInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.RandomInitializer
import com.kotlinnlp.simplednn.core.functionalities.randomgenerators.RandomGenerator
import com.kotlinnlp.simplednn.core.layers.models.recurrent.gru.GRULayerParameters
import org.mockito.kotlin.mock
import org.mockito.kotlin.whenever
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.*

/**
 *
 */
class GRULayerParametersSpec : Spek({

  describe("a GRULayerParameters") {

    context("initialization") {

      context("dense input") {

        var k = 0
        val initValues = floatArrayOf(
          0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f,
          0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f,
          1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f,
          1.9f, 2.0f, 2.1f, 2.2f, 2.3f, 2.4f,
          2.5f, 2.6f, 2.7f, 2.8f, 2.9f, 3.0f)
        val randomGenerator = mock<RandomGenerator>()
        whenever(randomGenerator.next()).thenAnswer { initValues[k++] }

        val params = GRULayerParameters(
          inputSize = 3,
          outputSize = 2,
          weightsInitializer = RandomInitializer(randomGenerator),
          biasesInitializer = ConstantInitializer(0.9f))

        val wc = params.candidate.weights.values
        val wr = params.resetGate.weights.values
        val wp = params.partitionGate.weights.values

        val bc = params.candidate.biases.values
        val br = params.resetGate.biases.values
        val bp = params.partitionGate.biases.values

        val wcr = params.candidate.recurrentWeights.values
        val wrr = params.resetGate.recurrentWeights.values
        val wpr = params.partitionGate.recurrentWeights.values

        it("should contain the expected initialized weights of the candidate") {
          (0 until wc.length).forEach { i -> assertEquals(initValues[i], wc[i]) }
        }

        it("should contain the expected initialized weights of the reset gate") {
          (0 until wr.length).forEach { i -> assertEquals(initValues[i + 6], wr[i]) }
        }

        it("should contain the expected initialized weights of the partition gate") {
          (0 until wp.length).forEach { i -> assertEquals(initValues[i + 12], wp[i]) }
        }

        it("should contain the expected initialized biases of the candidate") {
          (0 until bc.length).forEach { i -> assertEquals(0.9f, bc[i]) }
        }

        it("should contain the expected initialized biases of the reset gate") {
          (0 until br.length).forEach { i -> assertEquals(0.9f, br[i]) }
        }

        it("should contain the expected initialized biases of the partition gate") {
          (0 until bp.length).forEach { i -> assertEquals(0.9f, bp[i]) }
        }

        it("should contain the expected initialized recurrent weights of the candidate") {
          (0 until wcr.length).forEach { i -> assertEquals(initValues[i + 18], wcr[i]) }
        }

        it("should contain the expected initialized recurrent weights of the candidate") {
          (0 until wrr.length).forEach { i -> assertEquals(initValues[i + 22], wrr[i]) }
        }

        it("should contain the expected initialized recurrent weights of the candidate") {
          (0 until wpr.length).forEach { i -> assertEquals(initValues[i + 26], wpr[i]) }
        }
      }
    }
  }
})
