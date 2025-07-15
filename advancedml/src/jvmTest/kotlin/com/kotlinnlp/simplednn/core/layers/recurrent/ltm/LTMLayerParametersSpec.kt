/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.recurrent.ltm

import com.kotlinnlp.simplednn.core.functionalities.initializers.RandomInitializer
import com.kotlinnlp.simplednn.core.functionalities.randomgenerators.RandomGenerator
import com.kotlinnlp.simplednn.core.layers.models.recurrent.ltm.LTMLayerParameters
import org.mockito.kotlin.mock
import org.mockito.kotlin.whenever
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.*

/**
 *
 */
class LTMLayerParametersSpec : Spek({

  describe("a LTMLayerParameters") {

    context("initialization") {

      context("dense input") {

        var k = 0
        val initValues = floatArrayOf(
          0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f,
          0.9f, 1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f,
          1.7f, 1.8f, 1.9f, 2.0f, 2.1f, 2.2f, 2.3f, 2.4f,
          2.5f, 2.6f, 2.7f, 2.8f, 2.9f, 3.0f, 3.1f, 3.2f,
          3.3f, 3.4f, 3.5f, 3.6f)
        val randomGenerator = mock<RandomGenerator>()
        whenever(randomGenerator.next()).thenAnswer { initValues[k++] }

        val params = LTMLayerParameters(inputSize = 3, weightsInitializer = RandomInitializer(randomGenerator))

        val w1 = params.inputGate1.weights.values
        val w2 = params.inputGate2.weights.values
        val w3 = params.inputGate3.weights.values
        val wCell = params.cell.weights.values

        it("should contain the expected initialized weights of the input gate L1") {
          (0 until w1.length).forEach { i -> assertEquals(initValues[i], w1[i]) }
        }

        it("should contain the expected initialized weights of the input gate L2") {
          (0 until w2.length).forEach { i -> assertEquals(initValues[i + 9], w2[i]) }
        }

        it("should contain the expected initialized weights of the input gate L3") {
          (0 until w3.length).forEach { i -> assertEquals(initValues[i + 18], w3[i]) }
        }

        it("should contain the expected initialized weights of the cell") {
          (0 until wCell.length).forEach { i -> assertEquals(initValues[i + 27], wCell[i]) }
        }
      }
    }
  }
})
