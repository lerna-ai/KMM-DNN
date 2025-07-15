/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.attention

import com.kotlinnlp.simplednn.core.functionalities.initializers.RandomInitializer
import com.kotlinnlp.simplednn.core.functionalities.randomgenerators.RandomGenerator
import com.kotlinnlp.simplednn.core.layers.models.attention.attentionmechanism.AttentionMechanismLayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.mockito.kotlin.mock
import org.mockito.kotlin.whenever
import org.junit.runner.RunWith
import org.mockito.junit.MockitoJUnitRunner
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertTrue

/**
 *
 */
class AttentionParametersSpec : Spek({

  describe("an AttentionParameters") {

    context("initialization") {

      val randomGenerator = mock<RandomGenerator>()
      var i = 0.0f
      whenever(randomGenerator.next()).thenAnswer { i++ }

      val params = AttentionMechanismLayerParameters(inputSize = 2, weightsInitializer = RandomInitializer(randomGenerator))

      it("should have a context vector with the expected initialized values") {
        assertTrue {
          params.contextVector.values.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(0.0f, 1.0f)),
            tolerance = 1.0e-06f
          )
        }
      }
    }
  }
})
