/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.updatemethods

import com.kotlinnlp.simplednn.core.arrays.ParamsArray
import com.kotlinnlp.simplednn.core.functionalities.decaymethods.DecayMethod
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.momentum.MomentumMethod
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.mockito.kotlin.any
import org.mockito.kotlin.eq
import org.mockito.kotlin.mock
import org.mockito.kotlin.whenever
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 *
 */
class MomentumSpec: Spek({

  describe("the Momentum update method") {

    context("update with dense errors") {

      context("update") {

        val updateHelper = MomentumMethod(learningRate = 0.001f, momentum = 0.9f)
        val updatableArray: ParamsArray = UpdateMethodsUtils.buildParamsArray()
        val supportStructure = updateHelper.getSupportStructure(updatableArray)

        supportStructure.v.assignValues(UpdateMethodsUtils.supportArray1())

        updateHelper.update(array = updatableArray, errors = UpdateMethodsUtils.buildDenseErrors())

        it("should match the expected updated array") {
          assertTrue {
            updatableArray.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.2309f, -0.3207f, 0.0496f, 0.7292f, 0.6199f)),
              tolerance = 1.0e-6f)
          }
        }
      }
    }

    context("update with sparse errors") {

      context("update") {

        val updateHelper = MomentumMethod(learningRate = 0.001f, momentum = 0.9f)
        val updatableArray: ParamsArray = UpdateMethodsUtils.buildParamsArray()
        val supportStructure = updateHelper.getSupportStructure(updatableArray)

        supportStructure.v.assignValues(UpdateMethodsUtils.supportArray1())

        updateHelper.update(array = updatableArray, errors = UpdateMethodsUtils.buildSparseErrors())

        it("should match the expected updated array") {
          assertTrue {
            updatableArray.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.23f, -0.3207f, 0.05f, 0.73f, 0.6197f)),
              tolerance = 1.0e-5f)
          }
        }
      }
    }

    context("epoch scheduling") {

      val decayMethod = mock<DecayMethod>()
      whenever(decayMethod.update(learningRate = any(), timeStep = eq(1))).thenReturn(0.03f)
      whenever(decayMethod.update(learningRate = any(), timeStep = eq(2))).thenReturn(0.05f)

      context("first epoch") {

        val updateHelper = MomentumMethod(learningRate = 0.001f, decayMethod = decayMethod)

        updateHelper.newEpoch()

        it("should match the expected alpha in the first epoch") {
          assertEquals(0.03f, updateHelper.alpha)
        }
      }

      context("second epoch") {

        val updateHelper = MomentumMethod(learningRate = 0.001f, decayMethod = decayMethod)

        updateHelper.newEpoch()
        updateHelper.newEpoch()

        it("should match the expected alpha in the second epoch") {
          assertEquals(0.05f, updateHelper.alpha)
        }
      }
    }
  }
})
