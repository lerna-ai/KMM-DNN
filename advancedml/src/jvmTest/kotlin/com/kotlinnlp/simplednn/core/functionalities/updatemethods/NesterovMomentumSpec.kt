/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.updatemethods

import com.kotlinnlp.simplednn.core.arrays.ParamsArray
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.nesterovmomentum.NesterovMomentumMethod
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertTrue

/**
 *
 */
class NesterovMomentumSpec: Spek({

  describe("the NesterovMomentum update method") {

    context("update with dense errors") {

      context("update") {

        val updateHelper = NesterovMomentumMethod(learningRate = 0.001f, momentum = 0.9f)
        val updatableArray: ParamsArray = UpdateMethodsUtils.buildParamsArray()
        val supportStructure = updateHelper.getSupportStructure(updatableArray)

        supportStructure.v.assignValues(UpdateMethodsUtils.supportArray1())

        updateHelper.update(array = updatableArray, errors = UpdateMethodsUtils.buildDenseErrors())

        it("should match the expected updated array") {
          assertTrue {
            updatableArray.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.16871f, -0.24933f, 0.09424f, 0.75548f, 0.63781f)),
              tolerance = 1.0e-6f)
          }
        }
      }
    }

    context("update with sparse errors") {

      context("update") {

        val updateHelper = NesterovMomentumMethod(learningRate = 0.001f, momentum = 0.9f)
        val updatableArray: ParamsArray = UpdateMethodsUtils.buildParamsArray()
        val supportStructure = updateHelper.getSupportStructure(updatableArray)

        supportStructure.v.assignValues(UpdateMethodsUtils.supportArray1())

        updateHelper.update(array = updatableArray, errors = UpdateMethodsUtils.buildSparseErrors())

        it("should match the expected updated array") {
          assertTrue {
            updatableArray.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.167f, -0.24933f, 0.095f, 0.757f, 0.63743f)),
              tolerance = 1.0e-6f)
          }
        }
      }
    }
  }
})
