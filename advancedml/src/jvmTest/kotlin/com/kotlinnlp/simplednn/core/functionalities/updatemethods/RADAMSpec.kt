/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.updatemethods

import com.kotlinnlp.simplednn.core.arrays.ParamsArray
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.radam.RADAMMethod
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertTrue

/**
 *
 */
class RADAMSpec : Spek({

  describe("the RADAM update method") {

    context("time step = 1") {

      context("update with dense errors") {

        context("update") {

          val updateHelper = RADAMMethod(stepSize = 0.001f, beta1 = 0.9f, beta2 = 0.999f, epsilon = 1.0e-8f)
          val updatableArray: ParamsArray = UpdateMethodsUtils.buildParamsArray()
          val supportStructure = updateHelper.getSupportStructure(updatableArray)

          supportStructure.firstOrderMoments.assignValues(UpdateMethodsUtils.supportArray1())
          supportStructure.secondOrderMoments.assignValues(UpdateMethodsUtils.supportArray2())

          updateHelper.newBatch()
          updateHelper.update(array = updatableArray, errors = UpdateMethodsUtils.buildDenseErrors())

          it("should match the expected updated array") {
            assertTrue {
              updatableArray.values.equals(
                DenseNDArrayFactory.arrayOf(floatArrayOf(0.399772f, 0.399605f, 0.499815f, 0.995625f, 0.799866f)),
                tolerance = 1.0e-6f)
            }
          }
        }
      }

      context("update with sparse errors") {

        context("update") {

          val updateHelper = RADAMMethod(stepSize = 0.001f, beta1 = 0.9f, beta2 = 0.999f, epsilon = 1.0e-8f)
          val updatableArray: ParamsArray = UpdateMethodsUtils.buildParamsArray()
          val supportStructure = updateHelper.getSupportStructure(updatableArray)

          supportStructure.firstOrderMoments.assignValues(UpdateMethodsUtils.supportArray1())
          supportStructure.secondOrderMoments.assignValues(UpdateMethodsUtils.supportArray2())

          updateHelper.newBatch()
          updateHelper.update(array = updatableArray, errors = UpdateMethodsUtils.buildSparseErrors())

          it("should match the expected updated array") {
            assertTrue {
              updatableArray.values.equals(
                DenseNDArrayFactory.arrayOf(floatArrayOf(0.3998007f, 0.39960507f, 0.49982983f, -269998.94f, 0.7998515f)),
                tolerance = 1.0e-6f)
            }
          }
        }
      }
    }

    context("time step = 6") {

      context("update with dense errors") {

        context("update") {

          val updateHelper = RADAMMethod(stepSize = 0.001f, beta1 = 0.9f, beta2 = 0.999f, epsilon = 1.0e-8f)
          val updatableArray: ParamsArray = UpdateMethodsUtils.buildParamsArray()
          val supportStructure = updateHelper.getSupportStructure(updatableArray)

          supportStructure.firstOrderMoments.assignValues(UpdateMethodsUtils.supportArray1())
          supportStructure.secondOrderMoments.assignValues(UpdateMethodsUtils.supportArray2())

          repeat(5) { updateHelper.newBatch() }

          updateHelper.newBatch()
          updateHelper.update(array = updatableArray, errors = UpdateMethodsUtils.buildDenseErrors())

          it("should match the expected updated array") {
            assertTrue {
              updatableArray.values.equals(
                DenseNDArrayFactory.arrayOf(floatArrayOf(0.399997f, 0.399995f, 0.499998f, 0.999941f, 0.799998f)),
                tolerance = 1.0e-6f)
            }
          }
        }
      }

      context("update with sparse errors") {

        context("update") {

          val updateHelper = RADAMMethod(stepSize = 0.001f, beta1 = 0.9f, beta2 = 0.999f, epsilon = 1.0e-8f)
          val updatableArray: ParamsArray = UpdateMethodsUtils.buildParamsArray()
          val supportStructure = updateHelper.getSupportStructure(updatableArray)

          supportStructure.firstOrderMoments.assignValues(UpdateMethodsUtils.supportArray1())
          supportStructure.secondOrderMoments.assignValues(UpdateMethodsUtils.supportArray2())

          repeat(5) { updateHelper.newBatch() }

          updateHelper.newBatch()
          updateHelper.update(array = updatableArray, errors = UpdateMethodsUtils.buildSparseErrors())

          it("should match the expected updated array") {
            assertTrue {
              updatableArray.values.equals(
                DenseNDArrayFactory.arrayOf(floatArrayOf(0.39999732f, 0.39999467f, 0.4999977f, -1489.6433f, 0.799998f)),
                tolerance = 1.0e-6f)
            }
          }
        }
      }
    }
  }
})
