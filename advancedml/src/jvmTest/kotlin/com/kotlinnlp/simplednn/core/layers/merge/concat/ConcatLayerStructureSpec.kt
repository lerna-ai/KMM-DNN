/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain contexte at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.merge.concat

import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertTrue

/**
 *
 */
class ConcatLayerStructureSpec : Spek({

  describe("a ConcatLayer") {

    context("forward") {

      val layer = ConcatLayerUtils.buildLayer()
      layer.forward()

      it("should match the expected outputArray") {
        assertTrue {
          layer.outputArray.values.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(-0.9f, 0.9f, 0.6f, 0.1f, 0.0f, 0.5f, -0.7f, -0.7f, 0.8f)),
            tolerance = 1.0e-05f)
        }
      }
    }

    context("backward") {

      val layer = ConcatLayerUtils.buildLayer()

      layer.forward()

      layer.outputArray.assignErrors(ConcatLayerUtils.getOutputErrors())
      layer.backward(propagateToInput = true)

      it("should match the expected errors of the inputArray at index 0") {
        assertTrue {
          layer.inputArrays[0].errors.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(-1.0f, -0.2f, 0.4f, -0.2f)),
            tolerance = 1.0e-05f)
        }
      }

      it("should match the expected errors of the inputArray at index 1") {
        assertTrue {
          layer.inputArrays[1].errors.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(0.0f, -0.7f)),
            tolerance = 1.0e-05f)
        }
      }

      it("should match the expected errors of the inputArray at index 2") {
        assertTrue {
          layer.inputArrays[2].errors.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(0.2f, -0.1f, -0.7f)),
            tolerance = 1.0e-05f)
        }
      }
    }
  }
})
