/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain1contexte at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.merge.distance

import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertTrue

/**
 *
 */
class DistanceLayerStructureSpec: Spek({

  describe("a DistanceLayer") {

    context("forward") {

      val layer = DistanceLayerUtils.buildLayer()
      layer.forward()

      it("should match the expected outputArray") {
        assertTrue {
          layer.outputArray.values.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(0.06081f)),
            tolerance = 1.0e-05f)
        }
      }
    }

    context("backward") {

      val layer = DistanceLayerUtils.buildLayer()

      layer.forward()

      layer.outputArray.assignErrors(DistanceLayerUtils.getOutputErrors())
      layer.backward(propagateToInput = true)

      it("should match the expected errors of the inputArray1") {
        assertTrue {
          layer.inputArrays[0].errors.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(-0.04864f, -0.04864f, 0.0f, 0.04864f)),
            tolerance = 1.0e-05f)
        }
      }

      it("should match the expected errors of the inputArray2") {
        assertTrue {
          layer.inputArrays[1].errors.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(0.04864f, 0.04864f, 0.0f, -0.04864f)),
            tolerance = 1.0e-05f)
        }
      }
    }
  }
})
