/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.gradientclipping

import com.kotlinnlp.simplednn.core.functionalities.gradientclipping.GradientClipping
import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsList
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertTrue

/**
 *
 */
class GradientClippingSpec: Spek({

  describe("the gradient clipping") {

    context("clip at value") {
      val paramsErrors: ParamsErrorsList = GradientClippingUtils.buildErrors()
      val gradientClipping = GradientClipping.byValue(0.7f)
      gradientClipping.clip(paramsErrors)

      it("should match the expected parameters at index 0") {
        assertTrue {
          paramsErrors[0].values.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                  floatArrayOf(0.5f, 0.6f, -0.7f, -0.6f),
                  floatArrayOf(0.7f, -0.4f, 0.1f, -0.7f),
                  floatArrayOf(0.7f, -0.7f, 0.3f, 0.5f),
                  floatArrayOf(0.7f, -0.7f, 0.0f, -0.1f),
                  floatArrayOf(0.4f, 0.7f, -0.7f, 0.7f)
              )),
              tolerance = 1.0e-6f)
        }
      }

      it("should match the expected parameters at index 1") {
        assertTrue {
          paramsErrors[1].values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.7f, 0.7f, 0.4f, 0.7f, 0.1f)),
              tolerance = 1.0e-6f)
        }
      }
    }

    context("clip at 2-norm") {
      val paramsErrors: ParamsErrorsList = GradientClippingUtils.buildErrors()
      val gradientClipping = GradientClipping.byNorm(maxNorm = 2.0f, normType = 2f)
      gradientClipping.clip(paramsErrors)

      it("should match the expected parameters at index 0") {
        assertTrue {
          paramsErrors[0].values.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                  floatArrayOf(0.314814f, 0.377777f, -0.503702f, -0.377777f),
                  floatArrayOf(0.440739f, -0.251851f, 0.062962f, -0.503702f),
                  floatArrayOf(0.440739f, -0.440739f, 0.188888f, 0.314814f),
                  floatArrayOf(0.503702f, -0.566665f, 0.0f, -0.062962f),
                  floatArrayOf(0.251851f, 0.629628f,  -0.440739f, 0.503702f)
              )),
              tolerance = 1.0e-6f)
        }
      }

      it("should match the expected parameters at index 1") {
        assertTrue {
          paramsErrors[1].values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.566665f, 0.440739f, 0.251851f, 0.503702f, 0.062962f)),
              tolerance = 1.0e-6f)
        }
      }
    }

    context("clip at inf-norm") {
      val paramsErrors: ParamsErrorsList = GradientClippingUtils.buildErrors()
      val gradientClipping = GradientClipping.byNorm(maxNorm = 0.5f, normType = Float.POSITIVE_INFINITY)
      gradientClipping.clip(paramsErrors)

      it("should match the expected parameters at index 0") {
        assertTrue {
          paramsErrors[0].values.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                  floatArrayOf(0.25f, 0.3f, -0.4f, -0.3f),
                  floatArrayOf(0.35f, -0.2f, 0.05f, -0.4f),
                  floatArrayOf(0.35f, -0.35f, 0.15f, 0.25f),
                  floatArrayOf(0.4f, -0.45f, 0.0f,	-0.05f),
                  floatArrayOf(0.2f, 0.5f, -0.35f, 0.4f)
              )),
              tolerance = 1.0e-6f)
        }
      }

      it("should match the expected parameters at index 1") {
        assertTrue {
          paramsErrors[1].values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.45f, 0.35f, 0.2f, 0.4f, 0.05f)),
              tolerance = 1.0e-6f)
        }
      }
    }
  }
})
