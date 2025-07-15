/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.arrays

import com.kotlinnlp.simplednn.core.functionalities.initializers.ConstantInitializer
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.learningrate.LearningRateStructure
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.simplemath.ndarray.sparse.SparseNDArray
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.*

/**
 *
 */
class ParamsArraySpec : Spek({

  describe("a ParamsArray") {

    context("initialization") {

      context("with an NDArray") {

        val paramsArray = ParamsArray(DenseNDArrayFactory.zeros(Shape(3, 7)))
        val paramsArray2 = ParamsArray(DenseNDArrayFactory.zeros(Shape(3, 7)))

        it("should contain values with the expected number of rows") {
          assertEquals(3, paramsArray.values.rows)
        }

        it("should contain values with the expected number of columns") {
          assertEquals(7, paramsArray.values.columns)
        }

        it("should contain a support structure initialized with null") {
          assertNull(paramsArray.updaterSupportStructure)
        }

        it("should have a different uuid of the one of another instance") {
          assertNotEquals(paramsArray.uuid, paramsArray2.uuid)
        }
      }

      context("with a FloatArray") {

        val paramsArray = ParamsArray(floatArrayOf(0.3f, 0.4f, 0.2f, -0.2f))

        it("should contain the expected values") {
          assertEquals(paramsArray.values, DenseNDArrayFactory.arrayOf(floatArrayOf(0.3f, 0.4f, 0.2f, -0.2f)))
        }
      }

      context("with a list of FloatArray") {

        val paramsArray = ParamsArray(listOf(
          floatArrayOf(0.3f, 0.4f, 0.2f, -0.2f),
          floatArrayOf(0.2f, -0.1f, 0.1f, 0.6f)
        ))

        it("should contain the expected values") {
          assertEquals(paramsArray.values, DenseNDArrayFactory.arrayOf(listOf(
            floatArrayOf(0.3f, 0.4f, 0.2f, -0.2f),
            floatArrayOf(0.2f, -0.1f, 0.1f, 0.6f)
          )))
        }
      }

      context("with a matrix shape and initialized values") {

        val paramsArray = ParamsArray(Shape(2, 4), initializer = ConstantInitializer(0.42f))

        it("should contain the expected values") {
          assertEquals(paramsArray.values, DenseNDArrayFactory.arrayOf(listOf(
            floatArrayOf(0.42f, 0.42f, 0.42f, 0.42f),
            floatArrayOf(0.42f, 0.42f, 0.42f, 0.42f)
          )))
        }
      }

      context("with a vector shape and initialized values") {

        val paramsArray = ParamsArray(size = 4, initializer = ConstantInitializer(0.42f))

        it("should contain the expected values") {
          assertEquals(paramsArray.values, DenseNDArrayFactory.arrayOf(
            floatArrayOf(0.42f, 0.42f, 0.42f, 0.42f)
          ))
        }
      }
    }

    context("support structure") {

      val paramsArray = ParamsArray(DenseNDArrayFactory.zeros(Shape(3, 7))).apply {
        getOrSetSupportStructure<LearningRateStructure>()
      }

      it("should have the expected support structure type") {
        assertTrue { paramsArray.updaterSupportStructure is LearningRateStructure }
      }
    }

    context("params errors") {

      context("build a dense errors without values") {

        val paramsArray = ParamsArray(DenseNDArrayFactory.zeros(Shape(3, 7)))
        val paramsArray2 = ParamsArray(DenseNDArrayFactory.zeros(Shape(3, 7)))

        val paramsErrors = paramsArray.buildDenseErrors()

        it("should contain values with the expected shape") {
          assertEquals(paramsArray.values.shape, paramsErrors.values.shape)
        }

        it("should contain the expected values") {
          assertEquals(paramsErrors.values, DenseNDArrayFactory.zeros(Shape(3, 7)))
        }

        it("should contains the right reference to the paramsArray"){
          assertSame(paramsErrors.refParams, paramsArray)
        }

        it("shouldn't contains the reference to the paramsArray2"){
          assertNotSame(paramsErrors.refParams, paramsArray2)
        }

        it("should create its copy with the same reference to the paramsArray"){
          assertSame(paramsErrors.copy().refParams, paramsArray)
        }
      }

      context("build with default sparse errors") {

        val paramsArray = ParamsArray(
          values = DenseNDArrayFactory.zeros(Shape(3, 7)),
          defaultErrorsType = ParamsArray.ErrorsType.Sparse)

        val paramsErrors = paramsArray.buildDefaultErrors()

        it("should create sparse errors"){
          assertTrue { paramsErrors.values is SparseNDArray }
        }
      }
    }

    context("build with default dense errors") {

      val paramsArray = ParamsArray(
        values = DenseNDArrayFactory.zeros(Shape(3, 7)),
        defaultErrorsType = ParamsArray.ErrorsType.Dense)

      val paramsErrors = paramsArray.buildDefaultErrors()

      it("should create dense errors"){
        assertTrue { paramsErrors.values is DenseNDArray }
      }
    }

    context("build with default errors") {

      val paramsArray = ParamsArray(DenseNDArrayFactory.zeros(Shape(3, 7)))

      val paramsErrors = paramsArray.buildDefaultErrors()

      it("should create dense errors"){
        assertTrue { paramsErrors.values is DenseNDArray }
      }
    }
  }
})
