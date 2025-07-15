/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.updatemethods.adagrad

import com.kotlinnlp.simplednn.core.arrays.ParamsArray
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.core.functionalities.regularization.ParamsRegularization
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethodConfig
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdaterSupportStructure
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.sparse.SparseNDArray

/**
 * The AdaGrad method assigns a different learning rate to each parameter
 * using the sum of squares of its all historical gradients.
 *
 * References
 * [Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
 *
 * @property learningRate Initial learning rate
 * @property epsilon Bias parameter
 * @property regularization a parameters regularization method
 */
class AdaGradMethod(
  val learningRate: Double = 0.01,
  val epsilon: Double = 1.0E-8,
  regularization: ParamsRegularization? = null
) : UpdateMethod<AdaGradStructure>(regularization) {

  /**
   * Build an [AdaGradMethod] with a given configuration object.
   *
   * @param config the configuration of this update method
   */
  constructor(config: UpdateMethodConfig.AdaGradConfig): this(
    learningRate = config.learningRate,
    epsilon = config.epsilon,
    regularization = config.regularization
  )

  /**
   * @param array the array from which to extract the support structure
   *
   * @return the [UpdaterSupportStructure] extracted from the given [array]
   */
  override fun getSupportStructure(array: ParamsArray): AdaGradStructure = array.getOrSetSupportStructure()

  /**
   * Optimize sparse errors.
   *
   * @param errors the [SparseNDArray] errors to optimize
   * @param supportStructure the support structure of the [UpdateMethod]
   *
   * @return optimized sparse errors
   */
  override fun optimizeSparseErrors(errors: SparseNDArray, supportStructure: AdaGradStructure): SparseNDArray {

    val m = supportStructure.secondOrderMoments

    m.assignSum(errors.prod(errors))

    return errors.div(m.sqrt(mask = errors.mask).assignSum(this.epsilon)).assignProd(this.learningRate)
  }

  /**
   * Optimize dense errors.
   *
   * @param errors the [DenseNDArray] errors to optimize
   * @param supportStructure the support structure of the [UpdateMethod]
   *
   * @return optimized dense errors
   */
  override fun optimizeDenseErrors(errors: DenseNDArray, supportStructure: AdaGradStructure): DenseNDArray {

    val m = supportStructure.secondOrderMoments

    m.assignSum(errors.prod(errors))

    return errors.div(m.sqrt().assignSum(this.epsilon)).assignProd(this.learningRate)
  }
}
