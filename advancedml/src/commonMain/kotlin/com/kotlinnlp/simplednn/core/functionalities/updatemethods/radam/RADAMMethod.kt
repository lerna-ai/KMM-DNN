/* Copyright 2020-present Simone Cangialosi. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.updatemethods.radam

import com.kotlinnlp.simplednn.core.functionalities.regularization.ParamsRegularization
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethodConfig
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.adam.ADAMMethod
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * The Rectified ADAM method.
 *
 * @property stepSize the initial step size
 * @property beta1 the beta1 hyper-parameter
 * @property beta2 the beta2 hyper-parameter
 * @property epsilon the epsilon hyper-parameter
 * @property regularization a parameters regularization method
 */
class RADAMMethod(
  stepSize: Float = 0.001f,
  beta1: Float = 0.9f,
  beta2: Float = 0.999f,
  epsilon: Float = 1.0E-8f,
  regularization: ParamsRegularization? = null
) : ADAMMethod(
  stepSize = stepSize,
  beta1 = beta1,
  beta2 = beta2,
  epsilon = epsilon,
  regularization = regularization
) {

  /**
   * Build a [RADAMMethod] with a given configuration object.
   *
   * @param config the configuration of this update method
   */
  constructor(config: UpdateMethodConfig.RADAMConfig) : this(
    stepSize = config.stepSize,
    beta1 = config.beta1,
    beta2 = config.beta2,
    epsilon = config.epsilon,
    regularization = config.regularization
  )

  /**
   * The maximum length of the approximated SMA.
   */
  private val roMax: Float = 2.0f / (1 - this.beta2) - 1.0f

  /**
   * @return the `alpha` coefficient
   */
  override fun calcAlpha(): Float {

    val b1T: Float = this.beta1.pow(this.timeStep)
    val b2T: Float = this.beta2.pow(this.timeStep)
    val ro: Float = this.roMax - 2.0f * this.timeStep * b2T / (1.0f - b2T)

    val rect: Float = if (ro > 4.0)
      sqrt((ro - 4.0f) * (ro - 2.0f) * this.roMax / ((this.roMax - 4.0f) * (this.roMax - 2.0f) * ro))
    else
      1.0f

    return this.stepSize * rect * sqrt(1.0f - b2T) / (1.0f - b1T)
  }
}
