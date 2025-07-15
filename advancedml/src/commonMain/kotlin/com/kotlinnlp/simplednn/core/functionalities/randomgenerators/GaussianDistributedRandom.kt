/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.randomgenerators

import kotlin.math.ln
import kotlin.math.sqrt
import kotlin.random.Random


/**
 * A generator of random numbers with a Gaussian distribution.
 *
 * @property variance the variance of the distribution (e.g. 2.0 / n)
 * @property enablePseudoRandom whether to use a pseudo-random generation with the given [seed]
 * @property seed seed used for the pseudo-random generation
 */
class GaussianDistributedRandom(
  val variance: Double = 1.0,
  val enablePseudoRandom: Boolean = true,
  val seed: Long = 1
) : RandomGenerator {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * A random numbers generator with a uniform distribution.
   */
  private val rndGenerator = if (enablePseudoRandom) Random(seed) else Random(Random.nextInt())

  /**
   * @return a random value generated following a Gaussian distribution
   */
  override fun next(): Double {
    return gaussRandom() * sqrt(variance)
  }

  private fun gaussRandom(): Double {
    val u: Double = 2 * rndGenerator.nextDouble() - 1.0
    val v: Double = 2 * rndGenerator.nextDouble() - 1.0
    val r = u * u + v * v
    if (r == 0.0) return 0.0 else if (r > 1) return gaussRandom()
    val c: Double = sqrt(-2 * ln(r) / r)
    return u * c
  }
}
