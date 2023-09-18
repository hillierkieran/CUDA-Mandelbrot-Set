/**
 * @file    colour_mixer.h
 * @author  Kieran Hillier
 * @brief   Provides function to compute colour gradients 
 *          for the Mandelbrot fractal.
 */

#ifndef COLOUR_MIXER_H
#define COLOUR_MIXER_H

#include <math.h>

/* RGB colour channel indices */
#define R 0  /* Red   */
#define G 1  /* Green */
#define B 2  /* Blue  */

/* Conversion and scaling factors */
#define DEGREES_TO_RADIANS (M_PI / 180.0)
#define ONE_THIRD_M_PI (M_PI / 3.0)  // Represents 120° in radians
#define SINE_SCALE 0.5

/**
 * Computes the colour gradient
 * @param colour: the output vector 
 * @param x: the gradient (between 0 and 360)
 * @param min: Minimum variation of the RGB channels
 * @param max: Maximum variation of the RGB channels
 */
__device__ void GroundColourMix(
    double *colour, double x, double min, double max);

#endif /* COLOUR_MIXER_H */
