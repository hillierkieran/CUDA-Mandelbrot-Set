/**
 * @file    colour_mixer.cu
 * @author  Kieran Hillier
 * @brief   Implementation of colour gradient computation
 *          for the Mandelbrot fractal.
 */

#include "colour_mixer.h"

/**
 * Computes the colour gradient
 * @param colour: the output vector 
 * @param x: the gradient (between 0 and 360)
 * @param min: Minimum variation of the RGB channels
 * @param max: Maximum variation of the RGB channels
 */
__device__ void GroundColourMix(
    double *colour, double x, double min, double max)
{
    double normalized_x = x * DEGREES_TO_RADIANS;  // Convert x to radians

    /* Use trigonometric functions to compute RGB values. 
       Shift and scale the sine wave to oscillate between min and max. */
    colour[R] = SINE_SCALE * 
        (sin(normalized_x) + 1) * (max - min) + min;
    colour[G] = SINE_SCALE * 
        (sin(normalized_x - ONE_THIRD_M_PI) + 1) * (max - min) + min;
    colour[B] = SINE_SCALE * 
        (sin(normalized_x + ONE_THIRD_M_PI) + 1) * (max - min) + min;
}
