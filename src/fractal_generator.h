/**
 * @file    fractal_generator.h
 * @author  Kieran Hillier
 * @brief   Provides CUDA kernel for Mandelbrot fractal generation.
 */

#ifndef FRACTAL_GENERATOR_H
#define FRACTAL_GENERATOR_H

#include <math.h>
#include "header.h"

/* RGB colour channel indices */
#define R 0  /* Red   */
#define G 1  /* Green */
#define B 2  /* Blue  */

/* Conversion and scaling factors */
#define DEGREES_TO_RADIANS (M_PI / 180.0)
#define ONE_THIRD_M_PI (M_PI / 3.0)  // Represents 120° in radians
#define SINE_SCALE 0.5

/* Scaling factor for imaginary part of calculation.*/
#define MANDELBROT_SCALE 2.0

/* Threshold for squared magnitude, indicating escape.*/
#define ESCAPE_RADIUS_SQUARED 4.0

/**
 * Computes the colour gradient
 * @param colour: the output vector 
 * @param x: the gradient (between 0 and 360)
 * @param min: Minimum variation of the RGB channels
 * @param max: Maximum variation of the RGB channels
 */
__device__ void GroundColourMix(
    double *colour, double x, double min, double max);

/**
 * CUDA Kernel for generating Mandelbrot fractal.
 * @param pixels: Pointer to the output pixel data.
 * @param width: Width of the output image.
 * @param height: Height of the output image.
 * @param pitch: Width of the allocated 2D pitched memory for the pixels.
 */
__global__ void fractal_generator(
    rgb_pixel_t *pixels, int width, int height, int pitch);

#endif /* FRACTAL_GENERATOR_H */