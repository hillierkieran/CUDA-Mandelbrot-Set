/**
 * @file    fractal_generator.h
 * @author  Kieran Hillier
 * @brief   Provides CUDA kernel for Mandelbrot fractal generation.
 */

#ifndef FRACTAL_GENERATOR_H
#define FRACTAL_GENERATOR_H

#include "header.h"

/* Scaling factor for imaginary part of calculation.*/
#define MANDELBROT_SCALE 2.0

/* Threshold for squared magnitude, indicating escape.*/
#define ESCAPE_RADIUS_SQUARED 4.0

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