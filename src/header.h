/**
 * @file    header.h
 * @author  Kieran Hillier
 * @brief   Defines constants and includes for Mandelbrot
 *          fractal generation and image creation.
 */

#ifndef HEADER_H
#define HEADER_H

#include <stdio.h>
#include <cuda_runtime.h>

#include "bmpfile.h"
#include "colour_mixer.h"
#include "fractal_generator.h"

/* Mandelbrot values */
#define RESOLUTION 8700.0           /* Fractal zoom level */
#define XCENTER -0.55               /* X coordinate of fractal center */
#define YCENTER 0.6                 /* Y coordinate of fractal center */
#define MAX_ITER 1000               /* Max iterations for Mandelbrot calc */
#define WIDTH 1920                  /* Image width in pixels */
#define HEIGHT 1080                 /* Image height in pixels */

/* Colour Values */
#define COLOUR_MIN 1                /* Minimum RGB value */
#define COLOUR_DEPTH 255            /* Maximum RGB value */
#define COLOUR_MAX 240.0            /* Max colour gradient value */
#define GRADIENT_COLOUR_MAX 230.0   /* Max gradient colour value */

/* Output filename */
#define FILENAME "my_mandelbrot_fractal.bmp"

/* CUDA block size */
#define BLOCK_SIZE 16               /* Number of threads in a CUDA block */

#endif /* HEADER_H */
