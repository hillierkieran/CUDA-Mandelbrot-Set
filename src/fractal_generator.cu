/**
 * @file    fractal_generator.cu
 * @author  Kieran Hillier
 * @brief   CUDA kernel implementation for Mandelbrot fractal generation.
 */

#include "fractal_generator.h"

/**
 * CUDA Kernel for generating Mandelbrot fractal.
 * @param pixels: Pointer to the output pixel data.
 * @param width: Width of the output image.
 * @param height: Height of the output image.
 * @param pitch: Width of the allocated 2D pitched memory for the pixels.
 */
__global__ void fractal_generator(
    rgb_pixel_t *pixels, int width, int height, int pitch)
{
    int col, row;       /* 2D position of the thread in the grid */
    int index;          /* 1D mapping of 2D position */
    int i;              /* Iterator for Mandelbrot calculations */
    double x, y;        /* Coordinates mapped to the complex plane */
    double a, b;        /* Complex number components for Mandelbrot calcs */
    double aold, bold;  /* Previous values of 'a' and 'b' */
    double zmagsqr;     /* Magnitude square of complex number */
    double colourValue; /* Computed colour value */
    double colour[3];   /* RGB colour components */

    /* Shared memory buffer for each block's pixel calculations. */
    __shared__ rgb_pixel_t sharedPixels[BLOCK_SIZE][BLOCK_SIZE];

    /* Calculate the 2D position of the thread in the global grid. */
    col = blockIdx.x * blockDim.x + threadIdx.x;
    row = blockIdx.y * blockDim.y + threadIdx.y;

    /* Out-of-bounds check */
    if (col >= width || row >= height)
        return;

    /* Map 2D position to complex plane for Mandelbrot calculations. */
    x = XCENTER + (col - width/2) / RESOLUTION;
    y = YCENTER + (row - height/2) / RESOLUTION;

    /* Initialise variables for Mandelbrot calculations. */
    a = 0; b = 0; aold = 0; bold = 0; zmagsqr = 0; i = 0;

    /* Mandelbrot calculation loop. */
    while (i < MAX_ITER && zmagsqr <= ESCAPE_RADIUS_SQUARED) {
        ++i;
        a = (aold * aold) - (bold * bold) + x;
        b = MANDELBROT_SCALE * aold * bold + y;
        zmagsqr = a * a + b * b;
        aold = a;
        bold = b;
    }


    /* Calculate the colour based on the number of iterations. */
    colourValue = (COLOUR_MAX - ((i / (float) MAX_ITER) * GRADIENT_COLOUR_MAX));
    GroundColourMix(colour, colourValue, COLOUR_MIN, COLOUR_DEPTH);

    /* Store the computed colour in shared memory. */
    sharedPixels[threadIdx.y][threadIdx.x].red = (uint8_t)colour[0];
    sharedPixels[threadIdx.y][threadIdx.x].green = (uint8_t)colour[1];
    sharedPixels[threadIdx.y][threadIdx.x].blue = (uint8_t)colour[2];

    /* Sync all threads in the block to ensure computations are done. */
    __syncthreads();

    /* Transfer shared memory pixel values to global memory using pitch. */
    if (col < width && row < height) {
        index = row * pitch + col;
        pixels[index] = sharedPixels[threadIdx.y][threadIdx.x];
    }
}