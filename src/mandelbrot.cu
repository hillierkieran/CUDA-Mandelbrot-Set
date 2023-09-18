/**
 * @file    mandelbrot.cu
 * @author  Kieran Hillier
 * @brief   Generate Mandelbrot fractal using CUDA and save as BMP image.
 * 
 * Compilation:
 * Use the provided makefile to compile and run the program.
 */

#include "header.h"

/* Macro for CUDA error checking */
#define CUDA_CALL(x) {\
    cudaError_t call_result = x;\
    if (call_result != cudaSuccess) {\
        printf("CUDA error %d %s:%d\n", call_result, __FILE__, __LINE__);\
        exit(1); }}

/* Macro for null error checking */
#define NULL_CHECK(ptr, msg) {\
    if (!(ptr)) {\
        fprintf(stderr, "%s\n", (msg));\
        cleanup_resources(bmp, host_pixels, d_pixels);\
        exit(1); }}

/**
 * Frees allocated resources: BMP file, host pixels, and device pixels.
 */
void cleanup_resources(
    bmpfile_t *bmp, rgb_pixel_t *host_pixels, rgb_pixel_t *d_pixels)
{
    if (bmp) bmp_destroy(bmp);
    if (host_pixels) free(host_pixels);
    if (d_pixels) CUDA_CALL(cudaFree(d_pixels));
}

/**
 * Main entry point of the application.
 * Generates a Mandelbrot fractal and saves it as a BMP file.
 */
int main()
{
    bmpfile_t *bmp = NULL;          /* Pointer to BMP image object */
    rgb_pixel_t *host_pixels = NULL;/* Pixel data on host */
    rgb_pixel_t *d_pixels = NULL;   /* Pixel data on device (GPU) */
    size_t pitch;                   /* Allocated width for 2D pitched pixels */
    dim3 threadsPerBlock;           /* Number of threads per CUDA block */
    dim3 numBlocks;                 /* Number of blocks in the CUDA grid */
    int col, row;                   /* Iterators for image columns and rows */

    /* Create a new BMP image */
    bmp = bmp_create(WIDTH, HEIGHT, 32);
    NULL_CHECK(bmp, "Failed to create BMP file");

    /* Allocate memory on the host for pixel data */
    host_pixels = (rgb_pixel_t *) malloc(WIDTH * HEIGHT * sizeof(rgb_pixel_t));
    NULL_CHECK(host_pixels, "Failed to allocate memory for host_pixels\n");

    /* Allocate memory on the device using 2D pitched memory */
    CUDA_CALL(cudaMallocPitch(
        &d_pixels, &pitch, WIDTH * sizeof(rgb_pixel_t), HEIGHT));

    /* Set grid and block dimensions for kernel launch */
    threadsPerBlock.x = BLOCK_SIZE;
    threadsPerBlock.y = BLOCK_SIZE;
    numBlocks.x = (WIDTH + BLOCK_SIZE - 1) / BLOCK_SIZE;
    numBlocks.y = (HEIGHT + BLOCK_SIZE - 1) / BLOCK_SIZE;

    /* Launch Mandelbrot generation CUDA kernel */
    fractal_generator<<<numBlocks, threadsPerBlock>>>(
        d_pixels, WIDTH, HEIGHT, pitch/sizeof(rgb_pixel_t));

    /* Check for kernel errors and synchronize device */
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaDeviceSynchronize());

    /* Copy computed pixel data from device to host memory */
    CUDA_CALL(cudaMemcpy2D(
        host_pixels, WIDTH * sizeof(rgb_pixel_t), d_pixels, pitch, 
        WIDTH * sizeof(rgb_pixel_t), HEIGHT, cudaMemcpyDeviceToHost));

    /* Populate the BMP with the computed pixel data */
    for (col = 0; col < WIDTH; col++) {
        for (row = 0; row < HEIGHT; row++) {
            NULL_CHECK(
                bmp_set_pixel(bmp, col, row, host_pixels[row * WIDTH + col]),
                "Failed to set pixel");
        }
    }

    /* Save the populated BMP to a file and cleanup resources */
    NULL_CHECK(bmp_save(bmp, FILENAME), "Failed to save BMP file");
    cleanup_resources(bmp, host_pixels, d_pixels);
    return 0;
}
