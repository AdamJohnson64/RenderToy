////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// This file contains the core CUDA implementation of the raytracer.
////////////////////////////////////////////////////////////////////////////////

#include <cuda_runtime.h>
#include <math.h>
#include <cfloat>
#include <functional>
#include "Raytrace.h"

namespace RaytraceCUDA {
	#define DEVICE_PREFIX __device__
	#define DEVICE_SUFFIX
	#include "Raytrace.inc"
	#undef DEVICE_SUFFIX
	#undef DEVICE_PREFIX
}

template <typename FLOAT, typename T, int X_SUPERSAMPLES = 1, int Y_SUPERSAMPLES = 1>
__device__ void cudaFill(const Scene<FLOAT>* pScene, Matrix44<FLOAT> inverse_mvp, void *bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	RaytraceCUDA::SetPixel<FLOAT> setpixel(bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, x, y);
	RaytraceCUDA::ComputePixel<FLOAT, T, X_SUPERSAMPLES, Y_SUPERSAMPLES>(pScene, inverse_mvp, setpixel);
}

////////////////////////////////////////////////////////////////////////////////
// Host Code
////////////////////////////////////////////////////////////////////////////////

void CUDA_CALL(cudaError_t error) {
	if (error == 0) return;
}

#define TRY_CUDA(fn) CUDA_CALL(fn);

template <typename FLOAT>
void cudaRender(const void* pScene, FLOAT* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride, std::function<void(const Scene<FLOAT>*, const Matrix44<FLOAT>&, void*, int, int, const dim3&, const dim3&)> fn) {
	// Allocate the scene buffer for CUDA.
	Scene<FLOAT>* host_scene = (Scene<FLOAT>*)pScene;
	Scene<FLOAT>* device_scene = nullptr;
	TRY_CUDA(cudaMalloc((void**)&device_scene, host_scene->FileSize));
	TRY_CUDA(cudaMemcpy(device_scene, host_scene, host_scene->FileSize, cudaMemcpyHostToDevice));
	// Allocate the bitmap buffer for CUDA.
	void *device_bitmap_ptr = nullptr;
	int device_bitmap_stride = 4 * bitmap_width;
	TRY_CUDA(cudaMalloc((void **)&device_bitmap_ptr, device_bitmap_stride * bitmap_height));
	// Launch the kernel.
	dim3 grid((bitmap_width + 15) / 16, (bitmap_height + 15) / 16, 1);
	dim3 threads(16, 16, 1);
	fn(device_scene, *(Matrix44<FLOAT>*)pInverseMVP, device_bitmap_ptr, bitmap_width, bitmap_height, grid, threads);
	// Copy back the render result to the CPU buffer.
	for (int y = 0; y < bitmap_height; ++y)
	{
		void* pDevice = (unsigned char*)device_bitmap_ptr + device_bitmap_stride * y;
		void* pHost = (unsigned char*)bitmap_ptr + bitmap_stride * y;
		TRY_CUDA(cudaMemcpy(pHost, pDevice, 4 * bitmap_width, cudaMemcpyDeviceToHost));
	}
	// Clean up.
	TRY_CUDA(cudaFree(device_bitmap_ptr));
	device_bitmap_ptr = nullptr;
	TRY_CUDA(cudaFree(device_scene));
	device_scene = nullptr;
}

////////////////////////////////////////////////////////////////////////////////
// External calls (Render Types).
////////////////////////////////////////////////////////////////////////////////

__global__ void cudaRaycastKernel(const Scene<double> *pScene, Matrix44<double> inverse_mvp, void *bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {
	cudaFill<double, RaytraceCUDA::DoRaycast<double>>(pScene, inverse_mvp, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
}

extern "C" void RaycastCUDA(void* pScene, double* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
{
	cudaRender<double>(pScene, pInverseMVP, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, [](const Scene<double>* device_scene, const Matrix44<double>& InverseMVP, void* device_bitmap_ptr, int bitmap_width, int bitmap_height, const dim3& grid, const dim3& threads) {
		cudaRaycastKernel<<<grid, threads>>>(device_scene, InverseMVP, device_bitmap_ptr, bitmap_width, bitmap_height, 4 * bitmap_width);
	});
}

__global__ void cudaRaycastBitangentsKernel(const Scene<double> *pScene, Matrix44<double> inverse_mvp, void *bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {
	cudaFill<double, RaytraceCUDA::DoRaycastBitangents<double>>(pScene, inverse_mvp, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
}

extern "C" void RaycastBitangentsCUDA(void* pScene, double* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
{
	cudaRender<double>(pScene, pInverseMVP, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, [](const Scene<double>* device_scene, const Matrix44<double>& InverseMVP, void* device_bitmap_ptr, int bitmap_width, int bitmap_height, const dim3& grid, const dim3& threads) {
		cudaRaycastBitangentsKernel<<<grid, threads>>>(device_scene, InverseMVP, device_bitmap_ptr, bitmap_width, bitmap_height, 4 * bitmap_width);
	});
}

__global__ void cudaRaycastNormalsKernel(const Scene<double> *pScene, Matrix44<double> inverse_mvp, void *bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {
	cudaFill<double, RaytraceCUDA::DoRaycastNormals<double>>(pScene, inverse_mvp, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
}

extern "C" void RaycastNormalsCUDA(void* pScene, double* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
{
	cudaRender<double>(pScene, pInverseMVP, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, [](const Scene<double>* device_scene, const Matrix44<double>& InverseMVP, void* device_bitmap_ptr, int bitmap_width, int bitmap_height, const dim3& grid, const dim3& threads) {
		cudaRaycastNormalsKernel<<<grid, threads>>>(device_scene, InverseMVP, device_bitmap_ptr, bitmap_width, bitmap_height, 4 * bitmap_width);
	});
}

__global__ void cudaRaycastTangentsKernel(const Scene<double> *pScene, Matrix44<double> inverse_mvp, void *bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {
	cudaFill<double, RaytraceCUDA::DoRaycastTangents<double>>(pScene, inverse_mvp, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
}

extern "C" void RaycastTangentsCUDA(void* pScene, double* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
{
	cudaRender<double>(pScene, pInverseMVP, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, [](const Scene<double>* device_scene, const Matrix44<double>& InverseMVP, void* device_bitmap_ptr, int bitmap_width, int bitmap_height, const dim3& grid, const dim3& threads) {
		cudaRaycastTangentsKernel<<<grid, threads>>>(device_scene, InverseMVP, device_bitmap_ptr, bitmap_width, bitmap_height, 4 * bitmap_width);
	});
}

__global__ void cudaRaytraceKernelF32(const Scene<float> *pScene, Matrix44<float> inverse_mvp, void *bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {
	cudaFill<float, RaytraceCUDA::DoRaytrace<float>, 2, 2>(pScene, inverse_mvp, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
}

extern "C" void RaytraceCUDAF32(void* pScene, float* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
{
	cudaRender<float>(pScene, pInverseMVP, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, [](const Scene<float>* device_scene, const Matrix44<float>& InverseMVP, void* device_bitmap_ptr, int bitmap_width, int bitmap_height, const dim3& grid, const dim3& threads) {
		cudaRaytraceKernelF32<<<grid, threads>>>(device_scene, InverseMVP, device_bitmap_ptr, bitmap_width, bitmap_height, 4 * bitmap_width);
	});
}

__global__ void cudaRaytraceKernelF64(const Scene<double> *pScene, Matrix44<double> inverse_mvp, void *bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {
	cudaFill<double, RaytraceCUDA::DoRaytrace<double>, 2, 2>(pScene, inverse_mvp, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
}

extern "C" void RaytraceCUDAF64(void* pScene, double* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
{
	cudaRender<double>(pScene, pInverseMVP, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, [](const Scene<double>* device_scene, const Matrix44<double>& InverseMVP, void* device_bitmap_ptr, int bitmap_width, int bitmap_height, const dim3& grid, const dim3& threads) {
		cudaRaytraceKernelF64<<<grid, threads>>>(device_scene, InverseMVP, device_bitmap_ptr, bitmap_width, bitmap_height, 4 * bitmap_width);
	});
}