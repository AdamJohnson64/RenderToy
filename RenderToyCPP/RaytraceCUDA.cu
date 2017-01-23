////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

#include <cuda_runtime.h>
#include <math.h>
#include <cfloat>
#include <functional>
#include "Raytrace.h"

namespace RaytraceCUDA {
	#define DEVICE_PROTO __device__
	#include "Raytrace.inc"
	#undef DEVICE_PROTO
}

template <typename T, int X_SUPERSAMPLES, int Y_SUPERSAMPLES>
__device__ void cudaFill(const Scene* pScene, Matrix4D inverse_mvp, void *bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	RaytraceCUDA::cudaFill2<T, X_SUPERSAMPLES, Y_SUPERSAMPLES>(pScene, inverse_mvp, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, x, y);
}

////////////////////////////////////////////////////////////////////////////////
// Host Code
////////////////////////////////////////////////////////////////////////////////

void CUDA_CALL(cudaError_t error) {
	if (error == 0) return;
}

#define TRY_CUDA(fn) CUDA_CALL(fn);

typedef std::function<void(const Scene*, const Matrix4D&, void*, int, int, const dim3&, const dim3&)> CUDAFN;

void cudaRender(void* pScene, double* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride, CUDAFN fn) {
	// Allocate the scene buffer for CUDA.
	Scene *host_scene = (Scene*)pScene;
	Scene *device_scene = nullptr;
	TRY_CUDA(cudaMalloc((void**)&device_scene, host_scene->FileSize));
	TRY_CUDA(cudaMemcpy(device_scene, host_scene, host_scene->FileSize, cudaMemcpyHostToDevice));
	// Allocate the bitmap buffer for CUDA.
	void *device_bitmap_ptr = nullptr;
	int device_bitmap_stride = 4 * bitmap_width;
	TRY_CUDA(cudaMalloc((void **)&device_bitmap_ptr, device_bitmap_stride * bitmap_height));
	// Launch the kernel.
	dim3 grid((bitmap_width + 15) / 16, (bitmap_height + 15) / 16, 1);
	dim3 threads(16, 16, 1);
	fn(device_scene, *(Matrix4D*)pInverseMVP, device_bitmap_ptr, bitmap_width, bitmap_height, grid, threads);
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

__global__ void cudaRaycastKernel(const Scene *pScene, Matrix4D inverse_mvp, void *bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {
	cudaFill<RaytraceCUDA::DoRaycast, 1, 1>(pScene, inverse_mvp, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
}

extern "C" void CUDARaycast(void* pScene, double* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
{
	cudaRender(pScene, pInverseMVP, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, [](const Scene* device_scene, const Matrix4D& InverseMVP, void* device_bitmap_ptr, int bitmap_width, int bitmap_height, const dim3& grid, const dim3& threads) {
		cudaRaycastKernel<<<grid, threads>>>(device_scene, InverseMVP, device_bitmap_ptr, bitmap_width, bitmap_height, 4 * bitmap_width);
	});
}

__global__ void cudaRaycastBitangentsKernel(const Scene *pScene, Matrix4D inverse_mvp, void *bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {
	cudaFill<RaytraceCUDA::DoRaycastBitangents, 1, 1>(pScene, inverse_mvp, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
}

extern "C" void CUDARaycastBitangents(void* pScene, double* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
{
	cudaRender(pScene, pInverseMVP, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, [](const Scene* device_scene, const Matrix4D& InverseMVP, void* device_bitmap_ptr, int bitmap_width, int bitmap_height, const dim3& grid, const dim3& threads) {
		cudaRaycastBitangentsKernel<<<grid, threads>>>(device_scene, InverseMVP, device_bitmap_ptr, bitmap_width, bitmap_height, 4 * bitmap_width);
	});
}

__global__ void cudaRaycastNormalsKernel(const Scene *pScene, Matrix4D inverse_mvp, void *bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {
	cudaFill<RaytraceCUDA::DoRaycastNormals, 1, 1>(pScene, inverse_mvp, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
}

extern "C" void CUDARaycastNormals(void* pScene, double* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
{
	cudaRender(pScene, pInverseMVP, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, [](const Scene* device_scene, const Matrix4D& InverseMVP, void* device_bitmap_ptr, int bitmap_width, int bitmap_height, const dim3& grid, const dim3& threads) {
		cudaRaycastNormalsKernel<<<grid, threads>>>(device_scene, InverseMVP, device_bitmap_ptr, bitmap_width, bitmap_height, 4 * bitmap_width);
	});
}

__global__ void cudaRaycastTangentsKernel(const Scene *pScene, Matrix4D inverse_mvp, void *bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {
	cudaFill<RaytraceCUDA::DoRaycastTangents, 1, 1>(pScene, inverse_mvp, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
}

extern "C" void CUDARaycastTangents(void* pScene, double* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
{
	cudaRender(pScene, pInverseMVP, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, [](const Scene* device_scene, const Matrix4D& InverseMVP, void* device_bitmap_ptr, int bitmap_width, int bitmap_height, const dim3& grid, const dim3& threads) {
		cudaRaycastTangentsKernel<<<grid, threads>>>(device_scene, InverseMVP, device_bitmap_ptr, bitmap_width, bitmap_height, 4 * bitmap_width);
	});
}

__global__ void cudaRaytraceKernel(const Scene *pScene, Matrix4D inverse_mvp, void *bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {
	cudaFill<RaytraceCUDA::DoRaytrace, 2, 2>(pScene, inverse_mvp, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
}

extern "C" void CUDARaytrace(void* pScene, double* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
{
	cudaRender(pScene, pInverseMVP, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, [](const Scene* device_scene, const Matrix4D& InverseMVP, void* device_bitmap_ptr, int bitmap_width, int bitmap_height, const dim3& grid, const dim3& threads) {
		cudaRaytraceKernel<<<grid, threads>>>(device_scene, InverseMVP, device_bitmap_ptr, bitmap_width, bitmap_height, 4 * bitmap_width);
	});
}