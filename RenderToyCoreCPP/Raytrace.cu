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
__device__ void cudaFill(const Scene<FLOAT>& pScene, Matrix44<FLOAT> inverse_mvp, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	RaytraceCUDA::SetPixel<FLOAT> setpixel(bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, x, y);
	RaytraceCUDA::ComputePixel<FLOAT, T>(pScene, inverse_mvp, setpixel, X_SUPERSAMPLES, Y_SUPERSAMPLES);
}

////////////////////////////////////////////////////////////////////////////////
// Host Code
////////////////////////////////////////////////////////////////////////////////

void CUDA_CALL(cudaError_t error) {
	if (error == 0) return;
}

#define TRY_CUDA(fn) CUDA_CALL(fn);

template <typename FLOAT>
void cudaRender(const void* pScene, const void* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride, std::function<void(const Scene<FLOAT>&, const Matrix44<FLOAT>&, void*, int, int, const dim3&, const dim3&)> fn) {
	// Allocate the scene buffer for CUDA.
	Scene<FLOAT>* host_scene = (Scene<FLOAT>*)pScene;
	Scene<FLOAT>* device_scene = nullptr;
	TRY_CUDA(cudaMalloc((void**)&device_scene, host_scene->FileSize));
	TRY_CUDA(cudaMemcpy(device_scene, host_scene, host_scene->FileSize, cudaMemcpyHostToDevice));
	// Allocate the bitmap buffer for CUDA.
	void* device_bitmap_ptr = nullptr;
	int device_bitmap_stride = 4 * bitmap_width;
	TRY_CUDA(cudaMalloc((void **)&device_bitmap_ptr, device_bitmap_stride * bitmap_height));
	// Launch the kernel.
	dim3 grid((bitmap_width + 15) / 16, (bitmap_height + 15) / 16, 1);
	dim3 threads(16, 16, 1);
	fn(*device_scene, *(Matrix44<FLOAT>*)pInverseMVP, device_bitmap_ptr, bitmap_width, bitmap_height, grid, threads);
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

__global__ void cudaRaycastKernelF32(const Scene<float>& pScene, Matrix44<float> inverse_mvp, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {
	cudaFill<float, RaytraceCUDA::RenderModeRaycast<float>>(pScene, inverse_mvp, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
}

extern "C" void RaycastCUDAF32(const void* pScene, const void* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
{
	cudaRender<float>(pScene, pInverseMVP, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, [](const Scene<float>& device_scene, const Matrix44<float>& InverseMVP, void* device_bitmap_ptr, int bitmap_width, int bitmap_height, const dim3& grid, const dim3& threads) {
		cudaRaycastKernelF32<<<grid, threads>>>(device_scene, InverseMVP, device_bitmap_ptr, bitmap_width, bitmap_height, 4 * bitmap_width);
	});
}

__global__ void cudaRaycastKernelF64(const Scene<double>& pScene, Matrix44<double> inverse_mvp, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {
	cudaFill<double, RaytraceCUDA::RenderModeRaycast<double>>(pScene, inverse_mvp, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
}

extern "C" void RaycastCUDAF64(const void* pScene, const void* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
{
	cudaRender<double>(pScene, pInverseMVP, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, [](const Scene<double>& device_scene, const Matrix44<double>& InverseMVP, void* device_bitmap_ptr, int bitmap_width, int bitmap_height, const dim3& grid, const dim3& threads) {
		cudaRaycastKernelF64<<<grid, threads>>>(device_scene, InverseMVP, device_bitmap_ptr, bitmap_width, bitmap_height, 4 * bitmap_width);
	});
}

__global__ void cudaRaycastBitangentsKernelF32(const Scene<float>& pScene, Matrix44<float> inverse_mvp, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {
	cudaFill<float, RaytraceCUDA::RenderModeRaycastBitangents<float>>(pScene, inverse_mvp, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
}

extern "C" void RaycastBitangentsCUDAF32(const void* pScene, const void* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
{
	cudaRender<float>(pScene, pInverseMVP, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, [](const Scene<float>& device_scene, const Matrix44<float>& InverseMVP, void* device_bitmap_ptr, int bitmap_width, int bitmap_height, const dim3& grid, const dim3& threads) {
		cudaRaycastBitangentsKernelF32<<<grid, threads>>>(device_scene, InverseMVP, device_bitmap_ptr, bitmap_width, bitmap_height, 4 * bitmap_width);
	});
}

__global__ void cudaRaycastBitangentsKernelF64(const Scene<double>& pScene, Matrix44<double> inverse_mvp, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {
	cudaFill<double, RaytraceCUDA::RenderModeRaycastBitangents<double>>(pScene, inverse_mvp, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
}

extern "C" void RaycastBitangentsCUDAF64(const void* pScene, const void* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
{
	cudaRender<double>(pScene, pInverseMVP, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, [](const Scene<double>& device_scene, const Matrix44<double>& InverseMVP, void* device_bitmap_ptr, int bitmap_width, int bitmap_height, const dim3& grid, const dim3& threads) {
		cudaRaycastBitangentsKernelF64<<<grid, threads>>>(device_scene, InverseMVP, device_bitmap_ptr, bitmap_width, bitmap_height, 4 * bitmap_width);
	});
}

__global__ void cudaRaycastNormalsKernelF32(const Scene<float>& pScene, Matrix44<float> inverse_mvp, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {
	cudaFill<float, RaytraceCUDA::RenderModeRaycastNormals<float>>(pScene, inverse_mvp, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
}

extern "C" void RaycastNormalsCUDAF32(const void* pScene, const void* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
{
	cudaRender<float>(pScene, pInverseMVP, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, [](const Scene<float>& device_scene, const Matrix44<float>& InverseMVP, void* device_bitmap_ptr, int bitmap_width, int bitmap_height, const dim3& grid, const dim3& threads) {
		cudaRaycastNormalsKernelF32<<<grid, threads>>>(device_scene, InverseMVP, device_bitmap_ptr, bitmap_width, bitmap_height, 4 * bitmap_width);
	});
}

__global__ void cudaRaycastNormalsKernelF64(const Scene<double>& pScene, Matrix44<double> inverse_mvp, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {
	cudaFill<double, RaytraceCUDA::RenderModeRaycastNormals<double>>(pScene, inverse_mvp, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
}

extern "C" void RaycastNormalsCUDAF64(const void* pScene, const void* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
{
	cudaRender<double>(pScene, pInverseMVP, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, [](const Scene<double>& device_scene, const Matrix44<double>& InverseMVP, void* device_bitmap_ptr, int bitmap_width, int bitmap_height, const dim3& grid, const dim3& threads) {
		cudaRaycastNormalsKernelF64<<<grid, threads>>>(device_scene, InverseMVP, device_bitmap_ptr, bitmap_width, bitmap_height, 4 * bitmap_width);
	});
}

__global__ void cudaRaycastTangentsKernelF32(const Scene<float>& pScene, Matrix44<float> inverse_mvp, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {
	cudaFill<float, RaytraceCUDA::RenderModeRaycastTangents<float>>(pScene, inverse_mvp, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
}

extern "C" void RaycastTangentsCUDAF32(const void* pScene, const void* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
{
	cudaRender<float>(pScene, pInverseMVP, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, [](const Scene<float>& device_scene, const Matrix44<float>& InverseMVP, void* device_bitmap_ptr, int bitmap_width, int bitmap_height, const dim3& grid, const dim3& threads) {
		cudaRaycastTangentsKernelF32<<<grid, threads>>>(device_scene, InverseMVP, device_bitmap_ptr, bitmap_width, bitmap_height, 4 * bitmap_width);
	});
}

__global__ void cudaRaycastTangentsKernelF64(const Scene<double>& pScene, Matrix44<double> inverse_mvp, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {
	cudaFill<double, RaytraceCUDA::RenderModeRaycastTangents<double>>(pScene, inverse_mvp, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
}

extern "C" void RaycastTangentsCUDAF64(const void* pScene, const void* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
{
	cudaRender<double>(pScene, pInverseMVP, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, [](const Scene<double>& device_scene, const Matrix44<double>& InverseMVP, void* device_bitmap_ptr, int bitmap_width, int bitmap_height, const dim3& grid, const dim3& threads) {
		cudaRaycastTangentsKernelF64<<<grid, threads>>>(device_scene, InverseMVP, device_bitmap_ptr, bitmap_width, bitmap_height, 4 * bitmap_width);
	});
}

__global__ void cudaRaytraceKernelF32(const Scene<float>& pScene, Matrix44<float> inverse_mvp, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {
	cudaFill<float, RaytraceCUDA::RenderModeRaytrace<float>, 2, 2>(pScene, inverse_mvp, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
}

extern "C" void RaytraceCUDAF32(const void* pScene, const void* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
{
	cudaRender<float>(pScene, pInverseMVP, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, [](const Scene<float>& device_scene, const Matrix44<float>& InverseMVP, void* device_bitmap_ptr, int bitmap_width, int bitmap_height, const dim3& grid, const dim3& threads) {
		cudaRaytraceKernelF32<<<grid, threads>>>(device_scene, InverseMVP, device_bitmap_ptr, bitmap_width, bitmap_height, 4 * bitmap_width);
	});
}

__global__ void cudaRaytraceKernelF64(const Scene<double>& pScene, Matrix44<double> inverse_mvp, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {
	cudaFill<double, RaytraceCUDA::RenderModeRaytrace<double>, 2, 2>(pScene, inverse_mvp, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
}

extern "C" void RaytraceCUDAF64(const void* pScene, const void* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
{
	cudaRender<double>(pScene, pInverseMVP, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, [](const Scene<double>& device_scene, const Matrix44<double>& InverseMVP, void* device_bitmap_ptr, int bitmap_width, int bitmap_height, const dim3& grid, const dim3& threads) {
		cudaRaytraceKernelF64<<<grid, threads>>>(device_scene, InverseMVP, device_bitmap_ptr, bitmap_width, bitmap_height, 4 * bitmap_width);
	});
}

////////////////////////////////////////////////////////////////////////////////
// Ambient Occlusion.
////////////////////////////////////////////////////////////////////////////////

template <typename FLOAT>
__global__ void cudaAOC(const Scene<FLOAT>& pScene, Matrix44<FLOAT> inverse_mvp, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride, int hemisample_count, const Vector4<FLOAT>* hemisamples) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	RaytraceCUDA::SetPixel<FLOAT> setpixel(bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, x, y);
	RaytraceCUDA::ComputePixelAOC<FLOAT>(pScene, inverse_mvp, setpixel, hemisample_count, hemisamples);
}

template <typename FLOAT>
void AmbientOcclusionCUDA(const void* pScene, const void* pMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride, int hemisample_count, const void* hemisamples)
{
	// Allocate the scene buffer for CUDA.
	Scene<FLOAT>* host_scene = (Scene<FLOAT>*)pScene;
	Scene<FLOAT>* device_scene = nullptr;
	TRY_CUDA(cudaMalloc((void**)&device_scene, host_scene->FileSize));
	TRY_CUDA(cudaMemcpy(device_scene, host_scene, host_scene->FileSize, cudaMemcpyHostToDevice));
	// Allocate the bitmap buffer for CUDA.
	void* device_bitmap_ptr = nullptr;
	int device_bitmap_stride = 4 * bitmap_width;
	TRY_CUDA(cudaMalloc((void**)&device_bitmap_ptr, device_bitmap_stride * bitmap_height));
	// Allocate the hemisample buffer for CUDA.
	Vector4<FLOAT>* device_hemisamples = nullptr;
	TRY_CUDA(cudaMalloc((void**)&device_hemisamples, sizeof(Vector4<FLOAT>) * hemisample_count));
	TRY_CUDA(cudaMemcpy(device_hemisamples, hemisamples, sizeof(Vector4<FLOAT>) * hemisample_count, cudaMemcpyHostToDevice));
	// Launch the kernel.
	dim3 grid((bitmap_width + 15) / 16, (bitmap_height + 15) / 16, 1);
	dim3 threads(16, 16, 1);
	cudaAOC<<<grid, threads>>>(*device_scene, *(Matrix44<FLOAT>*)pMVP, device_bitmap_ptr, bitmap_width, bitmap_height, 4 * bitmap_width, hemisample_count, device_hemisamples);
	// Copy back the render result to the CPU buffer.
	for (int y = 0; y < bitmap_height; ++y)
	{
		void* pDevice = (unsigned char*)device_bitmap_ptr + device_bitmap_stride * y;
		void* pHost = (unsigned char*)bitmap_ptr + bitmap_stride * y;
		TRY_CUDA(cudaMemcpy(pHost, pDevice, 4 * bitmap_width, cudaMemcpyDeviceToHost));
	}
	// Clean up.
	TRY_CUDA(cudaFree(device_hemisamples));
	device_hemisamples = nullptr;
	TRY_CUDA(cudaFree(device_bitmap_ptr));
	device_bitmap_ptr = nullptr;
	TRY_CUDA(cudaFree(device_scene));
	device_scene = nullptr;
}

extern "C" void AmbientOcclusionCUDAF32(const void* pScene, const void* pMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride, int hemisample_count, const void* hemisamples)
{
	AmbientOcclusionCUDA<float>(pScene, pMVP, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, hemisample_count, hemisamples);
}

extern "C" void AmbientOcclusionCUDAF64(const void* pScene, const void* pMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride, int hemisample_count, const void* hemisamples)
{
	AmbientOcclusionCUDA<double>(pScene, pMVP, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, hemisample_count, hemisamples);
}