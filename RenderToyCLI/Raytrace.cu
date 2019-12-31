////////////////////////////////////////////////////////////////////////////////
// This file contains the core CUDA implementation of the raytracer.
////////////////////////////////////////////////////////////////////////////////

#pragma region - Section : Boilerplate & Platform -
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

void CUDA_CALL(cudaError_t error) {
	if (error == 0) return;
}

#define TRY_CUDA(fn) CUDA_CALL(fn);

// CudaMemoryBase is the base of all in/out and backed memory helper objects.
class CudaMemoryBase abstract
{
public:
	CudaMemoryBase(int width) {
		TRY_CUDA(cudaMalloc((void**)&device_ptr, width));
	}
	~CudaMemoryBase() {
		TRY_CUDA(cudaFree(device_ptr));
		device_ptr = nullptr;
	}
	void* DeviceMemory() {
		return device_ptr;
	}
protected:
	void* device_ptr;
};

// CudaMemory1DBuffer allocates unbacked temporary device memory.
class CudaMemory1DBuffer : public CudaMemoryBase {
public:
	CudaMemory1DBuffer(int width) : CudaMemoryBase(width) {
	}
};

// CudaMemory1DIn automatically copies memory from host to device.
class CudaMemory1DIn : public CudaMemoryBase {
public:
	CudaMemory1DIn(const void* host_ptr, int width) : CudaMemoryBase(width) {
		TRY_CUDA(cudaMemcpy(device_ptr, host_ptr, width, cudaMemcpyHostToDevice));
	}
};

// CudaMemory2DIn allocates device memory and copies to device.
class CudaMemory2DIn : public CudaMemoryBase {
public:
	CudaMemory2DIn(const void* host_ptr, int stride, int width, int height) : CudaMemoryBase(width * height) {
		TRY_CUDA(cudaMemcpy2D(device_ptr, width, host_ptr, stride, width, height, cudaMemcpyHostToDevice));
	}
};

// CudaMemory2DInOut allocates device memory, copies to device and copies back to host on destruct.
class CudaMemory2DInOut : public CudaMemoryBase {
public:
	CudaMemory2DInOut(void* host_ptr, int stride, int width, int height) : CudaMemoryBase(width * height), host_ptr(host_ptr), stride(stride), width(width), height(height) {
		TRY_CUDA(cudaMemcpy2D(device_ptr, width, host_ptr, stride, width, height, cudaMemcpyHostToDevice));
	}
	~CudaMemory2DInOut() {
		TRY_CUDA(cudaMemcpy2D(host_ptr, stride, device_ptr, width, width, height, cudaMemcpyDeviceToHost));
	}
private:
	void* host_ptr;
	int stride, width, height;
};

// CudaMemory2DOut allocates device memory and copies back to host on destruct.
class CudaMemory2DOut : public CudaMemoryBase {
public:
	CudaMemory2DOut(void* host_ptr, int stride, int width, int height) : CudaMemoryBase(width * height), host_ptr(host_ptr), stride(stride), width(width), height(height) {
	}
	~CudaMemory2DOut() {
		TRY_CUDA(cudaMemcpy2D(host_ptr, stride, device_ptr, width, width, height, cudaMemcpyDeviceToHost));
	}
private:
	void* host_ptr;
	int stride, width, height;
};
#pragma endregion

#pragma region - Render Mode : Common -
template <typename FLOAT, typename T, int X_SUPERSAMPLES = 1, int Y_SUPERSAMPLES = 1>
__device__ void cudaFill(const Scene<FLOAT>& pScene, Matrix44<FLOAT> inverse_mvp, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	RaytraceCUDA::PixelSetARGB<FLOAT> setpixel(bitmap_ptr, render_width, render_height, bitmap_stride, x, y);
	RaytraceCUDA::ComputePixel<FLOAT, T>(pScene, inverse_mvp, setpixel, X_SUPERSAMPLES, Y_SUPERSAMPLES);
}

template <typename FLOAT>
void cudaRender(const void* pScene, const void* pInverseMVP, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride, std::function<void(const Scene<FLOAT>&, const Matrix44<FLOAT>&, void*, int, int, const dim3&, const dim3&)> fn) {
	CudaMemory1DIn cudaScene((const Scene<FLOAT>*)pScene, ((const Scene<FLOAT>*)pScene)->FileSize);
	CudaMemory2DOut cudaBitmap(bitmap_ptr, bitmap_stride, sizeof(int) * render_width, render_height);
	dim3 grid((render_width + 15) / 16, (render_height + 15) / 16, 1);
	dim3 threads(16, 16, 1);
	fn(*(const Scene<FLOAT>*)cudaScene.DeviceMemory(), *(Matrix44<FLOAT>*)pInverseMVP, cudaBitmap.DeviceMemory(), render_width, render_height, grid, threads);
}
#pragma endregion

#pragma region - Render Mode : Raycast -
__global__ void cudaRaycastKernelF32(const Scene<float>& pScene, Matrix44<float> inverse_mvp, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride) {
	cudaFill<float, RaytraceCUDA::RenderModeRaycast<float>>(pScene, inverse_mvp, bitmap_ptr, render_width, render_height, bitmap_stride);
}

extern "C" void RaycastCUDAF32(const void* pScene, const void* pInverseMVP, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride)
{
	cudaRender<float>(pScene, pInverseMVP, bitmap_ptr, render_width, render_height, bitmap_stride, [](const Scene<float>& device_scene, const Matrix44<float>& InverseMVP, void* device_bitmap_ptr, int render_width, int render_height, const dim3& grid, const dim3& threads) {
		cudaRaycastKernelF32<<<grid, threads>>>(device_scene, InverseMVP, device_bitmap_ptr, render_width, render_height, 4 * render_width);
	});
}

__global__ void cudaRaycastKernelF64(const Scene<double>& pScene, Matrix44<double> inverse_mvp, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride) {
	cudaFill<double, RaytraceCUDA::RenderModeRaycast<double>>(pScene, inverse_mvp, bitmap_ptr, render_width, render_height, bitmap_stride);
}

extern "C" void RaycastCUDAF64(const void* pScene, const void* pInverseMVP, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride)
{
	cudaRender<double>(pScene, pInverseMVP, bitmap_ptr, render_width, render_height, bitmap_stride, [](const Scene<double>& device_scene, const Matrix44<double>& InverseMVP, void* device_bitmap_ptr, int render_width, int render_height, const dim3& grid, const dim3& threads) {
		cudaRaycastKernelF64<<<grid, threads>>>(device_scene, InverseMVP, device_bitmap_ptr, render_width, render_height, 4 * render_width);
	});
}
#pragma endregion

#pragma region - Render Mode : Raycast Bitangents -
__global__ void cudaRaycastBitangentsKernelF32(const Scene<float>& pScene, Matrix44<float> inverse_mvp, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride) {
	cudaFill<float, RaytraceCUDA::RenderModeRaycastBitangents<float>>(pScene, inverse_mvp, bitmap_ptr, render_width, render_height, bitmap_stride);
}

extern "C" void RaycastBitangentsCUDAF32(const void* pScene, const void* pInverseMVP, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride)
{
	cudaRender<float>(pScene, pInverseMVP, bitmap_ptr, render_width, render_height, bitmap_stride, [](const Scene<float>& device_scene, const Matrix44<float>& InverseMVP, void* device_bitmap_ptr, int render_width, int render_height, const dim3& grid, const dim3& threads) {
		cudaRaycastBitangentsKernelF32<<<grid, threads>>>(device_scene, InverseMVP, device_bitmap_ptr, render_width, render_height, 4 * render_width);
	});
}

#ifdef USE_F64
__global__ void cudaRaycastBitangentsKernelF64(const Scene<double>& pScene, Matrix44<double> inverse_mvp, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride) {
	cudaFill<double, RaytraceCUDA::RenderModeRaycastBitangents<double>>(pScene, inverse_mvp, bitmap_ptr, render_width, render_height, bitmap_stride);
}

extern "C" void RaycastBitangentsCUDAF64(const void* pScene, const void* pInverseMVP, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride)
{
	cudaRender<double>(pScene, pInverseMVP, bitmap_ptr, render_width, render_height, bitmap_stride, [](const Scene<double>& device_scene, const Matrix44<double>& InverseMVP, void* device_bitmap_ptr, int render_width, int render_height, const dim3& grid, const dim3& threads) {
		cudaRaycastBitangentsKernelF64<<<grid, threads>>>(device_scene, InverseMVP, device_bitmap_ptr, render_width, render_height, 4 * render_width);
	});
}
#endif
#pragma endregion

#pragma region - Render Mode : Raycast Normals -
__global__ void cudaRaycastNormalsKernelF32(const Scene<float>& pScene, Matrix44<float> inverse_mvp, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride) {
	cudaFill<float, RaytraceCUDA::RenderModeRaycastNormals<float>>(pScene, inverse_mvp, bitmap_ptr, render_width, render_height, bitmap_stride);
}

extern "C" void RaycastNormalsCUDAF32(const void* pScene, const void* pInverseMVP, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride)
{
	cudaRender<float>(pScene, pInverseMVP, bitmap_ptr, render_width, render_height, bitmap_stride, [](const Scene<float>& device_scene, const Matrix44<float>& InverseMVP, void* device_bitmap_ptr, int render_width, int render_height, const dim3& grid, const dim3& threads) {
		cudaRaycastNormalsKernelF32<<<grid, threads>>>(device_scene, InverseMVP, device_bitmap_ptr, render_width, render_height, 4 * render_width);
	});
}

#ifdef USE_F64
__global__ void cudaRaycastNormalsKernelF64(const Scene<double>& pScene, Matrix44<double> inverse_mvp, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride) {
	cudaFill<double, RaytraceCUDA::RenderModeRaycastNormals<double>>(pScene, inverse_mvp, bitmap_ptr, render_width, render_height, bitmap_stride);
}

extern "C" void RaycastNormalsCUDAF64(const void* pScene, const void* pInverseMVP, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride)
{
	cudaRender<double>(pScene, pInverseMVP, bitmap_ptr, render_width, render_height, bitmap_stride, [](const Scene<double>& device_scene, const Matrix44<double>& InverseMVP, void* device_bitmap_ptr, int render_width, int render_height, const dim3& grid, const dim3& threads) {
		cudaRaycastNormalsKernelF64<<<grid, threads>>>(device_scene, InverseMVP, device_bitmap_ptr, render_width, render_height, 4 * render_width);
	});
}
#endif
#pragma endregion

#pragma region - Render Mode : Raycast Tangents -
__global__ void cudaRaycastTangentsKernelF32(const Scene<float>& pScene, Matrix44<float> inverse_mvp, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride) {
	cudaFill<float, RaytraceCUDA::RenderModeRaycastTangents<float>>(pScene, inverse_mvp, bitmap_ptr, render_width, render_height, bitmap_stride);
}

extern "C" void RaycastTangentsCUDAF32(const void* pScene, const void* pInverseMVP, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride)
{
	cudaRender<float>(pScene, pInverseMVP, bitmap_ptr, render_width, render_height, bitmap_stride, [](const Scene<float>& device_scene, const Matrix44<float>& InverseMVP, void* device_bitmap_ptr, int render_width, int render_height, const dim3& grid, const dim3& threads) {
		cudaRaycastTangentsKernelF32<<<grid, threads>>>(device_scene, InverseMVP, device_bitmap_ptr, render_width, render_height, 4 * render_width);
	});
}

#ifdef USE_F64
__global__ void cudaRaycastTangentsKernelF64(const Scene<double>& pScene, Matrix44<double> inverse_mvp, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride) {
	cudaFill<double, RaytraceCUDA::RenderModeRaycastTangents<double>>(pScene, inverse_mvp, bitmap_ptr, render_width, render_height, bitmap_stride);
}

extern "C" void RaycastTangentsCUDAF64(const void* pScene, const void* pInverseMVP, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride)
{
	cudaRender<double>(pScene, pInverseMVP, bitmap_ptr, render_width, render_height, bitmap_stride, [](const Scene<double>& device_scene, const Matrix44<double>& InverseMVP, void* device_bitmap_ptr, int render_width, int render_height, const dim3& grid, const dim3& threads) {
		cudaRaycastTangentsKernelF64<<<grid, threads>>>(device_scene, InverseMVP, device_bitmap_ptr, render_width, render_height, 4 * render_width);
	});
}
#endif
#pragma endregion

#pragma region - Render Mode : Raytrace -
__global__ void cudaRaytraceKernelF32(const Scene<float>& pScene, Matrix44<float> inverse_mvp, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride) {
	cudaFill<float, RaytraceCUDA::RenderModeRaytrace<float>, 2, 2>(pScene, inverse_mvp, bitmap_ptr, render_width, render_height, bitmap_stride);
}

extern "C" void RaytraceCUDAF32(const void* pScene, const void* pInverseMVP, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride)
{
	cudaRender<float>(pScene, pInverseMVP, bitmap_ptr, render_width, render_height, bitmap_stride, [](const Scene<float>& device_scene, const Matrix44<float>& InverseMVP, void* device_bitmap_ptr, int render_width, int render_height, const dim3& grid, const dim3& threads) {
		cudaRaytraceKernelF32<<<grid, threads>>>(device_scene, InverseMVP, device_bitmap_ptr, render_width, render_height, 4 * render_width);
	});
}

#ifdef USE_F64
__global__ void cudaRaytraceKernelF64(const Scene<double>& pScene, Matrix44<double> inverse_mvp, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride) {
	cudaFill<double, RaytraceCUDA::RenderModeRaytrace<double>, 2, 2>(pScene, inverse_mvp, bitmap_ptr, render_width, render_height, bitmap_stride);
}

extern "C" void RaytraceCUDAF64(const void* pScene, const void* pInverseMVP, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride)
{
	cudaRender<double>(pScene, pInverseMVP, bitmap_ptr, render_width, render_height, bitmap_stride, [](const Scene<double>& device_scene, const Matrix44<double>& InverseMVP, void* device_bitmap_ptr, int render_width, int render_height, const dim3& grid, const dim3& threads) {
		cudaRaytraceKernelF64<<<grid, threads>>>(device_scene, InverseMVP, device_bitmap_ptr, render_width, render_height, 4 * render_width);
	});
}
#endif
#pragma endregion

#pragma region - Render Mode : Debug Mesh -
__global__ void cudaDebugMeshKernelF32(const Scene<float>& pScene, Matrix44<float> inverse_mvp, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride) {
	cudaFill<float, RaytraceCUDA::RenderModeDebugMesh<float>>(pScene, inverse_mvp, bitmap_ptr, render_width, render_height, bitmap_stride);
}

extern "C" void DebugMeshCUDAF32(const void* pScene, const void* pInverseMVP, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride)
{
	cudaRender<float>(pScene, pInverseMVP, bitmap_ptr, render_width, render_height, bitmap_stride, [](const Scene<float>& device_scene, const Matrix44<float>& InverseMVP, void* device_bitmap_ptr, int render_width, int render_height, const dim3& grid, const dim3& threads) {
		cudaDebugMeshKernelF32<<<grid, threads>>>(device_scene, InverseMVP, device_bitmap_ptr, render_width, render_height, 4 * render_width);
	});
}

#ifdef USE_F64
__global__ void cudaDebugMeshKernelF64(const Scene<double>& pScene, Matrix44<double> inverse_mvp, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride) {
	cudaFill<double, RaytraceCUDA::RenderModeDebugMesh<double>>(pScene, inverse_mvp, bitmap_ptr, render_width, render_height, bitmap_stride);
}

extern "C" void DebugMeshCUDAF64(const void* pScene, const void* pInverseMVP, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride)
{
	cudaRender<double>(pScene, pInverseMVP, bitmap_ptr, render_width, render_height, bitmap_stride, [](const Scene<double>& device_scene, const Matrix44<double>& InverseMVP, void* device_bitmap_ptr, int render_width, int render_height, const dim3& grid, const dim3& threads) {
		cudaDebugMeshKernelF64<<<grid, threads>>>(device_scene, InverseMVP, device_bitmap_ptr, render_width, render_height, 4 * render_width);
	});
}
#endif
#pragma endregion

#pragma region - Render Mode : Ambient Occlusion -
template <typename FLOAT>
__global__ void cudaAOC(const Scene<FLOAT>& pScene, Matrix44<FLOAT> inverse_mvp, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride, int sample_offset, int sample_count) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	RaytraceCUDA::HemisampleHalton<FLOAT> hemisamples(sample_offset, sample_count);
	RaytraceCUDA::PixelSetScaledARGB<FLOAT> setpixel(bitmap_ptr, render_width, render_height, bitmap_stride, x, y, FLOAT(1) / hemisamples.Count());
	RaytraceCUDA::ComputePixelAOC<FLOAT>(pScene, inverse_mvp, setpixel, hemisamples);
}

template <typename FLOAT>
void AmbientOcclusionCUDA(const void* pScene, const void* pMVP, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride, int sample_offset, int sample_count)
{
	CudaMemory1DIn cudaScene((const Scene<FLOAT>*)pScene, ((const Scene<FLOAT>*)pScene)->FileSize);
	CudaMemory2DOut cudaBitmap(bitmap_ptr, bitmap_stride, sizeof(int) * render_width, render_height);
	dim3 grid((render_width + 15) / 16, (render_height + 15) / 16, 1);
	dim3 threads(16, 16, 1);
	cudaAOC<<<grid, threads>>>(*(const Scene<FLOAT>*)cudaScene.DeviceMemory(), *(Matrix44<FLOAT>*)pMVP, cudaBitmap.DeviceMemory(), render_width, render_height, sizeof(int) * render_width, sample_offset, sample_count);
}

extern "C" void AmbientOcclusionCUDAF32(const void* pScene, const void* pMVP, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride, int sample_offset, int sample_count)
{
	AmbientOcclusionCUDA<float>(pScene, pMVP, bitmap_ptr, render_width, render_height, bitmap_stride, sample_offset, sample_count);
}

#ifdef USE_F64
extern "C" void AmbientOcclusionCUDAF64(const void* pScene, const void* pMVP, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride, int sample_offset, int sample_count)
{
	AmbientOcclusionCUDA<double>(pScene, pMVP, bitmap_ptr, render_width, render_height, bitmap_stride, sample_offset, sample_count);
}
#endif
#pragma endregion

#pragma region - Render Mode : Ambient Occlusion (ToneMap) -
template <typename FLOAT>
__global__ void globalRescaleVec4(const Vector4<FLOAT>* accumulator_ptr, int accumulator_stride, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride, FLOAT scale) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x >= render_width || y >= render_height) return;
	Vector4<FLOAT>* pRaster_Accum = (Vector4<FLOAT>*)((unsigned char*)accumulator_ptr + accumulator_stride * y);
	unsigned int* pRaster_Bitmap = (unsigned int*)((unsigned char*)bitmap_ptr + bitmap_stride * y);
	Vector4<FLOAT>* pPixel_Accum = pRaster_Accum + x;
	unsigned int* pPixel_Bitmap = pRaster_Bitmap + x;
	Vector4<FLOAT> result = RaytraceCUDA::make_vector4(pPixel_Accum->x * scale, pPixel_Accum->y * scale, pPixel_Accum->z * scale, pPixel_Accum->w * scale);
	*pPixel_Bitmap = RaytraceCUDA::Vector4ToA8R8G8B8(result);
}

template <typename FLOAT>
void ToneMapCUDA(const Vector4<FLOAT>* accumulator_ptr, int accumulator_stride, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride, float rescale)
{
	CudaMemory2DIn cudaAccumulator(accumulator_ptr, accumulator_stride, sizeof(Vector4<FLOAT>) * render_width, render_height);
	CudaMemory2DOut cudaBitmap(bitmap_ptr, bitmap_stride, sizeof(int) * render_width, render_height);
	// Execute the tonemap kernel.
	dim3 grid((render_width + 15) / 16, (render_height + 15) / 16, 1);
	dim3 threads(16, 16, 1);
	globalRescaleVec4<<<grid, threads>>>((const Vector4<FLOAT>*)cudaAccumulator.DeviceMemory(), sizeof(Vector4<FLOAT>) * render_width, cudaBitmap.DeviceMemory(), render_width, render_height, sizeof(int) * render_width, rescale);
}

extern "C" void ToneMap(const void* accumulator_ptr, int accumulator_stride, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride, float rescale)
{
	ToneMapCUDA<float>((const Vector4<float>*)accumulator_ptr, accumulator_stride, bitmap_ptr, render_width, render_height, bitmap_stride, rescale);
}
#pragma endregion

#pragma region - Render Mode : Ambient Occlusion (FMP Buffered) - 
template <typename FLOAT>
__global__ void globalAmbientOcclusionFMPCUDA(const Scene<FLOAT>& pScene, Matrix44<FLOAT> inverse_mvp, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride, int sample_offset, int sample_count) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	RaytraceCUDA::PixelAccumulateVec4<FLOAT> setpixel(bitmap_ptr, render_width, render_height, bitmap_stride, x, y);
	RaytraceCUDA::HemisampleHalton<FLOAT> hemisamples(sample_offset, sample_count);
	RaytraceCUDA::ComputePixelAOC<FLOAT>(pScene, inverse_mvp, setpixel, hemisamples);
}

template <typename FLOAT>
void AmbientOcclusionFMPCUDA(const void* pScene, const void* pMVP, void* accumulator_ptr, int acc_width, int acc_height, int accumulator_stride, int sample_offset, int sample_count)
{
	CudaMemory1DIn cudaScene((const Scene<FLOAT>*)pScene, ((const Scene<FLOAT>*)pScene)->FileSize);
	CudaMemory2DInOut cudaAccumulator(accumulator_ptr, accumulator_stride, sizeof(Vector4<FLOAT>) * acc_width, acc_height);
	// Launch the accumulator kernel.
	int thread_tile = 16;
	dim3 grid((acc_width + thread_tile - 1) / thread_tile, (acc_height + thread_tile - 1) / thread_tile, 1);
	dim3 threads(thread_tile, thread_tile, 1);
	globalAmbientOcclusionFMPCUDA<<<grid, threads>>>(*(const Scene<FLOAT>*)cudaScene.DeviceMemory(), *(Matrix44<FLOAT>*)pMVP, cudaAccumulator.DeviceMemory(), acc_width, acc_height, sizeof(Vector4<FLOAT>) * acc_width, sample_offset, sample_count);
}

extern "C" void AmbientOcclusionFMPCUDAF32(const void* pScene, const void* pMVP, void* accumulator_ptr, int acc_width, int acc_height, int accumulator_stride, int sample_offset, int sample_count)
{
	AmbientOcclusionFMPCUDA<float>(pScene, pMVP, accumulator_ptr, acc_width, acc_height, accumulator_stride, sample_offset, sample_count);
}
#pragma endregion