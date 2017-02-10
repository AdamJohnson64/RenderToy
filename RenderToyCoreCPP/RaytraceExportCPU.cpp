////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// This file contains the core CPU implementation of the raytracer.
////////////////////////////////////////////////////////////////////////////////

#include <math.h>
#include <cfloat>
#include "Raytrace.h"

namespace RaytraceCLI {
	#define DEVICE_PREFIX
	#define DEVICE_SUFFIX
	#include "Raytrace.inc"
	#undef DEVICE_SUFFIX
	#undef DEVICE_PREFIX
}

template <typename FLOAT, typename RENDERMODE>
static void RenderImageCPU(const void* scene, const void* inverse_mvp, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride, int superx = 1, int supery = 1) {
	for (int y = 0; y < bitmap_height; ++y) {
		for (int x = 0; x < bitmap_width; ++x) {
			RaytraceCLI::SetPixel<FLOAT> setpixel(bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, x, y);
			RaytraceCLI::ComputePixel<FLOAT, RENDERMODE>(*(Scene<FLOAT>*)scene, *(Matrix44<FLOAT>*)inverse_mvp, setpixel, superx, supery);
		}
	}
}

extern "C" void RaycastCPUF32(const void* scene, const void* inverse_mvp, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {
	RenderImageCPU<float, RaytraceCLI::RenderModeRaycast<float>>(scene, inverse_mvp, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
}

extern "C" void RaycastCPUF64(const void* scene, const void* inverse_mvp, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {
	RenderImageCPU<double, RaytraceCLI::RenderModeRaycast<double>>(scene, inverse_mvp, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
}

extern "C" void RaycastNormalsCPUF32(const void* scene, const void* inverse_mvp, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {
	RenderImageCPU<float, RaytraceCLI::RenderModeRaycastNormals<float>>(scene, inverse_mvp, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
}

extern "C" void RaycastNormalsCPUF64(const void* scene, const void* inverse_mvp, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {
	RenderImageCPU<double, RaytraceCLI::RenderModeRaycastNormals<double>>(scene, inverse_mvp, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
}

extern "C" void RaycastTangentsCPUF32(const void* scene, const void* inverse_mvp, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {
	RenderImageCPU<float, RaytraceCLI::RenderModeRaycastTangents<float>>(scene, inverse_mvp, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
}

extern "C" void RaycastTangentsCPUF64(const void* scene, const void* inverse_mvp, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {
	RenderImageCPU<double, RaytraceCLI::RenderModeRaycastTangents<double>>(scene, inverse_mvp, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
}

extern "C" void RaycastBitangentsCPUF32(const void* scene, const void* inverse_mvp, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {
	RenderImageCPU<float, RaytraceCLI::RenderModeRaycastBitangents<float>>(scene, inverse_mvp, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
}

extern "C" void RaycastBitangentsCPUF64(const void* scene, const void* inverse_mvp, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {
	RenderImageCPU<double, RaytraceCLI::RenderModeRaycastBitangents<double>>(scene, inverse_mvp, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
}

extern "C" void RaytraceCPUF32(const void* scene, const void* inverse_mvp, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {
	RenderImageCPU<float, RaytraceCLI::RenderModeRaytrace<float>>(scene, inverse_mvp, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
}

extern "C" void RaytraceCPUF64(const void* scene, const void* inverse_mvp, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {
	RenderImageCPU<double, RaytraceCLI::RenderModeRaytrace<double>>(scene, inverse_mvp, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
}

extern "C" void RaytraceCPUF64AA(const void* scene, const void* inverse_mvp, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride, int superx, int supery) {
	RenderImageCPU<double, RaytraceCLI::RenderModeRaytrace<double>>(scene, inverse_mvp, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, superx, supery);
}

extern "C" void AmbientOcclusionCPUF32(const void* scene, const void* inverse_mvp, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride, int hemisample_count, const void* hemisamples) {
	for (int y = 0; y < bitmap_height; ++y) {
		for (int x = 0; x < bitmap_width; ++x) {
			RaytraceCLI::SetPixel<float> setpixel(bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, x, y);
			RaytraceCLI::ComputePixelAOC<float>(*(Scene<float>*)scene, *(Matrix44<float>*)inverse_mvp, setpixel, hemisample_count, (Vector4<float>*)hemisamples);
		}
	}
}

extern "C" void AmbientOcclusionCPUF64(const void* scene, const void* inverse_mvp, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride, int hemisample_count, const void* hemisamples) {
	for (int y = 0; y < bitmap_height; ++y) {
		for (int x = 0; x < bitmap_width; ++x) {
			RaytraceCLI::SetPixel<double> setpixel(bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, x, y);
			RaytraceCLI::ComputePixelAOC<double>(*(Scene<double>*)scene, *(Matrix44<double>*)inverse_mvp, setpixel, hemisample_count, (Vector4<double>*)hemisamples);
		}
	}
}