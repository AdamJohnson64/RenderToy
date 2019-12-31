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
static void RenderImageCPU(const void* scene, const void* inverse_mvp, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride, int superx = 1, int supery = 1) {
	for (int y = 0; y < render_height; ++y) {
		for (int x = 0; x < render_width; ++x) {
			RaytraceCLI::PixelSetARGB<FLOAT> setpixel(bitmap_ptr, render_width, render_height, bitmap_stride, x, y);
			RaytraceCLI::ComputePixel<FLOAT, RENDERMODE>(*(Scene<FLOAT>*)scene, *(Matrix44<FLOAT>*)inverse_mvp, setpixel, superx, supery);
		}
	}
}

extern "C" void RaycastCPUF32(const void* scene, const void* inverse_mvp, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride) {
	RenderImageCPU<float, RaytraceCLI::RenderModeRaycast<float>>(scene, inverse_mvp, bitmap_ptr, render_width, render_height, bitmap_stride);
}

extern "C" void RaycastCPUF64(const void* scene, const void* inverse_mvp, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride) {
	RenderImageCPU<double, RaytraceCLI::RenderModeRaycast<double>>(scene, inverse_mvp, bitmap_ptr, render_width, render_height, bitmap_stride);
}

extern "C" void RaycastNormalsCPUF32(const void* scene, const void* inverse_mvp, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride) {
	RenderImageCPU<float, RaytraceCLI::RenderModeRaycastNormals<float>>(scene, inverse_mvp, bitmap_ptr, render_width, render_height, bitmap_stride);
}

extern "C" void RaycastTangentsCPUF32(const void* scene, const void* inverse_mvp, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride) {
	RenderImageCPU<float, RaytraceCLI::RenderModeRaycastTangents<float>>(scene, inverse_mvp, bitmap_ptr, render_width, render_height, bitmap_stride);
}

extern "C" void RaycastBitangentsCPUF32(const void* scene, const void* inverse_mvp, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride) {
	RenderImageCPU<float, RaytraceCLI::RenderModeRaycastBitangents<float>>(scene, inverse_mvp, bitmap_ptr, render_width, render_height, bitmap_stride);
}

extern "C" void RaytraceCPUF32(const void* scene, const void* inverse_mvp, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride) {
	RenderImageCPU<float, RaytraceCLI::RenderModeRaytrace<float>>(scene, inverse_mvp, bitmap_ptr, render_width, render_height, bitmap_stride);
}

extern "C" void RaytraceCPUF32AA(const void* scene, const void* inverse_mvp, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride, int superx, int supery) {
	RenderImageCPU<float, RaytraceCLI::RenderModeRaytrace<float>>(scene, inverse_mvp, bitmap_ptr, render_width, render_height, bitmap_stride, superx, supery);
}

extern "C" void AmbientOcclusionCPUF32(const void* scene, const void* inverse_mvp, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride) {
	for (int y = 0; y < render_height; ++y) {
		for (int x = 0; x < render_width; ++x) {
			RaytraceCLI::PixelSetARGB<float> setpixel(bitmap_ptr, render_width, render_height, bitmap_stride, x, y);
			RaytraceCLI::HemisampleHalton<float> hemisamples(0, 256);
			RaytraceCLI::ComputePixelAOC<float>(*(Scene<float>*)scene, *(Matrix44<float>*)inverse_mvp, setpixel, hemisamples);
		}
	}
}

extern "C" void RaycastNormalsCPUF64(const void* scene, const void* inverse_mvp, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride) {
	RenderImageCPU<double, RaytraceCLI::RenderModeRaycastNormals<double>>(scene, inverse_mvp, bitmap_ptr, render_width, render_height, bitmap_stride);
}

extern "C" void RaycastTangentsCPUF64(const void* scene, const void* inverse_mvp, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride) {
	RenderImageCPU<double, RaytraceCLI::RenderModeRaycastTangents<double>>(scene, inverse_mvp, bitmap_ptr, render_width, render_height, bitmap_stride);
}

extern "C" void RaycastBitangentsCPUF64(const void* scene, const void* inverse_mvp, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride) {
	RenderImageCPU<double, RaytraceCLI::RenderModeRaycastBitangents<double>>(scene, inverse_mvp, bitmap_ptr, render_width, render_height, bitmap_stride);
}

extern "C" void RaytraceCPUF64(const void* scene, const void* inverse_mvp, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride) {
	RenderImageCPU<double, RaytraceCLI::RenderModeRaytrace<double>>(scene, inverse_mvp, bitmap_ptr, render_width, render_height, bitmap_stride);
}

extern "C" void AmbientOcclusionCPUF64(const void* scene, const void* inverse_mvp, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride) {
	for (int y = 0; y < render_height; ++y) {
		for (int x = 0; x < render_width; ++x) {
			RaytraceCLI::PixelSetARGB<double> setpixel(bitmap_ptr, render_width, render_height, bitmap_stride, x, y);
			RaytraceCLI::HemisampleHalton<double> hemisamples(0, 256);
			RaytraceCLI::ComputePixelAOC<double>(*(Scene<double>*)scene, *(Matrix44<double>*)inverse_mvp, setpixel, hemisamples);
		}
	}
}