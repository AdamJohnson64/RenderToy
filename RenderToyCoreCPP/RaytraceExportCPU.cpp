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

extern "C" void RaycastCPUF32(const void* pScene, const void* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {
	for (int y = 0; y < bitmap_height; ++y) {
		for (int x = 0; x < bitmap_width; ++x) {
			RaytraceCLI::SetPixel<float> setpixel(bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, x, y);
			RaytraceCLI::ComputePixel<float, RaytraceCLI::RenderModeRaycast<float>>(*(Scene<float>*)pScene, *(Matrix44<float>*)pInverseMVP, setpixel);
		}
	}
}

extern "C" void RaycastCPUF64(const void* pScene, const void* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {
	for (int y = 0; y < bitmap_height; ++y) {
		for (int x = 0; x < bitmap_width; ++x) {
			RaytraceCLI::SetPixel<double> setpixel(bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, x, y);
			RaytraceCLI::ComputePixel<double, RaytraceCLI::RenderModeRaycast<double>>(*(Scene<double>*)pScene, *(Matrix44<double>*)pInverseMVP, setpixel);
		}
	}
}

extern "C" void RaycastNormalsCPUF32(const void* pScene, const void* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {
	for (int y = 0; y < bitmap_height; ++y) {
		for (int x = 0; x < bitmap_width; ++x) {
			RaytraceCLI::SetPixel<float> setpixel(bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, x, y);
			RaytraceCLI::ComputePixel<float, RaytraceCLI::RenderModeRaycastNormals<float>>(*(Scene<float>*)pScene, *(Matrix44<float>*)pInverseMVP, setpixel);
		}
	}
}

extern "C" void RaycastNormalsCPUF64(const void* pScene, const void* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {
	for (int y = 0; y < bitmap_height; ++y) {
		for (int x = 0; x < bitmap_width; ++x) {
			RaytraceCLI::SetPixel<double> setpixel(bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, x, y);
			RaytraceCLI::ComputePixel<double, RaytraceCLI::RenderModeRaycastNormals<double>>(*(Scene<double>*)pScene, *(Matrix44<double>*)pInverseMVP, setpixel);
		}
	}
}

extern "C" void RaycastTangentsCPUF32(const void* pScene, const void* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {
	for (int y = 0; y < bitmap_height; ++y) {
		for (int x = 0; x < bitmap_width; ++x) {
			RaytraceCLI::SetPixel<float> setpixel(bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, x, y);
			RaytraceCLI::ComputePixel<float, RaytraceCLI::RenderModeRaycastTangents<float>>(*(Scene<float>*)pScene, *(Matrix44<float>*)pInverseMVP, setpixel);
		}
	}
}

extern "C" void RaycastTangentsCPUF64(const void* pScene, const void* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {
	for (int y = 0; y < bitmap_height; ++y) {
		for (int x = 0; x < bitmap_width; ++x) {
			RaytraceCLI::SetPixel<double> setpixel(bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, x, y);
			RaytraceCLI::ComputePixel<double, RaytraceCLI::RenderModeRaycastTangents<double>>(*(Scene<double>*)pScene, *(Matrix44<double>*)pInverseMVP, setpixel);
		}
	}
}

extern "C" void RaycastBitangentsCPUF32(const void* pScene, const void* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {
	for (int y = 0; y < bitmap_height; ++y) {
		for (int x = 0; x < bitmap_width; ++x) {
			RaytraceCLI::SetPixel<float> setpixel(bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, x, y);
			RaytraceCLI::ComputePixel<float, RaytraceCLI::RenderModeRaycastBitangents<float>>(*(Scene<float>*)pScene, *(Matrix44<float>*)pInverseMVP, setpixel);
		}
	}
}

extern "C" void RaycastBitangentsCPUF64(const void* pScene, const void* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {
	for (int y = 0; y < bitmap_height; ++y) {
		for (int x = 0; x < bitmap_width; ++x) {
			RaytraceCLI::SetPixel<double> setpixel(bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, x, y);
			RaytraceCLI::ComputePixel<double, RaytraceCLI::RenderModeRaycastBitangents<double>>(*(Scene<double>*)pScene, *(Matrix44<double>*)pInverseMVP, setpixel);
		}
	}
}

extern "C" void RaytraceCPUF32(const void* pScene, const void* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {
	for (int y = 0; y < bitmap_height; ++y) {
		for (int x = 0; x < bitmap_width; ++x) {
			RaytraceCLI::SetPixel<float> setpixel(bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, x, y);
			RaytraceCLI::ComputePixel<float, RaytraceCLI::RenderModeRaytrace<float>>(*(Scene<float>*)pScene, *(Matrix44<float>*)pInverseMVP, setpixel);
		}
	}
}

extern "C" void RaytraceCPUF64(const void* pScene, const void* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {
	for (int y = 0; y < bitmap_height; ++y) {
		for (int x = 0; x < bitmap_width; ++x) {
			RaytraceCLI::SetPixel<double> setpixel(bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, x, y);
			RaytraceCLI::ComputePixel<double, RaytraceCLI::RenderModeRaytrace<double>>(*(Scene<double>*)pScene, *(Matrix44<double>*)pInverseMVP, setpixel);
		}
	}
}

extern "C" void RaytraceCPUF64AA(const void* pScene, const void* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride, int superx, int supery) {
	for (int y = 0; y < bitmap_height; ++y) {
		for (int x = 0; x < bitmap_width; ++x) {
			RaytraceCLI::SetPixel<double> setpixel(bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, x, y);
			RaytraceCLI::ComputePixel<double, RaytraceCLI::RenderModeRaytrace<double>>(*(Scene<double>*)pScene, *(Matrix44<double>*)pInverseMVP, setpixel, superx, supery);
		}
	}
}