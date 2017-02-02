////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// This file contains the core CPU implementation of the raytracer and the
// fake CUDA implementation if CUDA is not installed.
////////////////////////////////////////////////////////////////////////////////

#include <math.h>
#include <cfloat>
#include "Raytrace.h"

namespace RaytraceCPP {
	#define DEVICE_PROTO
	#include "Raytrace.inc"
	#undef DEVICE_PROTO
}

extern "C" void RaycastCPU(void* pScene, void* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {
	for (int y = 0; y < bitmap_height; ++y) {
		for (int x = 0; x < bitmap_width; ++x) {
			RaytraceCPP::cudaFill2<double, RaytraceCPP::DoRaycast<double>, 1, 1>((Scene<double>*)pScene, *(Matrix44<double>*)pInverseMVP, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, x, y);
		}
	}
}

extern "C" void RaycastNormalsCPU(void* pScene, void* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {
	for (int y = 0; y < bitmap_height; ++y) {
		for (int x = 0; x < bitmap_width; ++x) {
			RaytraceCPP::cudaFill2<double, RaytraceCPP::DoRaycastNormals<double>, 1, 1>((Scene<double>*)pScene, *(Matrix44<double>*)pInverseMVP, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, x, y);
		}
	}
}

extern "C" void RaycastTangentsCPU(void* pScene, void* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {
	for (int y = 0; y < bitmap_height; ++y) {
		for (int x = 0; x < bitmap_width; ++x) {
			RaytraceCPP::cudaFill2<double, RaytraceCPP::DoRaycastTangents<double>, 1, 1>((Scene<double>*)pScene, *(Matrix44<double>*)pInverseMVP, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, x, y);
		}
	}
}

extern "C" void RaycastBitangentsCPU(void* pScene, void* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {
	for (int y = 0; y < bitmap_height; ++y) {
		for (int x = 0; x < bitmap_width; ++x) {
			RaytraceCPP::cudaFill2<double, RaytraceCPP::DoRaycastBitangents<double>, 1, 1>((Scene<double>*)pScene, *(Matrix44<double>*)pInverseMVP, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, x, y);
		}
	}
}

extern "C" void RaytraceCPUF32(void* pScene, void* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {
	for (int y = 0; y < bitmap_height; ++y) {
		for (int x = 0; x < bitmap_width; ++x) {
			RaytraceCPP::cudaFill2<float, RaytraceCPP::DoRaytrace<float>, 1, 1>((Scene<float>*)pScene, *(Matrix44<float>*)pInverseMVP, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, x, y);
		}
	}
}

extern "C" void RaytraceCPUF64(void* pScene, void* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {
	for (int y = 0; y < bitmap_height; ++y) {
		for (int x = 0; x < bitmap_width; ++x) {
			RaytraceCPP::cudaFill2<double, RaytraceCPP::DoRaytrace<double>, 1, 1>((Scene<double>*)pScene, *(Matrix44<double>*)pInverseMVP, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, x, y);
		}
	}
}