////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

#include <math.h>
#include <cfloat>
#include "Raytrace.h"

namespace RaytraceCPP {
	#define DEVICE_PROTO
	#include "Raytrace.inc"
	#undef DEVICE_PROTO
}

extern "C" void CPURaycast(void* pScene, void* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {
	for (int y = 0; y < bitmap_height; ++y) {
		for (int x = 0; x < bitmap_width; ++x) {
			RaytraceCPP::cudaFill2<double, RaytraceCPP::DoRaycast<double>, 1, 1>((Scene<double>*)pScene, *(Matrix44<double>*)pInverseMVP, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, x, y);
		}
	}
}

extern "C" void CPURaycastNormals(void* pScene, void* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {
	for (int y = 0; y < bitmap_height; ++y) {
		for (int x = 0; x < bitmap_width; ++x) {
			RaytraceCPP::cudaFill2<double, RaytraceCPP::DoRaycastNormals<double>, 1, 1>((Scene<double>*)pScene, *(Matrix44<double>*)pInverseMVP, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, x, y);
		}
	}
}

extern "C" void CPURaycastTangents(void* pScene, void* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {
	for (int y = 0; y < bitmap_height; ++y) {
		for (int x = 0; x < bitmap_width; ++x) {
			RaytraceCPP::cudaFill2<double, RaytraceCPP::DoRaycastTangents<double>, 1, 1>((Scene<double>*)pScene, *(Matrix44<double>*)pInverseMVP, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, x, y);
		}
	}
}

extern "C" void CPURaycastBitangents(void* pScene, void* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {
	for (int y = 0; y < bitmap_height; ++y) {
		for (int x = 0; x < bitmap_width; ++x) {
			RaytraceCPP::cudaFill2<double, RaytraceCPP::DoRaycastBitangents<double>, 1, 1>((Scene<double>*)pScene, *(Matrix44<double>*)pInverseMVP, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, x, y);
		}
	}
}

extern "C" void CPUF32Raytrace(void* pScene, void* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {
	for (int y = 0; y < bitmap_height; ++y) {
		for (int x = 0; x < bitmap_width; ++x) {
			RaytraceCPP::cudaFill2<float, RaytraceCPP::DoRaytrace<float>, 1, 1>((Scene<float>*)pScene, *(Matrix44<float>*)pInverseMVP, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, x, y);
		}
	}
}

extern "C" void CPUF64Raytrace(void* pScene, void* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {
	for (int y = 0; y < bitmap_height; ++y) {
		for (int x = 0; x < bitmap_width; ++x) {
			RaytraceCPP::cudaFill2<double, RaytraceCPP::DoRaytrace<double>, 1, 1>((Scene<double>*)pScene, *(Matrix44<double>*)pInverseMVP, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, x, y);
		}
	}
}