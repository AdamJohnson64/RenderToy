////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

#include <math.h>
#include <cfloat>

struct double3 { double x, y, z; };
struct double4 { double x, y, z, w; };

double3 make_double3(double x, double y, double z) { return double3{ x, y, z }; }
double4 make_double4(double x, double y, double z, double w) { return double4{ x, y, z, w }; }
double norm3d(double x, double y, double z) { return sqrt(x * x + y * y + z * z); }
double rnorm3d(double x, double y, double z) { return 1 / norm3d(x, y, z); }

#include "Raytrace.h"

namespace RaytraceCPP {
	#define DEVICE_PROTO
	#include "Raytrace.inc"
	#undef DEVICE_PROTO
}

extern "C" void CPURaycast(void* pScene, double* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {
	for (int y = 0; y < bitmap_height; ++y) {
		for (int x = 0; x < bitmap_width; ++x) {
			RaytraceCPP::cudaFill2<RaytraceCPP::DoRaycast, 1, 1>((Scene*)pScene, *(Matrix4D*)pInverseMVP, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, x, y);
		}
	}
}

extern "C" void CPURaycastNormals(void* pScene, double* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {
	for (int y = 0; y < bitmap_height; ++y) {
		for (int x = 0; x < bitmap_width; ++x) {
			RaytraceCPP::cudaFill2<RaytraceCPP::DoRaycastNormals, 1, 1>((Scene*)pScene, *(Matrix4D*)pInverseMVP, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, x, y);
		}
	}
}

extern "C" void CPURaycastTangents(void* pScene, double* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {
	for (int y = 0; y < bitmap_height; ++y) {
		for (int x = 0; x < bitmap_width; ++x) {
			RaytraceCPP::cudaFill2<RaytraceCPP::DoRaycastTangents, 1, 1>((Scene*)pScene, *(Matrix4D*)pInverseMVP, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, x, y);
		}
	}
}

extern "C" void CPURaycastBitangents(void* pScene, double* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {
	for (int y = 0; y < bitmap_height; ++y) {
		for (int x = 0; x < bitmap_width; ++x) {
			RaytraceCPP::cudaFill2<RaytraceCPP::DoRaycastBitangents, 1, 1>((Scene*)pScene, *(Matrix4D*)pInverseMVP, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, x, y);
		}
	}
}

extern "C" void CPURaytrace(void* pScene, double* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {
	for (int y = 0; y < bitmap_height; ++y) {
		for (int x = 0; x < bitmap_width; ++x) {
			RaytraceCPP::cudaFill2<RaytraceCPP::DoRaytrace, 1, 1>((Scene*)pScene, *(Matrix4D*)pInverseMVP, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, x, y);
		}
	}
}