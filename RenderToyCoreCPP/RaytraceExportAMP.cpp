////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// This file contains the core AMP implementation of the raytracer.
////////////////////////////////////////////////////////////////////////////////

#include <amp.h>
#include <amp_math.h>
#include "Raytrace.h"

namespace RaytraceCX {
	using namespace Concurrency::fast_math;
#define DEVICE_PREFIX
#define DEVICE_SUFFIX restrict(amp)
#include "Raytrace.inc"
#undef DEVICE_SUFFIX
#undef DEVICE_PREFIX
}

template<typename FLOAT>
struct SetPixelAMP {
	SetPixelAMP(const Concurrency::array_view<int, 2>& data, Concurrency::index<2>& index) restrict(amp) : data(data), index(index) {}
	int GetX() const restrict(amp) { return index[1]; }
	int GetY() const restrict(amp) { return index[0]; }
	int GetWidth() const restrict(amp) { return data.extent[1]; }
	int GetHeight() const restrict(amp) { return data.extent[0]; }
	void Do(const Vector4<FLOAT>& color) const restrict(amp) {
		unsigned int r = unsigned int(color.x < 0 ? 0 : (color.x > 1 ? 1 : color.x) * 255);
		unsigned int g = unsigned int(color.y < 0 ? 0 : (color.y > 1 ? 1 : color.y) * 255);
		unsigned int b = unsigned int(color.z < 0 ? 0 : (color.z > 1 ? 1 : color.z) * 255);
		unsigned int a = unsigned int(color.w < 0 ? 0 : (color.w > 1 ? 1 : color.w) * 255);
		data[index] = (a << 24) | (r << 16) | (g << 8) | (b << 0);
	}
	const Concurrency::array_view<int, 2> data;
	const Concurrency::index<2> index;
};

template <typename FLOAT>
void AMPExecutor(const void* scene, const void* inverse_mvp, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride, std::function<void(Concurrency::array_view<int, 1> view_scene, Concurrency::array_view<FLOAT, 1> view_imvp, Concurrency::array_view<int, 2> view_bitmap, int bitmap_width, int bitmap_height, int bitmap_stride)> exec) {
	Concurrency::array_view<int, 1> view_scene(((Scene<FLOAT>*)scene)->FileSize / sizeof(int), (int*)scene);
	Concurrency::array_view<FLOAT, 1> view_imvp(4 * 4 * sizeof(FLOAT), (FLOAT*)inverse_mvp);
	std::unique_ptr<int[]> result(new int[bitmap_width * bitmap_height]);
	Concurrency::array_view<int, 2> view_bitmap(bitmap_height, bitmap_width, result.get());
	view_bitmap.discard_data();
	exec(view_scene, view_imvp, view_bitmap, bitmap_width, bitmap_height, bitmap_stride);
	view_bitmap.synchronize();
	for (int y = 0; y < bitmap_height; ++y) {
		memcpy((uint8*)bitmap_ptr + bitmap_stride * y, &result[bitmap_width * y], sizeof(int) * bitmap_width);
	}
}

extern "C" void RaycastAMPF32(const void* scene, const void* inverse_mvp, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {
	AMPExecutor<float>(scene, inverse_mvp, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, [](Concurrency::array_view<int, 1> view_scene, Concurrency::array_view<float, 1> view_imvp, Concurrency::array_view<int, 2> view_bitmap, int bitmap_width, int bitmap_height, int bitmap_stride) {
		Concurrency::parallel_for_each(view_bitmap.extent, [=](Concurrency::index<2> idx) restrict(amp) {
			SetPixelAMP<float> setpixel(view_bitmap, idx);
			RaytraceCX::ComputePixel<float, RaytraceCX::DoRaycast<float>>(*(Scene<float>*)&view_scene[0], *(Matrix44<float>*)&view_imvp[0], setpixel);
		});
	});
}

extern "C" void RaycastNormalsAMPF32(const void* scene, const void* inverse_mvp, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {
	AMPExecutor<float>(scene, inverse_mvp, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, [](Concurrency::array_view<int, 1> view_scene, Concurrency::array_view<float, 1> view_imvp, Concurrency::array_view<int, 2> view_bitmap, int bitmap_width, int bitmap_height, int bitmap_stride) {
		Concurrency::parallel_for_each(view_bitmap.extent, [=](Concurrency::index<2> idx) restrict(amp) {
			SetPixelAMP<float> setpixel(view_bitmap, idx);
			RaytraceCX::ComputePixel<float, RaytraceCX::DoRaycastNormals<float>>(*(Scene<float>*)&view_scene[0], *(Matrix44<float>*)&view_imvp[0], setpixel);
		});
	});
}

extern "C" void RaycastTangentsAMPF32(const void* scene, const void* inverse_mvp, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {
	AMPExecutor<float>(scene, inverse_mvp, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, [](Concurrency::array_view<int, 1> view_scene, Concurrency::array_view<float, 1> view_imvp, Concurrency::array_view<int, 2> view_bitmap, int bitmap_width, int bitmap_height, int bitmap_stride) {
		Concurrency::parallel_for_each(view_bitmap.extent, [=](Concurrency::index<2> idx) restrict(amp) {
			SetPixelAMP<float> setpixel(view_bitmap, idx);
			RaytraceCX::ComputePixel<float, RaytraceCX::DoRaycastTangents<float>>(*(Scene<float>*)&view_scene[0], *(Matrix44<float>*)&view_imvp[0], setpixel);
		});
	});
}

extern "C" void RaycastBitangentsAMPF32(const void* scene, const void* inverse_mvp, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {
	AMPExecutor<float>(scene, inverse_mvp, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, [](Concurrency::array_view<int, 1> view_scene, Concurrency::array_view<float, 1> view_imvp, Concurrency::array_view<int, 2> view_bitmap, int bitmap_width, int bitmap_height, int bitmap_stride) {
		Concurrency::parallel_for_each(view_bitmap.extent, [=](Concurrency::index<2> idx) restrict(amp) {
			SetPixelAMP<float> setpixel(view_bitmap, idx);
			RaytraceCX::ComputePixel<float, RaytraceCX::DoRaycastBitangents<float>>(*(Scene<float>*)&view_scene[0], *(Matrix44<float>*)&view_imvp[0], setpixel);
		});
	});
}

extern "C" void RaytraceAMPF32(const void* scene, const void* inverse_mvp, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {
	AMPExecutor<float>(scene, inverse_mvp, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, [](Concurrency::array_view<int, 1> view_scene, Concurrency::array_view<float, 1> view_imvp, Concurrency::array_view<int, 2> view_bitmap, int bitmap_width, int bitmap_height, int bitmap_stride) {
		Concurrency::parallel_for_each(view_bitmap.extent, [=](Concurrency::index<2> idx) restrict(amp) {
			SetPixelAMP<float> setpixel(view_bitmap, idx);
			RaytraceCX::ComputePixel<float, RaytraceCX::DoRaytrace<float, 1>>(*(Scene<float>*)&view_scene[0], *(Matrix44<float>*)&view_imvp[0], setpixel);
		});
	});
}