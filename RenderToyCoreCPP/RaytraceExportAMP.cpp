////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
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

template<typename FLOAT>
struct SetPixelAMP {
	SetPixelAMP(const Concurrency::array_view<int, 2>& data, Concurrency::index<2>& index) restrict(amp) : data(data), index(index) {}
	int GetX() const restrict(amp) { return index[1]; }
	int GetY() const restrict(amp) { return index[0]; }
	int GetWidth() const restrict(amp) { return data.extent[1]; }
	int GetHeight() const restrict(amp) { return data.extent[0]; }
	void PutPixel(const Vector4<FLOAT>& color) const restrict(amp) {
		data[index] = Vector4ToA8R8G8B8(color);
	}
	const Concurrency::array_view<int, 2> data;
	const Concurrency::index<2> index;
};

/*
template<typename FLOAT>
struct SetPixelScaledAMP {
	SetPixelScaledAMP(const Concurrency::array_view<int, 2>& data, Concurrency::index<2>& index, FLOAT scale) restrict(amp) : data(data), index(index), scale(scale) {}
	int GetX() const restrict(amp) { return index[1]; }
	int GetY() const restrict(amp) { return index[0]; }
	int GetWidth() const restrict(amp) { return data.extent[1]; }
	int GetHeight() const restrict(amp) { return data.extent[0]; }
	void PutPixel(const Vector4<FLOAT>& color) const restrict(amp) {
		data[index] = Vector4ToA8R8G8B8(color * scale);
	}
	const Concurrency::array_view<int, 2> data;
	const Concurrency::index<2> index;
	FLOAT scale;
};
*/
}

template <typename FLOAT>
void AMPExecutor(const void* scene, const void* inverse_mvp, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride, std::function<void(Concurrency::array_view<int, 1> view_scene, Concurrency::array_view<FLOAT, 1> view_imvp, Concurrency::array_view<int, 2> view_bitmap, int render_width, int render_height, int bitmap_stride)> exec) {
	Concurrency::array_view<int, 1> view_scene(((Scene<FLOAT>*)scene)->FileSize / sizeof(int), (int*)scene);
	Concurrency::array_view<FLOAT, 1> view_imvp(4 * 4 * sizeof(FLOAT), (FLOAT*)inverse_mvp);
	Concurrency::array_view<int, 2> view_bitmap(render_height, bitmap_stride / sizeof(int), (int*)bitmap_ptr);
	view_bitmap.discard_data();
	exec(view_scene, view_imvp, view_bitmap, render_width, render_height, bitmap_stride);
	view_bitmap.synchronize();
}

extern "C" void RaycastAMPF32(const void* scene, const void* inverse_mvp, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride) {
	AMPExecutor<float>(scene, inverse_mvp, bitmap_ptr, render_width, render_height, bitmap_stride, [](Concurrency::array_view<int, 1> view_scene, Concurrency::array_view<float, 1> view_imvp, Concurrency::array_view<int, 2> view_bitmap, int render_width, int render_height, int bitmap_stride) {
		Concurrency::parallel_for_each(view_bitmap.extent, [=](Concurrency::index<2> idx) restrict(amp) {
			RaytraceCX::SetPixelAMP<float> setpixel(view_bitmap, idx);
			RaytraceCX::ComputePixel<float, RaytraceCX::RenderModeRaycast<float>>(*(Scene<float>*)&view_scene[0], *(Matrix44<float>*)&view_imvp[0], setpixel);
		});
	});
}

extern "C" void RaycastNormalsAMPF32(const void* scene, const void* inverse_mvp, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride) {
	AMPExecutor<float>(scene, inverse_mvp, bitmap_ptr, render_width, render_height, bitmap_stride, [](Concurrency::array_view<int, 1> view_scene, Concurrency::array_view<float, 1> view_imvp, Concurrency::array_view<int, 2> view_bitmap, int render_width, int render_height, int bitmap_stride) {
		Concurrency::parallel_for_each(view_bitmap.extent, [=](Concurrency::index<2> idx) restrict(amp) {
			RaytraceCX::SetPixelAMP<float> setpixel(view_bitmap, idx);
			RaytraceCX::ComputePixel<float, RaytraceCX::RenderModeRaycastNormals<float>>(*(Scene<float>*)&view_scene[0], *(Matrix44<float>*)&view_imvp[0], setpixel);
		});
	});
}

extern "C" void RaycastTangentsAMPF32(const void* scene, const void* inverse_mvp, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride) {
	AMPExecutor<float>(scene, inverse_mvp, bitmap_ptr, render_width, render_height, bitmap_stride, [](Concurrency::array_view<int, 1> view_scene, Concurrency::array_view<float, 1> view_imvp, Concurrency::array_view<int, 2> view_bitmap, int render_width, int render_height, int bitmap_stride) {
		Concurrency::parallel_for_each(view_bitmap.extent, [=](Concurrency::index<2> idx) restrict(amp) {
			RaytraceCX::SetPixelAMP<float> setpixel(view_bitmap, idx);
			RaytraceCX::ComputePixel<float, RaytraceCX::RenderModeRaycastTangents<float>>(*(Scene<float>*)&view_scene[0], *(Matrix44<float>*)&view_imvp[0], setpixel);
		});
	});
}

extern "C" void RaycastBitangentsAMPF32(const void* scene, const void* inverse_mvp, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride) {
	AMPExecutor<float>(scene, inverse_mvp, bitmap_ptr, render_width, render_height, bitmap_stride, [](Concurrency::array_view<int, 1> view_scene, Concurrency::array_view<float, 1> view_imvp, Concurrency::array_view<int, 2> view_bitmap, int render_width, int render_height, int bitmap_stride) {
		Concurrency::parallel_for_each(view_bitmap.extent, [=](Concurrency::index<2> idx) restrict(amp) {
			RaytraceCX::SetPixelAMP<float> setpixel(view_bitmap, idx);
			RaytraceCX::ComputePixel<float, RaytraceCX::RenderModeRaycastBitangents<float>>(*(Scene<float>*)&view_scene[0], *(Matrix44<float>*)&view_imvp[0], setpixel);
		});
	});
}

extern "C" void RaytraceAMPF32(const void* scene, const void* inverse_mvp, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride) {
	AMPExecutor<float>(scene, inverse_mvp, bitmap_ptr, render_width, render_height, bitmap_stride, [](Concurrency::array_view<int, 1> view_scene, Concurrency::array_view<float, 1> view_imvp, Concurrency::array_view<int, 2> view_bitmap, int render_width, int render_height, int bitmap_stride) {
		Concurrency::parallel_for_each(view_bitmap.extent, [=](Concurrency::index<2> idx) restrict(amp) {
			RaytraceCX::SetPixelAMP<float> setpixel(view_bitmap, idx);
			RaytraceCX::ComputePixel<float, RaytraceCX::RenderModeRaytrace<float, 0>>(*(Scene<float>*)&view_scene[0], *(Matrix44<float>*)&view_imvp[0], setpixel);
		});
	});
}

extern "C" void DebugMeshAMPF32(const void* scene, const void* inverse_mvp, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride) {
	AMPExecutor<float>(scene, inverse_mvp, bitmap_ptr, render_width, render_height, bitmap_stride, [](Concurrency::array_view<int, 1> view_scene, Concurrency::array_view<float, 1> view_imvp, Concurrency::array_view<int, 2> view_bitmap, int render_width, int render_height, int bitmap_stride) {
		Concurrency::parallel_for_each(view_bitmap.extent, [=](Concurrency::index<2> idx) restrict(amp) {
			RaytraceCX::SetPixelAMP<float> setpixel(view_bitmap, idx);
			RaytraceCX::ComputePixel<float, RaytraceCX::RenderModeDebugMesh<float>>(*(Scene<float>*)&view_scene[0], *(Matrix44<float>*)&view_imvp[0], setpixel);
		});
	});
}

/*
extern "C" void AmbientOcclusionAMPF32(const void* scene, const void* inverse_mvp, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride, int sample_offset, int sample_count) {
	Concurrency::array_view<int, 1> view_scene(((Scene<float>*)scene)->FileSize / sizeof(int), (int*)scene);
	Concurrency::array_view<float, 1> view_imvp(4 * 4, (float*)inverse_mvp);
	std::unique_ptr<int[]> result(new int[render_width * render_height]);
	Concurrency::array_view<int, 2> view_bitmap(render_height, render_width, result.get());
	view_bitmap.discard_data();
	Concurrency::parallel_for_each(view_bitmap.extent, [=](Concurrency::index<2> idx) restrict(amp) {
		RaytraceCX::SetPixelScaledAMP<float> setpixel(view_bitmap, idx, 1.0f / sample_count);
		RaytraceCX::HemisampleHalton<float> hemisamples(sample_offset, sample_count);
		RaytraceCX::ComputePixelAOC<float>(*(Scene<float>*)&view_scene[0], *(Matrix44<float>*)&view_imvp[0], setpixel, hemisamples);
	});
	view_bitmap.synchronize();
	for (int y = 0; y < render_height; ++y) {
		memcpy((unsigned char*)bitmap_ptr + bitmap_stride * y, &result[render_width * y], sizeof(int) * render_width);
	}
}
*/