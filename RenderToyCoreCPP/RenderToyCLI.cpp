////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// These functions present the static interface of the raytracer to the Common
// Language Runtime for consumption by .NET languages.
////////////////////////////////////////////////////////////////////////////////

#include <memory>
#include "RaytraceExportCPU.h"
#include "RaytraceExportCUDA.h"

#define EXPORTGENERATOR(NAME) \
static void NAME(array<unsigned char>^ scene, array<unsigned char>^ inverse_mvp, System::IntPtr bitmap_ptr, int render_width, int render_height, int bitmap_stride) \
{ \
	pin_ptr<unsigned char> pin_scene = &scene[0]; \
	pin_ptr<unsigned char> pin_inverse_mvp = &inverse_mvp[0]; \
	::NAME(pin_scene, pin_inverse_mvp, (void*)bitmap_ptr, render_width, render_height, bitmap_stride); \
}

namespace RenderToy
{
	public ref class RenderToyCLI
	{
	public:
		static bool HaveCUDA() { return ::HaveCUDA(); }
		EXPORTGENERATOR(RaycastCPUF32)
		EXPORTGENERATOR(RaycastCPUF64)
		EXPORTGENERATOR(RaycastNormalsCPUF32)
		EXPORTGENERATOR(RaycastBitangentsCPUF32)
		EXPORTGENERATOR(RaycastTangentsCPUF32)
		EXPORTGENERATOR(RaytraceCPUF32)
		EXPORTGENERATOR(RaycastCUDAF32)
		EXPORTGENERATOR(RaycastCUDAF64)
		EXPORTGENERATOR(RaycastNormalsCUDAF32)
		EXPORTGENERATOR(RaycastTangentsCUDAF32)
		EXPORTGENERATOR(RaycastBitangentsCUDAF32)
		EXPORTGENERATOR(RaytraceCUDAF32)
		static void RaytraceCPUF32AA(array<unsigned char>^ scene, array<unsigned char>^ inverse_mvp, System::IntPtr bitmap_ptr, int render_width, int render_height, int bitmap_stride, int superx, int supery)
		{
			pin_ptr<unsigned char> pin_scene = &scene[0];
			pin_ptr<unsigned char> pin_inverse_mvp = &inverse_mvp[0];
			::RaytraceCPUF32AA(pin_scene, pin_inverse_mvp, (void*)bitmap_ptr, render_width, render_height, bitmap_stride, superx, supery);
		}
		static void AmbientOcclusionCPUF32(array<unsigned char>^ scene, array<unsigned char>^ inverse_mvp, System::IntPtr bitmap_ptr, int render_width, int render_height, int bitmap_stride, int sample_offset, int sample_count)
		{
			pin_ptr<unsigned char> pin_scene = &scene[0];
			pin_ptr<unsigned char> pin_inverse_mvp = &inverse_mvp[0];
			::AmbientOcclusionCPUF32(pin_scene, pin_inverse_mvp, (void*)bitmap_ptr, render_width, render_height, bitmap_stride, sample_offset, sample_count);
		}
		static void AmbientOcclusionCUDAF32(array<unsigned char>^ scene, array<unsigned char>^ inverse_mvp, System::IntPtr bitmap_ptr, int render_width, int render_height, int bitmap_stride, int sample_offset, int sample_count)
		{
			pin_ptr<unsigned char> pin_scene = &scene[0];
			pin_ptr<unsigned char> pin_inverse_mvp = &inverse_mvp[0];
			::AmbientOcclusionCUDAF32(pin_scene, pin_inverse_mvp, (void*)bitmap_ptr, render_width, render_height, bitmap_stride, sample_offset, sample_count);
		}
		static void AmbientOcclusionFMPCUDAF32(array<unsigned char>^ scene, array<unsigned char>^ inverse_mvp, System::IntPtr accumulator_ptr, int render_width, int render_height, int bitmap_stride, int sample_offset, int sample_count)
		{
			pin_ptr<unsigned char> pin_scene = &scene[0];
			pin_ptr<unsigned char> pin_inverse_mvp = &inverse_mvp[0];
			::AmbientOcclusionFMPCUDAF32(pin_scene, pin_inverse_mvp, (void*)accumulator_ptr, render_width, render_height, bitmap_stride, sample_offset, sample_count);
		}
		static void ToneMap(System::IntPtr accumulator_ptr, int accumulator_stride, System::IntPtr bitmap_ptr, int render_width, int render_height, int bitmap_stride, float rescale)
		{
			::ToneMap((const void*)accumulator_ptr, accumulator_stride, (void*)bitmap_ptr, render_width, render_height, bitmap_stride, rescale);
		}
		#ifdef USE_F64
		EXPORTGENERATOR(RaycastNormalsCPUF64)
		EXPORTGENERATOR(RaycastBitangentsCPUF64)
		EXPORTGENERATOR(RaycastTangentsCPUF64)
		EXPORTGENERATOR(RaytraceCPUF64)
		EXPORTGENERATOR(RaycastNormalsCUDAF64)
		EXPORTGENERATOR(RaycastTangentsCUDAF64)
		EXPORTGENERATOR(RaycastBitangentsCUDAF64)
		EXPORTGENERATOR(RaytraceCUDAF64)
		static void AmbientOcclusionCPUF64(array<unsigned char>^ scene, array<unsigned char>^ inverse_mvp, System::IntPtr bitmap_ptr, int render_width, int render_height, int bitmap_stride, int sample_offset, int sample_count)
		{
			pin_ptr<unsigned char> pin_scene = &scene[0];
			pin_ptr<unsigned char> pin_inverse_mvp = &inverse_mvp[0];
			::AmbientOcclusionCPUF64(pin_scene, pin_inverse_mvp, (void*)bitmap_ptr, render_width, render_height, bitmap_stride, sample_offset, sample_count);
		}
		static void AmbientOcclusionCUDAF64(array<unsigned char>^ scene, array<unsigned char>^ inverse_mvp, System::IntPtr bitmap_ptr, int render_width, int render_height, int bitmap_stride, int sample_offset, int sample_count)
		{
			pin_ptr<unsigned char> pin_scene = &scene[0];
			pin_ptr<unsigned char> pin_inverse_mvp = &inverse_mvp[0];
			::AmbientOcclusionCUDAF64(pin_scene, pin_inverse_mvp, (void*)bitmap_ptr, render_width, render_height, bitmap_stride, sample_offset, sample_count);
		}
		#endif  // USE_F64
	};
}