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
static void NAME(array<unsigned char>^ scene, array<unsigned char>^ inverse_mvp, System::IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) \
{ \
	pin_ptr<unsigned char> pin_scene = &scene[0]; \
	pin_ptr<unsigned char> pin_inverse_mvp = &inverse_mvp[0]; \
	::NAME(pin_scene, pin_inverse_mvp, (void*)bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride); \
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
		EXPORTGENERATOR(RaycastNormalsCPUF64)
		EXPORTGENERATOR(RaycastBitangentsCPUF32)
		EXPORTGENERATOR(RaycastBitangentsCPUF64)
		EXPORTGENERATOR(RaycastTangentsCPUF32)
		EXPORTGENERATOR(RaycastTangentsCPUF64)
		EXPORTGENERATOR(RaytraceCPUF32)
		EXPORTGENERATOR(RaytraceCPUF64)
		EXPORTGENERATOR(RaycastCUDAF32)
		EXPORTGENERATOR(RaycastCUDAF64)
		EXPORTGENERATOR(RaycastNormalsCUDAF32)
		EXPORTGENERATOR(RaycastNormalsCUDAF64)
		EXPORTGENERATOR(RaycastTangentsCUDAF32)
		EXPORTGENERATOR(RaycastTangentsCUDAF64)
		EXPORTGENERATOR(RaycastBitangentsCUDAF32)
		EXPORTGENERATOR(RaycastBitangentsCUDAF64)
		EXPORTGENERATOR(RaytraceCUDAF32)
		EXPORTGENERATOR(RaytraceCUDAF64)
		static void RaytraceCPUF64AA(array<unsigned char>^ scene, array<unsigned char>^ inverse_mvp, System::IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride, int superx, int supery)
		{
			pin_ptr<unsigned char> pin_scene = &scene[0];
			pin_ptr<unsigned char> pin_inverse_mvp = &inverse_mvp[0];
			::RaytraceCPUF64AA(pin_scene, pin_inverse_mvp, (void*)bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, superx, supery);
		}
		static void AmbientOcclusionCPUF32(array<unsigned char>^ scene, array<unsigned char>^ inverse_mvp, System::IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride, int hemisample_count, array<unsigned char>^ hemisamples)
		{
			pin_ptr<unsigned char> pin_scene = &scene[0];
			pin_ptr<unsigned char> pin_inverse_mvp = &inverse_mvp[0];
			pin_ptr<unsigned char> pin_hemisamples = &hemisamples[0];
			::AmbientOcclusionCPUF32(pin_scene, pin_inverse_mvp, (void*)bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, hemisample_count, pin_hemisamples);
		}
		static void AmbientOcclusionCPUF64(array<unsigned char>^ scene, array<unsigned char>^ inverse_mvp, System::IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride, int hemisample_count, array<unsigned char>^ hemisamples)
		{
			pin_ptr<unsigned char> pin_scene = &scene[0];
			pin_ptr<unsigned char> pin_inverse_mvp = &inverse_mvp[0];
			pin_ptr<unsigned char> pin_hemisamples = &hemisamples[0];
			::AmbientOcclusionCPUF64(pin_scene, pin_inverse_mvp, (void*)bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, hemisample_count, pin_hemisamples);
		}
		static void AmbientOcclusionCUDAF32(array<unsigned char>^ scene, array<unsigned char>^ inverse_mvp, System::IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride, int hemisample_count, array<unsigned char>^ hemisamples)
		{
			pin_ptr<unsigned char> pin_scene = &scene[0];
			pin_ptr<unsigned char> pin_inverse_mvp = &inverse_mvp[0];
			pin_ptr<unsigned char> pin_hemisamples = &hemisamples[0];
			::AmbientOcclusionCUDAF32(pin_scene, pin_inverse_mvp, (void*)bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, hemisample_count, pin_hemisamples);
		}
		static void AmbientOcclusionCUDAF64(array<unsigned char>^ scene, array<unsigned char>^ inverse_mvp, System::IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride, int hemisample_count, array<unsigned char>^ hemisamples)
		{
			pin_ptr<unsigned char> pin_scene = &scene[0];
			pin_ptr<unsigned char> pin_inverse_mvp = &inverse_mvp[0];
			pin_ptr<unsigned char> pin_hemisamples = &hemisamples[0];
			::AmbientOcclusionCUDAF64(pin_scene, pin_inverse_mvp, (void*)bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, hemisample_count, pin_hemisamples);
		}
		static void AmbientOcclusionMPCUDAF32(array<unsigned char>^ scene, array<unsigned char>^ inverse_mvp, System::IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride, int hemisample_count, array<unsigned char>^ hemisamples)
		{
			pin_ptr<unsigned char> pin_scene = &scene[0];
			pin_ptr<unsigned char> pin_inverse_mvp = &inverse_mvp[0];
			pin_ptr<unsigned char> pin_hemisamples = &hemisamples[0];
			::AmbientOcclusionMPCUDAF32(pin_scene, pin_inverse_mvp, (void*)bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, hemisample_count, pin_hemisamples);
		}
		static void AmbientOcclusionFMPCUDAF32(array<unsigned char>^ scene, array<unsigned char>^ inverse_mvp, System::IntPtr acc_ptr, int bitmap_width, int bitmap_height, int bitmap_stride, int hemisample_count, array<unsigned char>^ hemisamples)
		{
			pin_ptr<unsigned char> pin_scene = &scene[0];
			pin_ptr<unsigned char> pin_inverse_mvp = &inverse_mvp[0];
			pin_ptr<unsigned char> pin_hemisamples = &hemisamples[0];
			::AmbientOcclusionFMPCUDAF32(pin_scene, pin_inverse_mvp, (void*)acc_ptr, bitmap_width, bitmap_height, bitmap_stride, hemisample_count, pin_hemisamples);
		}
		static void ToneMap(System::IntPtr acc_ptr, int acc_stride, System::IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride, float rescale)
		{
			::ToneMap((const void*)acc_ptr, acc_stride, (void*)bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride, rescale);
		}
	};
}