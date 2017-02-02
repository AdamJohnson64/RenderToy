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

typedef void(RENDERFN)(void*, void*, void*, int, int, int);

namespace RenderToy
{
	public ref class RenderToyCPP
	{
	public:
		static bool HaveCUDA()
		{
			return ::HaveCUDA();
		}
		static void RaycastCPU(array<unsigned char>^ scene, array<unsigned char>^ inverse_mvp, System::IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
		{
			MarshalAndCall(CPURaycast, scene, inverse_mvp, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
		}
		static void RaycastNormalsCPU(array<unsigned char>^ scene, array<unsigned char>^ inverse_mvp, System::IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
		{
			MarshalAndCall(CPURaycastNormals, scene, inverse_mvp, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
		}
		static void RaycastTangentsCPU(array<unsigned char>^ scene, array<unsigned char>^ inverse_mvp, System::IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
		{
			MarshalAndCall(CPURaycastTangents, scene, inverse_mvp, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
		}
		static void RaycastBitangentsCPU(array<unsigned char>^ scene, array<unsigned char>^ inverse_mvp, System::IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
		{
			MarshalAndCall(CPURaycastBitangents, scene, inverse_mvp, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
		}
		static void RaytraceCPUF32(array<unsigned char>^ scene, array<unsigned char>^ inverse_mvp, System::IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
		{
			MarshalAndCall(CPUF32Raytrace, scene, inverse_mvp, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
		}
		static void RaytraceCPUF64(array<unsigned char>^ scene, array<unsigned char>^ inverse_mvp, System::IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
		{
			MarshalAndCall(CPUF64Raytrace, scene, inverse_mvp, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
		}
		static void RaycastCUDA(array<unsigned char>^ scene, array<unsigned char>^ inverse_mvp, System::IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
		{
			MarshalAndCall(CUDARaycast, scene, inverse_mvp, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
		}
		static void RaycastNormalsCUDA(array<unsigned char>^ scene, array<unsigned char>^ inverse_mvp, System::IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
		{
			MarshalAndCall(CUDARaycastNormals, scene, inverse_mvp, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
		}
		static void RaycastTangentsCUDA(array<unsigned char>^ scene, array<unsigned char>^ inverse_mvp, System::IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
		{
			MarshalAndCall(CUDARaycastTangents, scene, inverse_mvp, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
		}
		static void RaycastBitangentsCUDA(array<unsigned char>^ scene, array<unsigned char>^ inverse_mvp, System::IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
		{
			MarshalAndCall(CUDARaycastBitangents, scene, inverse_mvp, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
		}
		static void RaytraceCUDAF32(array<unsigned char>^ scene, array<unsigned char>^ inverse_mvp, System::IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
		{
			MarshalAndCall(CUDAF32Raytrace, scene, inverse_mvp, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
		}
		static void RaytraceCUDAF64(array<unsigned char>^ scene, array<unsigned char>^ inverse_mvp, System::IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
		{
			MarshalAndCall(CUDAF64Raytrace, scene, inverse_mvp, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
		}
	private:
		static void MarshalAndCall(RENDERFN& fn, array<unsigned char>^ scene, array<unsigned char>^ inverse_mvp, System::IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
		{
			pin_ptr<unsigned char> pin_scene = &scene[0];
			pin_ptr<unsigned char> pin_inverse_mvp = &inverse_mvp[0];
			fn(pin_scene, pin_inverse_mvp, (void*)bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
		}
	};
}