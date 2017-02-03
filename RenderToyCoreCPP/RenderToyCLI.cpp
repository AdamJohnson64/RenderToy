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
	};
}