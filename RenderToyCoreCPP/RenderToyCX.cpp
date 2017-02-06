////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

#include "RaytraceExportCPU.h"
#include "RaytraceExportAMP.h"

#define EXPORTGENERATOR(NAME) \
static void NAME(const Platform::Array<unsigned char>^ scene, const Platform::Array<unsigned char>^ inverse_mvp, Platform::WriteOnlyArray<unsigned char>^ bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) \
{ \
	::NAME(scene->Data, inverse_mvp->Data, bitmap_ptr->Data, bitmap_width, bitmap_height, bitmap_stride); \
}

namespace RenderToy
{
	public ref class RenderToyCX sealed
	{
	public:
		EXPORTGENERATOR(RaycastCPUF32)
		EXPORTGENERATOR(RaycastCPUF64)
		EXPORTGENERATOR(RaycastNormalsCPUF32)
		EXPORTGENERATOR(RaycastNormalsCPUF64)
		EXPORTGENERATOR(RaycastTangentsCPUF32)
		EXPORTGENERATOR(RaycastTangentsCPUF64)
		EXPORTGENERATOR(RaycastBitangentsCPUF32)
		EXPORTGENERATOR(RaycastBitangentsCPUF64)
		EXPORTGENERATOR(RaytraceCPUF32)
		EXPORTGENERATOR(RaytraceCPUF64)
		EXPORTGENERATOR(RaycastAMPF32)
		EXPORTGENERATOR(RaycastNormalsAMPF32)
		EXPORTGENERATOR(RaycastTangentsAMPF32)
		EXPORTGENERATOR(RaycastBitangentsAMPF32)
		EXPORTGENERATOR(RaytraceAMPF32)
		static void AmbientOcclusionAMPF32(const Platform::Array<unsigned char>^ scene, const Platform::Array<unsigned char>^ inverse_mvp, Platform::WriteOnlyArray<unsigned char>^ bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride, int hemisample_count, const Platform::Array<unsigned char>^ hemisamples)
		{
			::AmbientOcclusionAMPF32(scene->Data, inverse_mvp->Data, bitmap_ptr->Data, bitmap_width, bitmap_height, bitmap_stride, hemisample_count, hemisamples->Data); \
		}
	};
}