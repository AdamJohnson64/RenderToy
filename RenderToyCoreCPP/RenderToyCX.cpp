////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

#include "RaytraceExportCPU.h"
#include "RaytraceExportAMP.h"

#define EXPORTGENERATOR(NAME) \
static void NAME(const Platform::Array<unsigned char>^ scene, const Platform::Array<unsigned char>^ inverse_mvp, Platform::WriteOnlyArray<unsigned char>^ bitmap_ptr, int render_width, int render_height, int bitmap_stride) \
{ \
	::NAME(scene->Data, inverse_mvp->Data, bitmap_ptr->Data, render_width, render_height, bitmap_stride); \
}

namespace RenderToy
{
	public ref class RenderToyCX sealed
	{
	public:
		EXPORTGENERATOR(RaycastCPUF32)
		EXPORTGENERATOR(RaycastCPUF64)
		EXPORTGENERATOR(RaycastNormalsCPUF32)
		EXPORTGENERATOR(RaycastTangentsCPUF32)
		EXPORTGENERATOR(RaycastBitangentsCPUF32)
		EXPORTGENERATOR(RaytraceCPUF32)
		EXPORTGENERATOR(RaycastAMPF32)
		EXPORTGENERATOR(RaycastNormalsAMPF32)
		EXPORTGENERATOR(RaycastTangentsAMPF32)
		EXPORTGENERATOR(RaycastBitangentsAMPF32)
		EXPORTGENERATOR(RaytraceAMPF32)
		static void AmbientOcclusionCPUF32(const Platform::Array<unsigned char>^ scene, const Platform::Array<unsigned char>^ inverse_mvp, Platform::WriteOnlyArray<unsigned char>^ bitmap_ptr, int render_width, int render_height, int bitmap_stride, int sample_offset, int sample_count)
		{
			::AmbientOcclusionCPUF32(scene->Data, inverse_mvp->Data, bitmap_ptr->Data, render_width, render_height, bitmap_stride, sample_offset, sample_count);
		}
		static void AmbientOcclusionAMPF32(const Platform::Array<unsigned char>^ scene, const Platform::Array<unsigned char>^ inverse_mvp, Platform::WriteOnlyArray<unsigned char>^ bitmap_ptr, int render_width, int render_height, int bitmap_stride, int sample_offset, int sample_count)
		{
			::AmbientOcclusionAMPF32(scene->Data, inverse_mvp->Data, bitmap_ptr->Data, render_width, render_height, bitmap_stride, sample_offset, sample_count);
		}
		EXPORTGENERATOR(RaycastNormalsCPUF64)
		EXPORTGENERATOR(RaycastTangentsCPUF64)
		EXPORTGENERATOR(RaycastBitangentsCPUF64)
		EXPORTGENERATOR(RaytraceCPUF64)
		static void AmbientOcclusionCPUF64(const Platform::Array<unsigned char>^ scene, const Platform::Array<unsigned char>^ inverse_mvp, Platform::WriteOnlyArray<unsigned char>^ bitmap_ptr, int render_width, int render_height, int bitmap_stride, int sample_offset, int sample_count)
		{
			::AmbientOcclusionCPUF64(scene->Data, inverse_mvp->Data, bitmap_ptr->Data, render_width, render_height, bitmap_stride, sample_offset, sample_count);
		}
	};
}