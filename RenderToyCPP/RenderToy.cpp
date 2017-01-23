////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

#include <memory>

typedef void(RENDERFN)(void*, double*, void*, int, int, int);

extern "C" void CPURaycast(void *pScene, double *pMVP, void *bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
extern "C" void CPURaycastNormals(void *pScene, double *pMVP, void *bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
extern "C" void CPURaycastTangents(void *pScene, double *pMVP, void *bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
extern "C" void CPURaycastBitangents(void *pScene, double *pMVP, void *bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
extern "C" void CPURaytrace(void *pScene, double *pMVP, void *bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
extern "C" void CUDARaycast(void *pScene, double *pMVP, void *bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
extern "C" void CUDARaycastNormals(void *pScene, double *pMVP, void *bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
extern "C" void CUDARaycastTangents(void *pScene, double *pMVP, void *bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
extern "C" void CUDARaycastBitangents(void *pScene, double *pMVP, void *bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
extern "C" void CUDARaytrace(void *pScene, double *pMVP, void *bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);

namespace RenderToy
{
	public ref class RenderToyCPP
	{
	public:
		static bool HaveCUDA()
		{
			#ifdef CUDA_INSTALLED
			return true;
			#else
			return false;
			#endif
		}
		static void RaycastCPU(array<unsigned char>^ scene, array<double>^ inverse_mvp, System::IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
		{
			MarshalAndCall(CPURaycast, scene, inverse_mvp, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
		}
		static void RaycastNormalsCPU(array<unsigned char>^ scene, array<double>^ inverse_mvp, System::IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
		{
			MarshalAndCall(CPURaycastNormals, scene, inverse_mvp, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
		}
		static void RaycastTangentsCPU(array<unsigned char>^ scene, array<double>^ inverse_mvp, System::IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
		{
			MarshalAndCall(CPURaycastTangents, scene, inverse_mvp, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
		}
		static void RaycastBitangentsCPU(array<unsigned char>^ scene, array<double>^ inverse_mvp, System::IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
		{
			MarshalAndCall(CPURaycastBitangents, scene, inverse_mvp, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
		}
		static void RaytraceCPU(array<unsigned char>^ scene, array<double>^ inverse_mvp, System::IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
		{
			MarshalAndCall(CPURaytrace, scene, inverse_mvp, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
		}
		static void RaycastCUDA(array<unsigned char>^ scene, array<double>^ inverse_mvp, System::IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
		{
			#ifdef CUDA_INSTALLED
			MarshalAndCall(CUDARaycast, scene, inverse_mvp, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
			#endif
		}
		static void RaycastNormalsCUDA(array<unsigned char>^ scene, array<double>^ inverse_mvp, System::IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
		{
			#ifdef CUDA_INSTALLED
			MarshalAndCall(CUDARaycastNormals, scene, inverse_mvp, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
			#endif
		}
		static void RaycastTangentsCUDA(array<unsigned char>^ scene, array<double>^ inverse_mvp, System::IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
		{
			#ifdef CUDA_INSTALLED
			MarshalAndCall(CUDARaycastTangents, scene, inverse_mvp, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
			#endif
		}
		static void RaycastBitangentsCUDA(array<unsigned char>^ scene, array<double>^ inverse_mvp, System::IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
		{
			#ifdef CUDA_INSTALLED
			MarshalAndCall(CUDARaycastBitangents, scene, inverse_mvp, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
			#endif
		}
		static void RaytraceCUDA(array<unsigned char>^ scene, array<double>^ inverse_mvp, System::IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
		{
			#ifdef CUDA_INSTALLED
			MarshalAndCall(CUDARaytrace, scene, inverse_mvp, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
			#endif
		}
	private:
		static void MarshalAndCall(RENDERFN& fn, array<unsigned char>^ scene, array<double>^ inverse_mvp, System::IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
		{
			// Take a copy of the scene (and unpin to free up the GC as quickly as possible).
			int length = scene->Length;
			std::unique_ptr<unsigned char[]> c_scene(new unsigned char[length]);
			{
				pin_ptr<unsigned char> pin_scene = &scene[0];
				memcpy(c_scene.get(), pin_scene, length);
			}
			// Take a copy of the inverse MVP for the camera.
			double c_inverse_mvp[16] = {
				inverse_mvp[0], inverse_mvp[1], inverse_mvp[2], inverse_mvp[3],
				inverse_mvp[4], inverse_mvp[5], inverse_mvp[6], inverse_mvp[7],
				inverse_mvp[8], inverse_mvp[9], inverse_mvp[10], inverse_mvp[11],
				inverse_mvp[12], inverse_mvp[13], inverse_mvp[14], inverse_mvp[15] };
			fn(c_scene.get(), c_inverse_mvp, (void*)bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
		}
	};
}