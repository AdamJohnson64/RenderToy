#include <memory>

extern "C" void CPURaycast(void *pScene, double *pMVP, void *bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
extern "C" void CPURaycastNormals(void *pScene, double *pMVP, void *bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
extern "C" void CPURaytrace(void *pScene, double *pMVP, void *bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
extern "C" void CUDARaycast(void *pScene, double *pMVP, void *bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
extern "C" void CUDARaycastNormals(void *pScene, double *pMVP, void *bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
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
			// Render the scene (if possible).
			CPURaycast(c_scene.get(), c_inverse_mvp, (void*)bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
		}
		static void RaycastNormalsCPU(array<unsigned char>^ scene, array<double>^ inverse_mvp, System::IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
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
			// Render the scene (if possible).
			CPURaycastNormals(c_scene.get(), c_inverse_mvp, (void*)bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
		}
		static void RaytraceCPU(array<unsigned char>^ scene, array<double>^ inverse_mvp, System::IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
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
			// Render the scene (if possible).
			CPURaytrace(c_scene.get(), c_inverse_mvp, (void*)bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
		}
		static void RaycastCUDA(array<unsigned char>^ scene, array<double>^ inverse_mvp, System::IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
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
			// Render the scene (if possible).
#ifdef CUDA_INSTALLED
			CUDARaycast(c_scene.get(), c_inverse_mvp, (void*)bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
#endif
		}
		static void RaycastNormalsCUDA(array<unsigned char>^ scene, array<double>^ inverse_mvp, System::IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
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
			// Render the scene (if possible).
#ifdef CUDA_INSTALLED
			CUDARaycastNormals(c_scene.get(), c_inverse_mvp, (void*)bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
#endif
		}
		static void RaytraceCUDA(array<unsigned char>^ scene, array<double>^ inverse_mvp, System::IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
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
				inverse_mvp[ 0], inverse_mvp[ 1], inverse_mvp[ 2], inverse_mvp[ 3],
				inverse_mvp[ 4], inverse_mvp[ 5], inverse_mvp[ 6], inverse_mvp[ 7],
				inverse_mvp[ 8], inverse_mvp[ 9], inverse_mvp[10], inverse_mvp[11],
				inverse_mvp[12], inverse_mvp[13], inverse_mvp[14], inverse_mvp[15]};
			// Render the scene (if possible).
			#ifdef CUDA_INSTALLED
			CUDARaytrace(c_scene.get(), c_inverse_mvp, (void*)bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
			#endif
		}
	};
}