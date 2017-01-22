#include <memory>

extern "C" bool cudaRaycast(void *pScene, double *pMVP, void *bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
extern "C" bool cudaRaycastNormals(void *pScene, double *pMVP, void *bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
extern "C" bool cudaRaytrace(void *pScene, double *pMVP, void *bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);

namespace RenderToy
{
	public ref class CUDASupport
	{
	public:
		static bool Available()
		{
			#ifdef CUDA_INSTALLED
			return true;
			#else
			return false;
			#endif
		}
	};
	public ref class RaycastCUDA
	{
	public:
		static void Fill(array<unsigned char>^ scene, array<double>^ inverse_mvp, System::IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
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
			cudaRaycast(c_scene.get(), c_inverse_mvp, (void*)bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
#endif
		}
	};
	public ref class RaycastNormalsCUDA
	{
	public:
		static void Fill(array<unsigned char>^ scene, array<double>^ inverse_mvp, System::IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
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
			cudaRaycastNormals(c_scene.get(), c_inverse_mvp, (void*)bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
#endif
		}
	};
	public ref class RaytraceCUDA
	{
	public:
		static void Fill(array<unsigned char>^ scene, array<double>^ inverse_mvp, System::IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
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
			cudaRaytrace(c_scene.get(), c_inverse_mvp, (void*)bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
			#endif
		}
	};
}