extern "C" bool cudaRaytrace(double *pMVP, void *bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);

namespace RenderToy
{
	public ref class RaytraceCUDA
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
		static void Fill(array<double>^ MVP, System::IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
		{
			double mvp[16] = {
				MVP[ 0], MVP[ 1], MVP[ 2], MVP[ 3],
				MVP[ 4], MVP[ 5], MVP[ 6], MVP[ 7],
				MVP[ 8], MVP[ 9], MVP[10], MVP[11],
				MVP[12], MVP[13], MVP[14], MVP[15]};
			#ifdef CUDA_INSTALLED
			cudaRaytrace(mvp, (void*)bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
			#endif
		}
	};
}