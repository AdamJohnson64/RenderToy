extern "C" void CPURaycast(void *pScene, void *pMVP, void *bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
extern "C" void CPUF64Raytrace(void *pScene, void *pMVP, void *bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);

namespace RenderToy
{
	public ref class RenderToyCX sealed
	{
	public:
		static void RaycastCPU(const Platform::Array<unsigned char>^ scene, const Platform::Array<unsigned char>^ inverse_mvp, Platform::WriteOnlyArray<unsigned char>^ bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
		{
			CPUF64Raytrace(scene->Data, inverse_mvp->Data, bitmap_ptr->Data, bitmap_width, bitmap_height, bitmap_stride);
		}
	};
}