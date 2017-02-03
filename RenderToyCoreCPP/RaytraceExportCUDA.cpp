extern "C" bool HaveCUDA()
{
#ifdef CUDA_INSTALLED
	return true;
#else
	return false;
#endif
}

#ifndef CUDA_INSTALLED
extern "C" void RaycastCUDA(const void* pScene, const void* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
{
}

extern "C" void RaycastBitangentsCUDA(const void* pScene, const void* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
{
}

extern "C" void RaycastNormalsCUDA(const void* pScene, const void* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
{
}

extern "C" void RaycastTangentsCUDA(const void* pScene, const void* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
{
}

extern "C" void RaytraceCUDAF32(const void* pScene, const void* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
{
}

extern "C" void RaytraceCUDAF64(const void* pScene, const void* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
{
}
#endif