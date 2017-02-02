extern "C" bool HaveCUDA()
{
#ifdef CUDA_INSTALLED
	return true;
#else
	return false;
#endif
}

#ifndef CUDA_INSTALLED
extern "C" void RaycastCUDA(void* pScene, double* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
{
}

extern "C" void RaycastBitangentsCUDA(void* pScene, double* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
{
}

extern "C" void RaycastNormalsCUDA(void* pScene, double* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
{
}

extern "C" void RaycastTangentsCUDA(void* pScene, double* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
{
}

extern "C" void RaytraceCUDAF32(void* pScene, float* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
{
}

extern "C" void RaytraceCUDAF64(void* pScene, double* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
{
}
#endif