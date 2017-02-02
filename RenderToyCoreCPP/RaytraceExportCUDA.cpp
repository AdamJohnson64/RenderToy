extern "C" bool HaveCUDA()
{
#ifdef CUDA_INSTALLED
	return true;
#else
	return false;
#endif
}

#ifndef CUDA_INSTALLED
extern "C" void CUDARaycast(void* pScene, double* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
{
}

extern "C" void CUDARaycastBitangents(void* pScene, double* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
{
}

extern "C" void CUDARaycastNormals(void* pScene, double* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
{
}

extern "C" void CUDARaycastTangents(void* pScene, double* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
{
}

extern "C" void CUDAF32Raytrace(void* pScene, float* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
{
}

extern "C" void CUDAF64Raytrace(void* pScene, double* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
{
}
#endif