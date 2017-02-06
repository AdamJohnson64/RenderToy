extern "C" bool HaveCUDA()
{
#ifdef CUDA_INSTALLED
	return true;
#else
	return false;
#endif
}

#ifndef CUDA_INSTALLED
extern "C" void RaycastCUDAF32(const void* pScene, const void* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {}
extern "C" void RaycastCUDAF64(const void* pScene, const void* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {}
extern "C" void RaycastBitangentsCUDAF32(const void* pScene, const void* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {}
extern "C" void RaycastBitangentsCUDAF64(const void* pScene, const void* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {}
extern "C" void RaycastNormalsCUDAF32(const void* pScene, const void* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {}
extern "C" void RaycastNormalsCUDAF64(const void* pScene, const void* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {}
extern "C" void RaycastTangentsCUDAF32(const void* pScene, const void* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {}
extern "C" void RaycastTangentsCUDAF64(const void* pScene, const void* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {}
extern "C" void RaytraceCUDAF32(const void* pScene, const void* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {}
extern "C" void RaytraceCUDAF64(const void* pScene, const void* pInverseMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride) {}
#endif