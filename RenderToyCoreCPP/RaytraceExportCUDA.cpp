extern "C" bool HaveCUDA()
{
#ifdef CUDA_INSTALLED
	return true;
#else
	return false;
#endif
}

#ifndef CUDA_INSTALLED
extern "C" void RaycastCUDAF32(const void* pScene, const void* pInverseMVP, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride) {}
extern "C" void RaycastCUDAF64(const void* pScene, const void* pInverseMVP, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride) {}
extern "C" void RaycastBitangentsCUDAF32(const void* pScene, const void* pInverseMVP, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride) {}
extern "C" void RaycastBitangentsCUDAF64(const void* pScene, const void* pInverseMVP, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride) {}
extern "C" void RaycastNormalsCUDAF32(const void* pScene, const void* pInverseMVP, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride) {}
extern "C" void RaycastNormalsCUDAF64(const void* pScene, const void* pInverseMVP, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride) {}
extern "C" void RaycastTangentsCUDAF32(const void* pScene, const void* pInverseMVP, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride) {}
extern "C" void RaycastTangentsCUDAF64(const void* pScene, const void* pInverseMVP, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride) {}
extern "C" void RaytraceCUDAF32(const void* pScene, const void* pInverseMVP, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride) {}
extern "C" void RaytraceCUDAF64(const void* pScene, const void* pInverseMVP, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride) {}
extern "C" void AmbientOcclusionCUDAF32(const void* pScene, const void* pMVP, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride, int hemisample_count, const void* hemisamples) {}
extern "C" void AmbientOcclusionCUDAF64(const void* pScene, const void* pMVP, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride, int hemisample_count, const void* hemisamples) {}
extern "C" void AmbientOcclusionMPCUDAF32(const void* pScene, const void* pMVP, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride, int hemisample_count, const void* hemisamples) {}
extern "C" void AmbientOcclusionFMPCUDAF32(const void* pScene, const void* pMVP, void* accumulator_ptr, int render_width, int render_height, int bitmap_stride, int hemisample_count, const void* hemisamples) {}
extern "C" void ToneMap(const void* accumulator_ptr, int accumulator_stride, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride, float rescale) {}
#endif