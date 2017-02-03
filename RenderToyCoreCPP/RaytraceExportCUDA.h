////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// These are the core CUDA raytracer entrypoints.
////////////////////////////////////////////////////////////////////////////////

#pragma once
#ifndef RAYTRACEEXPORTCUDA_H
#define RAYTRACEEXPORTCUDA_H

extern "C" bool HaveCUDA();
extern "C" void RaycastCUDA(const void* pScene, const void* pMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
extern "C" void RaycastNormalsCUDA(const void* pScene, const void* pMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
extern "C" void RaycastTangentsCUDA(const void* pScene, const void* pMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
extern "C" void RaycastBitangentsCUDA(const void* pScene, const void* pMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
extern "C" void RaytraceCUDAF32(const void* pScene, const void* pMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
extern "C" void RaytraceCUDAF64(const void* pScene, const void* pMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);

#endif  // RAYTRACEEXPORTCUDA_H