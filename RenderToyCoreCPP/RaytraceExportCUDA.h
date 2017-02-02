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
extern "C" void RaycastCUDA(void *pScene, void *pMVP, void *bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
extern "C" void RaycastNormalsCUDA(void *pScene, void *pMVP, void *bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
extern "C" void RaycastTangentsCUDA(void *pScene, void *pMVP, void *bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
extern "C" void RaycastBitangentsCUDA(void *pScene, void *pMVP, void *bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
extern "C" void RaytraceCUDAF32(void *pScene, void *pMVP, void *bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
extern "C" void RaytraceCUDAF64(void *pScene, void *pMVP, void *bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);

#endif  // RAYTRACEEXPORTCUDA_H