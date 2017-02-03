////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// These are the core AMP raytracer entrypoints.
////////////////////////////////////////////////////////////////////////////////

#pragma once
#ifndef RAYTRACEEXPORTAMP_H
#define RAYTRACEEXPORTAMP_H

extern "C" void RaycastAMPF32(const void* pScene, const void* pMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
extern "C" void RaycastNormalsAMPF32(const void* pScene, const void* pMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
extern "C" void RaycastTangentsAMPF32(const void* pScene, const void* pMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
extern "C" void RaycastBitangentsAMPF32(const void* pScene, const void* pMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
extern "C" void RaytraceAMPF32(const void* pScene, const void* pMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
extern "C" void RaytraceAMPF64(const void* pScene, const void* pMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);

#endif  // RAYTRACEEXPORTAMP_H