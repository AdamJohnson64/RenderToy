////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// These are the core AMP raytracer entrypoints.
////////////////////////////////////////////////////////////////////////////////

#pragma once
#ifndef RAYTRACEEXPORTAMP_H
#define RAYTRACEEXPORTAMP_H

#ifndef RENDERTOY_NO_AMP
extern "C" void RaycastAMPF32(const void* pScene, const void* pMVP, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride);
extern "C" void RaycastNormalsAMPF32(const void* pScene, const void* pMVP, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride);
extern "C" void RaycastTangentsAMPF32(const void* pScene, const void* pMVP, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride);
extern "C" void RaycastBitangentsAMPF32(const void* pScene, const void* pMVP, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride);
extern "C" void RaytraceAMPF32(const void* pScene, const void* pMVP, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride);
extern "C" void DebugMeshAMPF32(const void* pScene, const void* pMVP, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride);
//extern "C" void AmbientOcclusionAMPF32(const void* pScene, const void* pMVP, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride, int sample_offset, int sample_count);
#endif

extern "C" void TEST_RaycastNormalsAMPF32D3D(const void* scene, const void* inverse_mvp, void* d3ddevice, void *d3dtexture);

#endif  // RAYTRACEEXPORTAMP_H