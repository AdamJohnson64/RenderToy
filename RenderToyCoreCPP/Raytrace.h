////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// These are the shared structures and types consumed by both CPU and CUDA
// raytracers. This code is device-invariant.
////////////////////////////////////////////////////////////////////////////////

#pragma once
#ifndef RAYTRACE_H
#define RAYTRACE_H

template <typename FLOAT> struct Vector3 { FLOAT x, y, z; };
template <typename FLOAT> struct Vector4 { FLOAT x, y, z, w; };
template <typename FLOAT> struct Matrix44 { FLOAT M[16]; };

enum GeometryType {
	GEOMETRY_NONE = 0,
	GEOMETRY_PLANE = 1,
	GEOMETRY_SPHERE = 2,
	GEOMETRY_CUBE = 3,
	GEOMETRY_TRIANGLE = 4,
};

enum MaterialType {
	MATERIAL_NONE = 0,
	MATERIAL_COMMON,
	MATERIAL_CHECKERBOARD_XZ,
};

template <typename FLOAT>
struct MaterialCommon {
	Vector4<FLOAT> Ambient;
	Vector4<FLOAT> Diffuse;
	Vector4<FLOAT> Specular;
	Vector4<FLOAT> Reflect;
	Vector4<FLOAT> Refract;
	FLOAT Ior;
};

template <typename FLOAT>
struct SceneObject {
	Matrix44<FLOAT> Transform;
	Matrix44<FLOAT> TransformInverse;
	GeometryType Geometry;
	int GeometryOffset;
	MaterialType Material;
	int MaterialOffset;
};

#pragma warning(push)
#pragma warning(disable:4200)
template <typename FLOAT>
struct Scene {
	int FileSize;
	int ObjectCount;
	int Reserved0;
	int Reserved1;
	SceneObject<FLOAT> Objects[];
};
#pragma warning(pop)

// Intersection Query Types.
// These structures determine the data to be queried when performing an
// intersection test against the scene. It is prudent to request minimal data
// as this will reduce the burden on the intersection engine.

// IntersectObject - retrieve only the intersected object and distance.
template <typename FLOAT>
struct IntersectObject {
	FLOAT Lambda;
	SceneObject<FLOAT> Object;
};

// IntersectSimple - retrieve only the intersection distance.
template <typename FLOAT>
struct IntersectSimple {
	FLOAT Lambda;
};

// IntersectNormal - retrieve the intersection distance and normal.
template <typename FLOAT>
struct IntersectNormal {
	FLOAT Lambda;
	Vector3<FLOAT> Normal;
};

// IntersectTBN - retrieve the intersection distance, normal, tangent and
// bitangent.
template <typename FLOAT>
struct IntersectTBN {
	FLOAT Lambda;
	Vector3<FLOAT> Normal;
	Vector3<FLOAT> Tangent;
	Vector3<FLOAT> Bitangent;
};

#endif