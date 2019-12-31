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
	GEOMETRY_PLANE = 0x6e616c50,		// FOURCC "Plan"
	GEOMETRY_SPHERE = 0x72687053,		// FOURCC "Sphr"
	GEOMETRY_CUBE = 0x65627543,			// FOURCC "Cube"
	GEOMETRY_MESHBVH = 0x4268734d,      // FOURCC "MshB"
};

enum MaterialType {
	MATERIAL_NONE = 0,
	MATERIAL_COMMON = 0x6c74614d,           // FOURCC "Matl"
	MATERIAL_CHECKERBOARD_XZ = 0x5a586843,  // FOURCC "ChXZ"
};

#pragma warning(push)
#pragma warning(disable:4200)
template <typename FLOAT>
struct TriangleList {
	int TriangleCount;
	int Padding0; // Forces padding to 8 bytes for double mode.
	Vector3<FLOAT> BoundMin;
	Vector3<FLOAT> BoundMax;
	Vector3<FLOAT> Vertices[];
};
#pragma warning(pop)

template <typename FLOAT>
struct MeshBVH {
	Vector3<FLOAT> Min;
	Vector3<FLOAT> Max;
	int TriangleOffset;
	int ChildOffset;
	int NextOffset;
	int Padding0;
};

template <typename FLOAT>
struct Triangle3D {
	Vector3<FLOAT> P[3];
};

#pragma warning(push)
#pragma warning(disable:4200)
template <typename FLOAT>
struct Triangle3DList {
	int Count;
	int Padding0;
	Triangle3D<FLOAT> Triangles[];
};
#pragma warning(pop)

#pragma warning(push)
#pragma warning(disable:4200)
template <typename FLOAT>
struct Vector3List
{
	int Count;
	int Padding0;
	Vector3<FLOAT> Data[];
};
#pragma warning(pop)

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

// IntersectDebugMesh - accumulate box and tri test counts.
template <typename FLOAT>
struct IntersectDebugMesh {
	FLOAT Lambda;
	int CountBoxTest;
	int CountTriangleTest;
	int CountTriangleHit;
};

#endif