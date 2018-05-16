////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// This file contains the common parts of the raytracer shared by CPU, AMP and
// CUDA implementations.
//
// All parts of this module are templated over their floating point type to
// support both IEEE 754 32-bit float and 64-bit double implementation.
////////////////////////////////////////////////////////////////////////////////

#pragma region - Section : Type Safe Shims & Traits. -
template <typename FLOAT>
struct FloatTraits {
};

template <>
struct FloatTraits<float> {
	constexpr static float Max = FLT_MAX;
	constexpr static float PI = 3.1415926535897932384626433832795028841971693993751058209749445923078164062;
};

template <>
struct FloatTraits<double> {
	constexpr static double Max = DBL_MAX;
	constexpr static double PI = 3.1415926535897932384626433832795028841971693993751058209749445923078164062;
};

DEVICE_PREFIX float Acos(float v) DEVICE_SUFFIX { return acosf(v); }
DEVICE_PREFIX double Acos(double v) DEVICE_SUFFIX { return acos(v); }
DEVICE_PREFIX float Cos(float v) DEVICE_SUFFIX { return cosf(v); }
DEVICE_PREFIX double Cos(double v) DEVICE_SUFFIX { return cos(v); }
DEVICE_PREFIX float Power(float v, float p) DEVICE_SUFFIX { return powf(v, p); }
DEVICE_PREFIX double Power(double v, double p) DEVICE_SUFFIX { return pow(v, p); }
DEVICE_PREFIX float Sin(float v) DEVICE_SUFFIX { return sinf(v); }
DEVICE_PREFIX double Sin(double v) DEVICE_SUFFIX { return sin(v); }
DEVICE_PREFIX float SquareRoot(float v) DEVICE_SUFFIX { return sqrtf(v); }
DEVICE_PREFIX double SquareRoot(double v) DEVICE_SUFFIX { return sqrt(v); }

template <typename FLOAT>
DEVICE_PREFIX const int* FollowPointer(const Scene<FLOAT>& pScene, int offset) DEVICE_SUFFIX {
	return (const int*)&pScene + offset / 4;
}
#pragma endregion

#pragma region - Section : Basic Vector Math. -
template <typename FLOAT> DEVICE_PREFIX Vector3<FLOAT> make_vector3(FLOAT x, FLOAT y, FLOAT z) DEVICE_SUFFIX { Vector3<FLOAT> o; o.x = x; o.y = y; o.z = z; return o; }
template <typename FLOAT> DEVICE_PREFIX Vector4<FLOAT> make_vector4(FLOAT x, FLOAT y, FLOAT z, FLOAT w) DEVICE_SUFFIX { Vector4<FLOAT> o; o.x = x; o.y = y; o.z = z; o.w = w; return o; }

template <typename FLOAT> DEVICE_PREFIX Vector3<FLOAT> operator-(const Vector3<FLOAT>& val) DEVICE_SUFFIX { return make_vector3<FLOAT>(-val.x, -val.y, -val.z); }
template <typename FLOAT> DEVICE_PREFIX Vector4<FLOAT> operator-(const Vector4<FLOAT>& val) DEVICE_SUFFIX { return make_vector4<FLOAT>(-val.z, -val.y, -val.z, -val.w); }
template <typename FLOAT> DEVICE_PREFIX Vector3<FLOAT> operator+(const Vector3<FLOAT>& lhs, const Vector3<FLOAT>& rhs) DEVICE_SUFFIX { return make_vector3<FLOAT>(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z); }
template <typename FLOAT> DEVICE_PREFIX Vector4<FLOAT> operator+(const Vector4<FLOAT>& lhs, const Vector4<FLOAT>& rhs) DEVICE_SUFFIX { return make_vector4<FLOAT>(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w); }
template <typename FLOAT> DEVICE_PREFIX Vector3<FLOAT> operator-(const Vector3<FLOAT>& lhs, const Vector3<FLOAT>& rhs) DEVICE_SUFFIX { return make_vector3<FLOAT>(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z); }
template <typename FLOAT> DEVICE_PREFIX Vector4<FLOAT> operator-(const Vector4<FLOAT>& lhs, const Vector4<FLOAT>& rhs) DEVICE_SUFFIX { return make_vector4<FLOAT>(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w); }
template <typename FLOAT> DEVICE_PREFIX Vector3<FLOAT> operator*(const Vector3<FLOAT>& lhs, FLOAT rhs) DEVICE_SUFFIX { return make_vector3<FLOAT>(lhs.x * rhs, lhs.y * rhs, lhs.z * rhs); }
template <typename FLOAT> DEVICE_PREFIX Vector3<FLOAT> operator*(FLOAT lhs, const Vector3<FLOAT>& rhs) DEVICE_SUFFIX { return rhs * lhs; }
template <typename FLOAT> DEVICE_PREFIX Vector4<FLOAT> operator*(const Vector4<FLOAT>& lhs, FLOAT rhs) DEVICE_SUFFIX { return make_vector4<FLOAT>(lhs.x * rhs, lhs.y * rhs, lhs.z * rhs, lhs.w * rhs); }
template <typename FLOAT> DEVICE_PREFIX Vector4<FLOAT> operator*(FLOAT lhs, const Vector4<FLOAT>& rhs) DEVICE_SUFFIX { return rhs * lhs; }
template <typename FLOAT> DEVICE_PREFIX Vector3<FLOAT> operator/(const Vector3<FLOAT>& lhs, FLOAT rhs) DEVICE_SUFFIX { return lhs * (1 / rhs); }
template <typename FLOAT> DEVICE_PREFIX Vector4<FLOAT> operator/(const Vector4<FLOAT>& lhs, FLOAT rhs) DEVICE_SUFFIX { return lhs * (1 / rhs); }

template <typename FLOAT> DEVICE_PREFIX FLOAT Clamp(FLOAT min, FLOAT max, FLOAT val) DEVICE_SUFFIX { return val < min ? min : (val > max ? max : val); }
template <typename FLOAT> DEVICE_PREFIX Vector3<FLOAT> Cross(const Vector3<FLOAT>& lhs, const Vector3<FLOAT> rhs) DEVICE_SUFFIX { return make_vector3<FLOAT>(lhs.y * rhs.z - lhs.z * rhs.y, lhs.z * rhs.x - lhs.x * rhs.z, lhs.x * rhs.y - lhs.y * rhs.x); }
template <typename FLOAT> DEVICE_PREFIX FLOAT Dot(const Vector3<FLOAT>& lhs, const Vector3<FLOAT>& rhs) DEVICE_SUFFIX { return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z; }
template <typename FLOAT> DEVICE_PREFIX FLOAT Length(FLOAT x, FLOAT y, FLOAT z) DEVICE_SUFFIX { return SquareRoot(x * x + y * y + z * z); }
template <typename FLOAT> DEVICE_PREFIX FLOAT Lerp(FLOAT y1, FLOAT y2, FLOAT x) DEVICE_SUFFIX { return y1 + (y2 - y1) * x; }
template <typename FLOAT> DEVICE_PREFIX FLOAT Max(FLOAT lhs, FLOAT rhs) DEVICE_SUFFIX { return lhs > rhs ? lhs : rhs; }
template <typename FLOAT> DEVICE_PREFIX FLOAT Min(FLOAT lhs, FLOAT rhs) DEVICE_SUFFIX { return lhs < rhs ? lhs : rhs; }
template <typename FLOAT> DEVICE_PREFIX Vector3<FLOAT> Normalize(const Vector3<FLOAT>& val) DEVICE_SUFFIX { return val / Length(val.x, val.y, val.z); }
#pragma endregion

#pragma region - Section : Basic Matrix Math. -
template <typename FLOAT>
DEVICE_PREFIX Vector3<FLOAT> TransformPoint(const Matrix44<FLOAT>& m, const Vector3<FLOAT>& p) DEVICE_SUFFIX {
	return make_vector3<FLOAT>(
		m.M[0] * p.x + m.M[4] * p.y + m.M[8] * p.z + m.M[12],
		m.M[1] * p.x + m.M[5] * p.y + m.M[9] * p.z + m.M[13],
		m.M[2] * p.x + m.M[6] * p.y + m.M[10] * p.z + m.M[14]);
}

template <typename FLOAT>
DEVICE_PREFIX Vector3<FLOAT> TransformVector(const Matrix44<FLOAT>& m, const Vector3<FLOAT>& p) DEVICE_SUFFIX {
	return make_vector3<FLOAT>(
		m.M[0] * p.x + m.M[4] * p.y + m.M[8] * p.z,
		m.M[1] * p.x + m.M[5] * p.y + m.M[9] * p.z,
		m.M[2] * p.x + m.M[6] * p.y + m.M[10] * p.z);
}

template <typename FLOAT>
DEVICE_PREFIX Vector4<FLOAT> Transform(const Matrix44<FLOAT>& m, const Vector4<FLOAT>& p) DEVICE_SUFFIX {
	return make_vector4<FLOAT>(
		m.M[0] * p.x + m.M[4] * p.y + m.M[8] * p.z + m.M[12] * p.w,
		m.M[1] * p.x + m.M[5] * p.y + m.M[9] * p.z + m.M[13] * p.w,
		m.M[2] * p.x + m.M[6] * p.y + m.M[10] * p.z + m.M[14] * p.w,
		m.M[3] * p.x + m.M[7] * p.y + m.M[11] * p.z + m.M[15] * p.w);
}
#pragma endregion

#pragma region - Section : Secondary Ray and Optics Helpers. -
template <typename FLOAT>
DEVICE_PREFIX Vector3<FLOAT> Reflect(const Vector3<FLOAT>& incident, const Vector3<FLOAT>& normal) DEVICE_SUFFIX {
	return incident - 2 * Dot(incident, normal) * normal;
}

template <typename FLOAT>
DEVICE_PREFIX Vector3<FLOAT> Refract(const Vector3<FLOAT>& incident, const Vector3<FLOAT>& normal, FLOAT ior) DEVICE_SUFFIX {
	FLOAT cosi = Clamp<FLOAT>(-1, 1, Dot(incident, normal));
	FLOAT etai = 1, etat = ior;
	Vector3<FLOAT> n = normal;
	if (cosi < 0) { cosi = -cosi; }
	else { FLOAT tmp = etai; etai = etat; etat = tmp; n = -normal; }
	FLOAT eta = etai / etat;
	FLOAT k = 1 - eta * eta * (1 - cosi * cosi);
	return k < 0 ? make_vector3<FLOAT>(0, 0, 0) : (eta * incident + (eta * cosi - SquareRoot(k)) * n);
}

template <typename FLOAT>
DEVICE_PREFIX FLOAT SchlickApprox(const Vector3<FLOAT>& incident, const Vector3<FLOAT>& normal, FLOAT ior1, FLOAT ior2) DEVICE_SUFFIX {
	FLOAT coeff = (ior1 - ior2) / (ior1 + ior2);
	coeff = coeff * coeff;
	return coeff + (1 - coeff) * Power((1 - Dot(-incident, normal)), 5);
}
#pragma endregion

#pragma region - Section : Intersection engine and attribute calculation. -
template <typename FLOAT>
DEVICE_PREFIX FLOAT IntersectPlane(const Vector3<FLOAT>& origin, const Vector3<FLOAT>& direction, const Vector3<FLOAT>& plane_normal, FLOAT plane_distance) DEVICE_SUFFIX {
	return (plane_distance - Dot(plane_normal, origin)) / Dot(plane_normal, direction);
}

template <typename FLOAT> DEVICE_PREFIX void SetNormal(IntersectSimple<FLOAT>& result, Vector3<FLOAT> value) DEVICE_SUFFIX {}
template <typename FLOAT> DEVICE_PREFIX void SetNormal(IntersectNormal<FLOAT>& result, Vector3<FLOAT> value) DEVICE_SUFFIX { result.Normal = value; }
template <typename FLOAT> DEVICE_PREFIX void SetNormal(IntersectTBN<FLOAT>& result, Vector3<FLOAT> value) DEVICE_SUFFIX { result.Normal = value; }
template <typename FLOAT> DEVICE_PREFIX void SetNormal(IntersectDebugMesh<FLOAT>& result, Vector3<FLOAT> value) DEVICE_SUFFIX {}
template <typename FLOAT> DEVICE_PREFIX void SetTangent(IntersectSimple<FLOAT>& result, Vector3<FLOAT> value) DEVICE_SUFFIX {}
template <typename FLOAT> DEVICE_PREFIX void SetTangent(IntersectNormal<FLOAT>& result, Vector3<FLOAT> value) DEVICE_SUFFIX {}
template <typename FLOAT> DEVICE_PREFIX void SetTangent(IntersectTBN<FLOAT>& result, Vector3<FLOAT> value) DEVICE_SUFFIX { result.Tangent = value; }
template <typename FLOAT> DEVICE_PREFIX void SetTangent(IntersectDebugMesh<FLOAT>& result, Vector3<FLOAT> value) DEVICE_SUFFIX {}
template <typename FLOAT> DEVICE_PREFIX void SetBitangent(IntersectSimple<FLOAT>& result, Vector3<FLOAT> value) DEVICE_SUFFIX {}
template <typename FLOAT> DEVICE_PREFIX void SetBitangent(IntersectNormal<FLOAT>& result, Vector3<FLOAT> value) DEVICE_SUFFIX {}
template <typename FLOAT> DEVICE_PREFIX void SetBitangent(IntersectTBN<FLOAT>& result, Vector3<FLOAT> value) DEVICE_SUFFIX { result.Bitangent = value; }
template <typename FLOAT> DEVICE_PREFIX void SetBitangent(IntersectDebugMesh<FLOAT>& result, Vector3<FLOAT> value) DEVICE_SUFFIX {}
template <typename FLOAT> DEVICE_PREFIX Vector3<FLOAT> GetNormal(const IntersectSimple<FLOAT>& result) DEVICE_SUFFIX { return make_vector3<FLOAT>(0, 0, 0); }
template <typename FLOAT> DEVICE_PREFIX Vector3<FLOAT> GetNormal(const IntersectNormal<FLOAT>& result) DEVICE_SUFFIX { return result.Normal; }
template <typename FLOAT> DEVICE_PREFIX Vector3<FLOAT> GetNormal(const IntersectTBN<FLOAT>& result) DEVICE_SUFFIX { return result.Normal; }
template <typename FLOAT> DEVICE_PREFIX Vector3<FLOAT> GetNormal(const IntersectDebugMesh<FLOAT>& result) DEVICE_SUFFIX { return make_vector3<FLOAT>(0, 0, 0); }
template <typename FLOAT> DEVICE_PREFIX Vector3<FLOAT> GetTangent(const IntersectSimple<FLOAT>& result) DEVICE_SUFFIX { return make_vector3<FLOAT>(0, 0, 0); }
template <typename FLOAT> DEVICE_PREFIX Vector3<FLOAT> GetTangent(const IntersectNormal<FLOAT>& result) DEVICE_SUFFIX { return make_vector3<FLOAT>(0, 0, 0); }
template <typename FLOAT> DEVICE_PREFIX Vector3<FLOAT> GetTangent(const IntersectTBN<FLOAT>& result) DEVICE_SUFFIX { return result.Tangent; }
template <typename FLOAT> DEVICE_PREFIX Vector3<FLOAT> GetTangent(const IntersectDebugMesh<FLOAT>& result) DEVICE_SUFFIX { return make_vector3<FLOAT>(0, 0, 0); }
template <typename FLOAT> DEVICE_PREFIX Vector3<FLOAT> GetBitangent(const IntersectSimple<FLOAT>& result) DEVICE_SUFFIX { return make_vector3<FLOAT>(0, 0, 0); }
template <typename FLOAT> DEVICE_PREFIX Vector3<FLOAT> GetBitangent(const IntersectNormal<FLOAT>& result) DEVICE_SUFFIX { return make_vector3<FLOAT>(0, 0, 0); }
template <typename FLOAT> DEVICE_PREFIX Vector3<FLOAT> GetBitangent(const IntersectTBN<FLOAT>& result) DEVICE_SUFFIX { return result.Bitangent; }
template <typename FLOAT> DEVICE_PREFIX Vector3<FLOAT> GetBitangent(const IntersectDebugMesh<FLOAT>& result) DEVICE_SUFFIX { return make_vector3<FLOAT>(0, 0, 0); }

template <typename FLOAT> DEVICE_PREFIX void SetCountBoxTest(IntersectSimple<FLOAT>& result, int value) DEVICE_SUFFIX { }
template <typename FLOAT> DEVICE_PREFIX void SetCountBoxTest(IntersectNormal<FLOAT>& result, int value) DEVICE_SUFFIX { }
template <typename FLOAT> DEVICE_PREFIX void SetCountBoxTest(IntersectTBN<FLOAT>& result, int value) DEVICE_SUFFIX { }
template <typename FLOAT> DEVICE_PREFIX void SetCountBoxTest(IntersectDebugMesh<FLOAT>& result, int value) DEVICE_SUFFIX { result.CountBoxTest = value; }
template <typename FLOAT> DEVICE_PREFIX void SetCountTriangleTest(IntersectSimple<FLOAT>& result, int value) DEVICE_SUFFIX { }
template <typename FLOAT> DEVICE_PREFIX void SetCountTriangleTest(IntersectNormal<FLOAT>& result, int value) DEVICE_SUFFIX { }
template <typename FLOAT> DEVICE_PREFIX void SetCountTriangleTest(IntersectTBN<FLOAT>& result, int value) DEVICE_SUFFIX { }
template <typename FLOAT> DEVICE_PREFIX void SetCountTriangleTest(IntersectDebugMesh<FLOAT>& result, int value) DEVICE_SUFFIX { result.CountTriangleTest = value; }
template <typename FLOAT> DEVICE_PREFIX void SetCountTriangleHit(IntersectSimple<FLOAT>& result, int value) DEVICE_SUFFIX { }
template <typename FLOAT> DEVICE_PREFIX void SetCountTriangleHit(IntersectNormal<FLOAT>& result, int value) DEVICE_SUFFIX { }
template <typename FLOAT> DEVICE_PREFIX void SetCountTriangleHit(IntersectTBN<FLOAT>& result, int value) DEVICE_SUFFIX { }
template <typename FLOAT> DEVICE_PREFIX void SetCountTriangleHit(IntersectDebugMesh<FLOAT>& result, int value) DEVICE_SUFFIX { result.CountTriangleHit = value; }
template <typename FLOAT> DEVICE_PREFIX int GetCountBoxTest(IntersectSimple<FLOAT>& result) DEVICE_SUFFIX { return 0; }
template <typename FLOAT> DEVICE_PREFIX int GetCountBoxTest(IntersectNormal<FLOAT>& result) DEVICE_SUFFIX { return 0; }
template <typename FLOAT> DEVICE_PREFIX int GetCountBoxTest(IntersectTBN<FLOAT>& result) DEVICE_SUFFIX { return 0; }
template <typename FLOAT> DEVICE_PREFIX int GetCountBoxTest(IntersectDebugMesh<FLOAT>& result) DEVICE_SUFFIX { return result.CountBoxTest; }
template <typename FLOAT> DEVICE_PREFIX int GetCountTriangleTest(IntersectSimple<FLOAT>& result) DEVICE_SUFFIX { return 0; }
template <typename FLOAT> DEVICE_PREFIX int GetCountTriangleTest(IntersectNormal<FLOAT>& result) DEVICE_SUFFIX { return 0; }
template <typename FLOAT> DEVICE_PREFIX int GetCountTriangleTest(IntersectTBN<FLOAT>& result) DEVICE_SUFFIX { return 0; }
template <typename FLOAT> DEVICE_PREFIX int GetCountTriangleTest(IntersectDebugMesh<FLOAT>& result) DEVICE_SUFFIX { return result.CountTriangleTest; }
template <typename FLOAT> DEVICE_PREFIX int GetCountTriangleHit(IntersectSimple<FLOAT>& result) DEVICE_SUFFIX { return 0; }
template <typename FLOAT> DEVICE_PREFIX int GetCountTriangleHit(IntersectNormal<FLOAT>& result) DEVICE_SUFFIX { return 0; }
template <typename FLOAT> DEVICE_PREFIX int GetCountTriangleHit(IntersectTBN<FLOAT>& result) DEVICE_SUFFIX { return 0; }
template <typename FLOAT> DEVICE_PREFIX int GetCountTriangleHit(IntersectDebugMesh<FLOAT>& result) DEVICE_SUFFIX { return result.CountTriangleHit; }

// This is the slab method described in SIGGRAPH notes.
// See http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm
template <typename FLOAT>
DEVICE_PREFIX bool IntersectBound(const Vector3<FLOAT>& origin, const Vector3<FLOAT>& direction, const Vector3<FLOAT>& min, const Vector3<FLOAT>& max) DEVICE_SUFFIX {
	double tmin = -FloatTraits<FLOAT>::Max;
	double tmax = FloatTraits<FLOAT>::Max;
	double tx1 = (min.x - origin.x) / direction.x;
	double tx2 = (max.x - origin.x) / direction.x;
	tmin = Max(tmin, Min(tx1, tx2));
	tmax = Min(tmax, Max(tx1, tx2));
	double ty1 = (min.y - origin.y) / direction.y;
	double ty2 = (max.y - origin.y) / direction.y;
	tmin = Max(tmin, Min(ty1, ty2));
	tmax = Min(tmax, Max(ty1, ty2));
	double tz1 = (min.z - origin.z) / direction.z;
	double tz2 = (max.z - origin.z) / direction.z;
	tmin = Max(tmin, Min(tz1, tz2));
	tmax = Min(tmax, Max(tz1, tz2));
	return tmax >= tmin && (tmin >= 0 || tmax >= 0);
}

template <typename FLOAT, typename RAYQUERY>
DEVICE_PREFIX bool IntersectCube(RAYQUERY& result, const Vector3<FLOAT>& origin, const Vector3<FLOAT>& direction) DEVICE_SUFFIX {
	Vector3<FLOAT> face_normal[] = {
		make_vector3<FLOAT>(-1,0,0), make_vector3<FLOAT>(+1,0,0),
		make_vector3<FLOAT>(0,-1,0), make_vector3<FLOAT>(0,+1,0),
		make_vector3<FLOAT>(0,0,-1), make_vector3<FLOAT>(0,0,+1),
	};
	Vector3<FLOAT> face_tangent[] = {
		make_vector3<FLOAT>(0,0,-1), make_vector3<FLOAT>(0,0,+1),
		make_vector3<FLOAT>(+1,0,0), make_vector3<FLOAT>(+1,0,0),
		make_vector3<FLOAT>(+1,0,0), make_vector3<FLOAT>(-1,0,0),
	
	};
	Vector3<FLOAT> face_bitangent[] = {
		make_vector3<FLOAT>(0,-1,0), make_vector3<FLOAT>(0,-1,0),
		make_vector3<FLOAT>(0,0,+1), make_vector3<FLOAT>(0,0,-1),
		make_vector3<FLOAT>(0,-1,0), make_vector3<FLOAT>(0,-1,0),
	};
	FLOAT best_lambda = FloatTraits<FLOAT>::Max;
	Vector3<FLOAT> best_normal;
	Vector3<FLOAT> best_tangent;
	Vector3<FLOAT> best_bitangent;
	for (int face_index = 0; face_index < 6; ++face_index) {
		FLOAT lambda = IntersectPlane<FLOAT>(origin, direction, face_normal[face_index], 1);
		if (lambda < 0 || lambda > best_lambda) continue;
		Vector3<FLOAT> point = origin + lambda * direction;
		// Check that the point is inside every other plane.
		bool use_face = true;
		for (int check_face = 0; check_face < 6; ++check_face) {
			if (face_index == check_face) continue;
			FLOAT inside = Dot(point, face_normal[check_face]) - 1;
			if (inside > 0) {
				use_face = false;
				break;
			}
		}
		if (use_face) {
			best_lambda = lambda;
			best_normal = face_normal[face_index];
			best_tangent = face_tangent[face_index];
			best_bitangent = face_bitangent[face_index];
		}
	}
	if (best_lambda < 0 || best_lambda > result.Lambda) return false;
	result.Lambda = best_lambda;
	SetNormal(result, best_normal);
	SetTangent(result, best_tangent);
	SetBitangent(result, best_bitangent);
	return true;
}

template <typename FLOAT, typename RAYQUERY>
DEVICE_PREFIX bool IntersectPlane(RAYQUERY& result, const Vector3<FLOAT>& origin, const Vector3<FLOAT>& direction) DEVICE_SUFFIX {
	const FLOAT PLANE_DISTANCE = 0;
	const Vector3<FLOAT> PLANE_NORMAL = make_vector3<FLOAT>(0, 1, 0);
	FLOAT lambda = (PLANE_DISTANCE - Dot(PLANE_NORMAL, origin)) / Dot(PLANE_NORMAL, direction);
	if (lambda < 0 || lambda > result.Lambda) return false;
	result.Lambda = lambda;
	SetNormal(result, PLANE_NORMAL);
	SetTangent(result, make_vector3<FLOAT>(1, 0, 0));
	SetBitangent(result, make_vector3<FLOAT>(0, 0, -1));
	return true;
}

template <typename FLOAT, typename RAYQUERY>
DEVICE_PREFIX bool IntersectSphere(RAYQUERY& result, const Vector3<FLOAT>& origin, const Vector3<FLOAT>& direction) DEVICE_SUFFIX {
	const FLOAT SPHERE_RADIUS = 1;
	FLOAT a = Dot(direction, direction);
	FLOAT b = 2 * Dot(origin, direction);
	FLOAT c = Dot(origin, origin) - SPHERE_RADIUS * SPHERE_RADIUS;
	// If the determinant is negative then there are no real roots and this will be NaN.
	FLOAT det = SquareRoot(b * b - 4 * a * c);
	// "a" cannot be negative so (worst case) these lambdas are +Inf.
	FLOAT den = 2 * a;
	FLOAT lambda1 = (-b - det) / den;
	FLOAT lambda2 = (-b + det) / den;
	FLOAT lambda_best = FloatTraits<FLOAT>::Max;
	if (lambda1 >= 0 && lambda1 < lambda_best) lambda_best = lambda1;
	if (lambda2 >= 0 && lambda2 < lambda_best) lambda_best = lambda2;
	if (lambda_best < 0 || lambda_best > result.Lambda) return false;
	// Construct a tangent basis at this point.
	Vector3<FLOAT> vector_normal = origin + lambda_best * direction;
	Vector3<FLOAT> vector_bitangent = make_vector3<FLOAT>(0,-1,0);
	Vector3<FLOAT> vector_tangent = Cross(vector_bitangent, vector_normal);
	vector_bitangent = Cross(vector_normal, vector_tangent); // No normalization required.
	result.Lambda = lambda_best;
	SetNormal(result, vector_normal);
	SetTangent(result, vector_tangent);
	SetBitangent(result, vector_bitangent);
	return true;
}

// This is the planeless intersection described in "Fast, Minimum Storage Ray/Triangle Intersection".
// Credit to Tomas Möller and Ben Trumbore.
template <typename FLOAT, typename RAYQUERY>
DEVICE_PREFIX bool IntersectTriangle(RAYQUERY& result, const Vector3<FLOAT>& origin, const Vector3<FLOAT>& direction, const Vector3<FLOAT>& p0, const Vector3<FLOAT>& p1, const Vector3<FLOAT>& p2) DEVICE_SUFFIX {
	SetCountTriangleTest(result, GetCountTriangleTest(result) + 1);
	Vector3<FLOAT> edge1 = p1 - p0;
	Vector3<FLOAT> edge2 = p2 - p0;
	Vector3<FLOAT> pvec = Cross(direction, edge2);
	FLOAT det = Dot(edge1, pvec);
	// the non-culling branch */
	FLOAT inv_det = 1 / det;
	/* calculate distance from vert0 to ray origin */
	Vector3<FLOAT> tvec = origin - p0;
	/* calculate U parameter and test bounds */
	FLOAT u = Dot(tvec, pvec) * inv_det;
	if (u < 0 || u > 1) return false;
	/* prepare to test V parameter */
	Vector3<FLOAT> qvec = Cross(tvec, edge1);
	/* calculate V parameter and test bounds */
	FLOAT v = Dot(direction, qvec) * inv_det;
	if (v < 0 || u + v > 1) return false;
	/* calculate t, ray intersects triangle */
	FLOAT t = Dot(edge2, qvec) * inv_det;
	if (t < 0 || t > result.Lambda) return false;
	// END, back to the reference implementation.
	result.Lambda = t;
	const Vector3<FLOAT> plane_normal = Cross(p1 - p0, p2 - p0);
	const Vector3<FLOAT> edge0n = p1 - p0;
	SetNormal(result, plane_normal);
	SetTangent(result, edge0n);
	SetBitangent(result, Cross(plane_normal, edge0n));
	SetCountTriangleHit(result, GetCountTriangleHit(result) + 1);
	return true;
}

template <typename FLOAT, typename RAYQUERY, int RECURSION>
struct IntersectMeshBVHRecurse {
	static DEVICE_PREFIX bool Traverse(RAYQUERY& result, const Vector3<FLOAT>& origin, const Vector3<FLOAT>& direction, const Scene<FLOAT>& pScene, const MeshBVH<FLOAT>& node) DEVICE_SUFFIX {
		// Keep tally of any hits.
		bool foundhit = false;
		// TODO: C++AMP doesn't like this code at all and CUDA has a hard time compiling it.
		//       We're only going to build this for CPU and CUDA.
		#ifndef _WINRT_DLL
		// Make sure we're hitting this node first.
		SetCountBoxTest(result, GetCountBoxTest(result) + 1);
		if (!IntersectBound(origin, direction, node.Min, node.Max)) return foundhit;
		// Descend into child nodes if we have them.
		if (node.ChildrenOffset != 0) {
			const MeshBVHList<FLOAT>& children = *(const MeshBVHList<FLOAT>*)FollowPointer(pScene, node.ChildrenOffset);
			for (int i = 0; i < children.Count; ++i) {
				const MeshBVH<FLOAT>& child = children.Nodes[i];
				foundhit |= IntersectMeshBVHRecurse<FLOAT, RAYQUERY, RECURSION - 1>::Traverse(result, origin, direction, pScene, child);
			}
		}
		// Check the triangles if we have them.
		if (node.TriangleOffset != 0) {
			const Triangle3DList<FLOAT>& triangles = *(const Triangle3DList<FLOAT>*)FollowPointer(pScene, node.TriangleOffset);
			for (int i = 0; i < triangles.Count; ++i) {
				const Vector3<FLOAT> p0 = triangles.Triangles[i].P[0];
				const Vector3<FLOAT> p1 = triangles.Triangles[i].P[1];
				const Vector3<FLOAT> p2 = triangles.Triangles[i].P[2];
				foundhit |= IntersectTriangle<FLOAT, RAYQUERY>(result, origin, direction, p0, p1, p2);
			}
		}
		#endif  //_WINRT_DLL
		return foundhit;
	}
};

template <typename FLOAT, typename RAYQUERY>
struct IntersectMeshBVHRecurse<FLOAT, RAYQUERY, -1> {
	static DEVICE_PREFIX bool Traverse(RAYQUERY& result, const Vector3<FLOAT>& origin, const Vector3<FLOAT>& direction, const Scene<FLOAT>& pScene, const MeshBVH<FLOAT>& node) DEVICE_SUFFIX {
		return false;
	}
};

template <typename FLOAT, typename RAYQUERY>
DEVICE_PREFIX bool IntersectMeshBVH(RAYQUERY& result, const Vector3<FLOAT>& origin, const Vector3<FLOAT>& direction, const Scene<FLOAT>& pScene, const MeshBVH<FLOAT>& mesh) DEVICE_SUFFIX {
	return IntersectMeshBVHRecurse<FLOAT, RAYQUERY, 8>::Traverse(result, origin, direction, pScene, mesh);
}

template <typename FLOAT, typename RAYQUERY>
DEVICE_PREFIX bool Intersect(RAYQUERY& result, const Vector3<FLOAT>& origin, const Vector3<FLOAT>& direction, const Scene<FLOAT>& pScene, const SceneObject<FLOAT>& scene_object, const void* pObjectData) DEVICE_SUFFIX {
	switch (scene_object.Geometry) {
	case GEOMETRY_PLANE:
		return IntersectPlane<FLOAT, RAYQUERY>(result, origin, direction);
	case GEOMETRY_SPHERE:
		return IntersectSphere<FLOAT, RAYQUERY>(result, origin, direction);
	case GEOMETRY_CUBE:
		return IntersectCube<FLOAT, RAYQUERY>(result, origin, direction);
	case GEOMETRY_MESHBVH:
		return IntersectMeshBVH<FLOAT, RAYQUERY>(result, origin, direction, pScene, *(const MeshBVH<FLOAT>*)pObjectData);
	default:
		return false;
	}
}

template <typename FLOAT, typename RAYQUERY>
DEVICE_PREFIX bool Intersect(RAYQUERY& result, const Vector3<FLOAT>& origin, const Vector3<FLOAT>& direction, const Scene<FLOAT>& pScene, const SceneObject<FLOAT>& scene_object) DEVICE_SUFFIX {
	Vector3<FLOAT> transformed_origin = TransformPoint(scene_object.TransformInverse, origin);
	Vector3<FLOAT> transformed_direction = TransformVector(scene_object.TransformInverse, direction);
	if (!Intersect<FLOAT, RAYQUERY>(result, transformed_origin, transformed_direction, pScene, scene_object, FollowPointer(pScene, scene_object.GeometryOffset))) return false;
	SetNormal(result, Normalize(TransformVector(scene_object.Transform, GetNormal(result))));
	SetTangent(result, Normalize(TransformVector(scene_object.Transform, GetTangent(result))));
	SetBitangent(result, Normalize(TransformVector(scene_object.Transform, GetBitangent(result))));
	return true;
}

template <typename FLOAT, typename RAYQUERY>
DEVICE_PREFIX RAYQUERY Intersect(const Vector3<FLOAT>& origin, const Vector3<FLOAT>& direction, const Scene<FLOAT>& pScene, const SceneObject<FLOAT>& scene_object) DEVICE_SUFFIX {
	RAYQUERY result = { 0 };
	result.Lambda = FloatTraits<FLOAT>::Max;
	Intersect(result, origin, direction, pScene, scene_object);
	return result;
}

template <typename FLOAT>
DEVICE_PREFIX IntersectObject<FLOAT> Intersect(const Vector3<FLOAT>& origin, const Vector3<FLOAT>& direction, const Scene<FLOAT>& pScene) DEVICE_SUFFIX {
	FLOAT best_lambda = FloatTraits<FLOAT>::Max;
	const SceneObject<FLOAT>* best_object = nullptr;
	for (int i = 0; i < pScene.ObjectCount; ++i) {
		const SceneObject<FLOAT>& scene_object = pScene.Objects[i];
		FLOAT lambda = Intersect<FLOAT, IntersectSimple<FLOAT>>(origin, direction, pScene, scene_object).Lambda;
		if (lambda >= 0 && lambda < best_lambda) {
			best_lambda = lambda;
			best_object = &scene_object;
		}
	}
	IntersectObject<FLOAT> result;
	result.Lambda = best_lambda;
	if (best_object != nullptr) {
		result.Object = *best_object;
	}
	return result;
}

template <typename FLOAT>
DEVICE_PREFIX IntersectDebugMesh<FLOAT> IntersectMeshingHeatmap(const Scene<FLOAT>& pScene, const Vector3<FLOAT>& origin, const Vector3<FLOAT>& direction) DEVICE_SUFFIX {
	IntersectDebugMesh<FLOAT> result = { 0 };
	result.Lambda = FloatTraits<FLOAT>::Max;
	for (int i = 0; i < pScene.ObjectCount; ++i) {
		const SceneObject<FLOAT>& scene_object = pScene.Objects[i];
		Intersect<FLOAT, IntersectDebugMesh<FLOAT>>(result, origin, direction, pScene, scene_object);
	}
	return result;
}

template <typename FLOAT>
DEVICE_PREFIX bool RayShadowBounded(const Scene<FLOAT>& pScene, const Vector3<FLOAT>& origin, const Vector3<FLOAT>& direction) DEVICE_SUFFIX {
	for (int i = 0; i < pScene.ObjectCount; ++i) {
		FLOAT lambda = Intersect<FLOAT, IntersectSimple<FLOAT>>(origin, direction, pScene, pScene.Objects[i]).Lambda;
		if (lambda >= 0 && lambda < 1) return true;
	}
	return false;
}

template <typename FLOAT>
DEVICE_PREFIX bool RayShadowInfinite(const Scene<FLOAT>& pScene, const Vector3<FLOAT>& origin, const Vector3<FLOAT>& direction) DEVICE_SUFFIX {
	for (int i = 0; i < pScene.ObjectCount; ++i) {
		FLOAT lambda = Intersect<FLOAT, IntersectSimple<FLOAT>>(origin, direction, pScene, pScene.Objects[i]).Lambda;
		if (lambda >= 0 && lambda < FloatTraits<FLOAT>::Max) return true;
	}
	return false;
}
#pragma endregion

#pragma region - Section : Core Raytracer. -
template <typename FLOAT>
DEVICE_PREFIX void ComputeScreenRay(const Matrix44<FLOAT>& inverse_mvp, FLOAT clipx, FLOAT clipy, Vector3<FLOAT>& origin, Vector3<FLOAT>& direction) DEVICE_SUFFIX {
	Vector4<FLOAT> v41 = Transform(inverse_mvp, make_vector4<FLOAT>(clipx, clipy, 0, 1));
	Vector4<FLOAT> v42 = Transform(inverse_mvp, make_vector4<FLOAT>(clipx, clipy, 1, 1));
	Vector3<FLOAT> ray_p1 = make_vector3<FLOAT>(v41.x / v41.w, v41.y / v41.w, v41.z / v41.w);
	Vector3<FLOAT> ray_p2 = make_vector3<FLOAT>(v42.x / v42.w, v42.y / v42.w, v42.z / v42.w);
	origin = ray_p1;
	direction = ray_p2 - ray_p1;
}

template <typename FLOAT>
DEVICE_PREFIX unsigned int Vector4ToA8R8G8B8(const Vector4<FLOAT>& color) DEVICE_SUFFIX {
	// We scale to 254 because there's a large gray difference between #FEFEFE and #FFFFFF.
	// I can only assume this is a color calibration issue on my laptop screen but whatever.
	unsigned int r = unsigned int(Clamp<FLOAT>(0, 1, color.x) * 254);
	unsigned int g = unsigned int(Clamp<FLOAT>(0, 1, color.y) * 254);
	unsigned int b = unsigned int(Clamp<FLOAT>(0, 1, color.z) * 254);
	unsigned int a = unsigned int(Clamp<FLOAT>(0, 1, color.w) * 255);
	return (a << 24) | (r << 16) | (g << 8) | (b << 0);
}

template<typename FLOAT>
struct PixelAccumulateVec4 {
	DEVICE_PREFIX PixelAccumulateVec4(void* bitmap_ptr, int render_width, int render_height, int bitmap_stride, int x, int y) DEVICE_SUFFIX :
	bitmap_ptr(bitmap_ptr), render_width(render_width), render_height(render_height), bitmap_stride(bitmap_stride), x(x), y(y) { }
	DEVICE_PREFIX int GetX() const DEVICE_SUFFIX { return x; }
	DEVICE_PREFIX int GetY() const DEVICE_SUFFIX { return y; }
	DEVICE_PREFIX int GetWidth() const DEVICE_SUFFIX { return render_width; }
	DEVICE_PREFIX int GetHeight() const DEVICE_SUFFIX { return render_height; }
	DEVICE_PREFIX void PutPixel(const Vector4<FLOAT>& color) const DEVICE_SUFFIX {
		void* pRaster = (unsigned char*)bitmap_ptr + bitmap_stride * y;
		Vector4<FLOAT>* pPixel = (Vector4<FLOAT>*)pRaster + x;
		*pPixel = *pPixel + color;
	}
	void* bitmap_ptr;
	int render_width, render_height, bitmap_stride, x, y;
};

template<typename FLOAT>
struct PixelSetARGB {
	DEVICE_PREFIX PixelSetARGB(void* bitmap_ptr, int render_width, int render_height, int bitmap_stride, int x, int y) DEVICE_SUFFIX :
		bitmap_ptr(bitmap_ptr), render_width(render_width), render_height(render_height), bitmap_stride(bitmap_stride), x(x), y(y) { }
	DEVICE_PREFIX int GetX() const DEVICE_SUFFIX { return x; }
	DEVICE_PREFIX int GetY() const DEVICE_SUFFIX { return y; }
	DEVICE_PREFIX int GetWidth() const DEVICE_SUFFIX { return render_width; }
	DEVICE_PREFIX int GetHeight() const DEVICE_SUFFIX { return render_height; }
	DEVICE_PREFIX void PutPixel(const Vector4<FLOAT>& color) const DEVICE_SUFFIX {
		void* pRaster = (unsigned char*)bitmap_ptr + bitmap_stride * y;
		unsigned int* pPixel = (unsigned int*)pRaster + x;
		*pPixel = Vector4ToA8R8G8B8(color);
	}
	void* bitmap_ptr;
	int render_width, render_height, bitmap_stride, x, y;
};

template<typename FLOAT>
struct PixelSetScaledARGB {
	DEVICE_PREFIX PixelSetScaledARGB(void* bitmap_ptr, int render_width, int render_height, int bitmap_stride, int x, int y, FLOAT scale) DEVICE_SUFFIX :
	bitmap_ptr(bitmap_ptr), render_width(render_width), render_height(render_height), bitmap_stride(bitmap_stride), x(x), y(y), scale(scale) { }
	DEVICE_PREFIX int GetX() const DEVICE_SUFFIX { return x; }
	DEVICE_PREFIX int GetY() const DEVICE_SUFFIX { return y; }
	DEVICE_PREFIX int GetWidth() const DEVICE_SUFFIX { return render_width; }
	DEVICE_PREFIX int GetHeight() const DEVICE_SUFFIX { return render_height; }
	DEVICE_PREFIX void PutPixel(const Vector4<FLOAT>& color) const DEVICE_SUFFIX {
		void* pRaster = (unsigned char*)bitmap_ptr + bitmap_stride * y;
		unsigned int* pPixel = (unsigned int*)pRaster + x;
		*pPixel = Vector4ToA8R8G8B8(color * scale);
	}
	void* bitmap_ptr;
	int render_width, render_height, bitmap_stride, x, y;
	FLOAT scale;
};
template <typename FLOAT, typename RENDERMODE, typename SETPIXEL>
DEVICE_PREFIX void ComputePixel(const Scene<FLOAT>& pScene, Matrix44<FLOAT> inverse_mvp, const SETPIXEL& setpixel, int superx = 1, int supery = 1) DEVICE_SUFFIX {
	if (setpixel.GetX() >= setpixel.GetWidth() || setpixel.GetY() >= setpixel.GetHeight()) return;
	Vector3<FLOAT> origin;
	Vector3<FLOAT> direction;
	Vector4<FLOAT> color = make_vector4<FLOAT>(0, 0, 0, 0);
	for (int y_supersample = 1; y_supersample <= supery; ++y_supersample) {
		for (int x_supersample = 1; x_supersample <= superx; ++x_supersample) {
			// Build a ray for this supersample.
			FLOAT vx = Lerp<FLOAT>(-1, +1, (setpixel.GetX() + FLOAT(x_supersample) / (superx + 1)) / setpixel.GetWidth());
			FLOAT vy = Lerp<FLOAT>(+1, -1, (setpixel.GetY() + FLOAT(y_supersample) / (supery + 1)) / setpixel.GetHeight());
			ComputeScreenRay(inverse_mvp, vx, vy, origin, direction);
			// Compute intersection with plane.
			color = color + RENDERMODE::CalculateColor(pScene, origin, direction);
		}
	}
	setpixel.PutPixel(color / FLOAT(superx * supery));
}
#pragma endregion

#pragma region - Render Mode : Raycast. -
template<typename FLOAT>
struct RenderModeRaycast {
public:
	DEVICE_PREFIX static Vector4<FLOAT> CalculateColor(const Scene<FLOAT>& pScene, const Vector3<FLOAT>& origin, const Vector3<FLOAT>& direction) DEVICE_SUFFIX {
		auto intersect = Intersect(origin, direction, pScene);
		if (intersect.Lambda == FloatTraits<FLOAT>::Max) {
			return make_vector4<FLOAT>(0, 0, 0, 0);
		}
		Vector3<FLOAT> vector_point = origin + intersect.Lambda * direction;
		switch (intersect.Object.Material) {
		case MATERIAL_COMMON:
		{
			if (intersect.Object.MaterialOffset != 0) {
				const MaterialCommon<FLOAT>* pMaterial = (const MaterialCommon<FLOAT>*)FollowPointer(pScene, intersect.Object.MaterialOffset);
				return pMaterial->Diffuse;
			}
		}
		break;
		case MATERIAL_CHECKERBOARD_XZ:
		{
			int mx = (vector_point.x - floor(vector_point.x)) * 2;
			int mz = (vector_point.z - floor(vector_point.z)) * 2;
			FLOAT c = FLOAT((mx + mz) & 1);
			return make_vector4<FLOAT>(c, c, c, 1);
		}
		}
		return make_vector4<FLOAT>(0, 0, 0, 0);
	}
};
#pragma endregion

#pragma region - Render Mode : Raycast Bitangents. -
template<typename FLOAT>
struct RenderModeRaycastBitangents {
public:
	DEVICE_PREFIX static Vector4<FLOAT> CalculateColor(const Scene<FLOAT>& pScene, const Vector3<FLOAT>& origin, const Vector3<FLOAT>& direction) DEVICE_SUFFIX {
		auto intersect = Intersect(origin, direction, pScene);
		if (intersect.Lambda == FloatTraits<FLOAT>::Max) {
			return make_vector4<FLOAT>(0, 0, 0, 0);
		}
		Vector3<FLOAT> n = Normalize(Intersect<FLOAT, IntersectTBN<FLOAT>>(origin, direction, pScene, intersect.Object).Bitangent);
		return make_vector4<FLOAT>((n.x + 1) / 2, (n.y + 1) / 2, (n.z + 1) / 2, 1);
	}
};
#pragma endregion

#pragma region - Render Mode : Raycast Normals. -
template<typename FLOAT>
struct RenderModeRaycastNormals {
public:
	DEVICE_PREFIX static Vector4<FLOAT> CalculateColor(const Scene<FLOAT>& pScene, const Vector3<FLOAT>& origin, const Vector3<FLOAT>& direction) DEVICE_SUFFIX {
		auto intersect = Intersect(origin, direction, pScene);
		if (intersect.Lambda == FloatTraits<FLOAT>::Max) {
			return make_vector4<FLOAT>(0, 0, 0, 0);
		}
		Vector3<FLOAT> n = Normalize(Intersect<FLOAT, IntersectNormal<FLOAT>>(origin, direction, pScene, intersect.Object).Normal);
		return make_vector4<FLOAT>((n.x + 1) / 2, (n.y + 1) / 2, (n.z + 1) / 2, 1);
	}
};
#pragma endregion

#pragma region - Render Mode : Raycast Tangents. -
template<typename FLOAT>
struct RenderModeRaycastTangents {
public:
	DEVICE_PREFIX static Vector4<FLOAT> CalculateColor(const Scene<FLOAT>& pScene, const Vector3<FLOAT>& origin, const Vector3<FLOAT>& direction) DEVICE_SUFFIX {
		auto intersect = Intersect(origin, direction, pScene);
		if (intersect.Lambda == FloatTraits<FLOAT>::Max) {
			return make_vector4<FLOAT>(0, 0, 0, 0);
		}
		Vector3<FLOAT> n = Normalize(Intersect<FLOAT, IntersectTBN<FLOAT>>(origin, direction, pScene, intersect.Object).Tangent);
		return make_vector4<FLOAT>((n.x + 1) / 2, (n.y + 1) / 2, (n.z + 1) / 2, 1);
	}
};
#pragma endregion

#pragma region - Render Mode : Raytrace. -
template<typename FLOAT, int RECURSE = 2>
struct RenderModeRaytrace {
public:
	DEVICE_PREFIX static Vector4<FLOAT> CalculateColor(const Scene<FLOAT>& pScene, const Vector3<FLOAT>& origin, const Vector3<FLOAT>& direction) DEVICE_SUFFIX {
		// Start intersecting objects.
		auto intersect = Intersect(origin, direction, pScene);
		if (intersect.Lambda == FloatTraits<FLOAT>::Max) {
			return make_vector4<FLOAT>(0, 0, 0, 0);
		}
		const Vector3<FLOAT> vector_point = origin + intersect.Lambda * direction;
		const Vector3<FLOAT> vector_light = make_vector3<FLOAT>(10, 10, -10) - vector_point;
		const Vector3<FLOAT> vector_light_normal = Normalize(vector_light);
		const Vector3<FLOAT> vector_normal = Normalize(Intersect<FLOAT, IntersectNormal<FLOAT>>(origin, direction, pScene, intersect.Object).Normal);
		Vector4<FLOAT> color_diffuse = make_vector4<FLOAT>(1, 1, 1, 0);
		Vector4<FLOAT> color_specular = make_vector4<FLOAT>(1, 1, 1, 0);
		FLOAT scale_reflect = 0;
		FLOAT scale_refract = 0;
		FLOAT scale_ior = 1;
		switch (intersect.Object.Material) {
		case MATERIAL_COMMON:
		{
			if (intersect.Object.MaterialOffset != 0) {
				const MaterialCommon<FLOAT>* pMaterial = (const MaterialCommon<FLOAT>*)FollowPointer(pScene, intersect.Object.MaterialOffset);
				color_diffuse = pMaterial->Diffuse;
				color_specular = pMaterial->Specular;
				scale_reflect = pMaterial->Reflect.w;
				scale_refract = pMaterial->Refract.w;
				scale_ior = pMaterial->Ior;
			}
		}
		break;
		case MATERIAL_CHECKERBOARD_XZ:
		{
			int mx = (vector_point.x - floor(vector_point.x)) < 0.5 ? 0 : 1;
			int mz = (vector_point.z - floor(vector_point.z)) < 0.5 ? 0 : 1;
			FLOAT c = FLOAT((mx + mz) % 2);
			color_diffuse = make_vector4<FLOAT>(c, c, c, 0);
			color_specular = make_vector4<FLOAT>(1, 1, 1, 0);
			scale_reflect = 1;
			scale_refract = 0;
		}
		break;
		}
		const Vector3<FLOAT> vector_incident = Normalize(direction);
		const Vector3<FLOAT> vector_reflect = Reflect(vector_incident, vector_normal);
		const Vector3<FLOAT> vector_refract = Refract(vector_incident, vector_normal, scale_ior);
		FLOAT scale_diffuse = Dot(vector_light_normal, vector_normal);
		scale_diffuse = scale_diffuse > 0 ? scale_diffuse : 0;
		FLOAT scale_specular = Dot(vector_light_normal, vector_reflect);
		scale_specular = scale_specular > 0 ? scale_specular : 0;
		scale_specular = Power(scale_specular, 100);
		if (RayShadowBounded(pScene, vector_point + FLOAT(0.0001) * vector_normal, vector_light)) scale_diffuse *= FLOAT(0.5);
		Vector4<FLOAT> color = color_diffuse * scale_diffuse + color_specular * scale_specular;
		if (scale_refract > 0) {
			//Schlick's Approximation for reflectance.
			FLOAT R = SchlickApprox<FLOAT>(vector_incident, vector_normal, 1, scale_ior);
			scale_reflect = R;
			//Add refraction contribution.
			color = color + RenderModeRaytrace<FLOAT, RECURSE - 1>::CalculateColor(pScene, vector_point + vector_refract * FLOAT(0.001), vector_refract) * scale_refract;
		}
		if (scale_reflect > 0) {
			color = color + RenderModeRaytrace<FLOAT, RECURSE - 1>::CalculateColor(pScene, vector_point + vector_reflect * FLOAT(0.001), vector_reflect) * scale_reflect;
		}
		color.w = 1;
		return color;
	}
};

// Recursion Terminator Specialization (Recursion Depth -1 always evaluates to black).
template <typename FLOAT>
struct RenderModeRaytrace<FLOAT, -1>
{
	DEVICE_PREFIX static Vector4<FLOAT> CalculateColor(const Scene<FLOAT>& pScene, const Vector3<FLOAT>& origin, const Vector3<FLOAT>& direction) DEVICE_SUFFIX {
		return make_vector4<FLOAT>(0, 0, 0, 0);
	}
};
#pragma endregion

#pragma region - Render Mode : Debug Mesh -
template<typename FLOAT>
struct RenderModeDebugMesh {
public:
	DEVICE_PREFIX static Vector4<FLOAT> CalculateColor(const Scene<FLOAT>& pScene, const Vector3<FLOAT>& origin, const Vector3<FLOAT>& direction) DEVICE_SUFFIX {
		auto result = IntersectMeshingHeatmap<FLOAT>(pScene, origin, direction);
		return make_vector4<FLOAT>(result.CountTriangleTest * FLOAT(0.01), result.CountBoxTest * FLOAT(0.01), result.CountTriangleHit * FLOAT(0.25), 1);
	}
};
#pragma endregion

#pragma region - Render Mode : Ambient Occlusion. -
template <typename FLOAT>
DEVICE_PREFIX Vector3<FLOAT> HemisphereCosineBias(FLOAT u, FLOAT v) DEVICE_SUFFIX {
	FLOAT azimuth = u * 2 * FloatTraits<FLOAT>::PI;
	FLOAT altitude = Acos(SquareRoot(v));
	FLOAT cos_alt = Cos(altitude);
	return make_vector3<FLOAT>(Sin(azimuth) * cos_alt, Cos(azimuth) * cos_alt, Sin(altitude));
}

template <typename FLOAT>
DEVICE_PREFIX FLOAT Halton(int radix, int x) DEVICE_SUFFIX {
	FLOAT result = 0;
	int div = 1;
	for (int i = 0; i < 16; ++i) {
		int mod = x % radix;
		div = div * radix;
		result += FLOAT(mod) / div;
		x = x / radix;
	}
	return result;
}

template <typename FLOAT>
DEVICE_PREFIX Vector3<FLOAT> HaltonSample(int x) DEVICE_SUFFIX {
	return HemisphereCosineBias<FLOAT>(Halton<FLOAT>(2, x), Halton<FLOAT>(3, x));
}

template <typename FLOAT>
struct HemisampleHalton {
	DEVICE_PREFIX HemisampleHalton(int sample_offset, int sample_count) DEVICE_SUFFIX : sample_offset(sample_offset), sample_count(sample_count) {}
	DEVICE_PREFIX int Count() DEVICE_SUFFIX { return sample_count; }
	DEVICE_PREFIX Vector3<FLOAT> Sample(int i) DEVICE_SUFFIX { return HaltonSample<FLOAT>(sample_offset + i); }
	int sample_offset, sample_count;
};

template <typename FLOAT, typename HEMISAMPLE>
DEVICE_PREFIX Vector4<FLOAT> CalculateColorAOC(const Scene<FLOAT>& pScene, const Vector3<FLOAT>& origin, const Vector3<FLOAT>& direction, HEMISAMPLE& hemisamples) DEVICE_SUFFIX {
	auto intersect = Intersect(origin, direction, pScene);
	if (intersect.Lambda == FloatTraits<FLOAT>::Max) {
		return make_vector4<FLOAT>(0, 0, 0, 0);
	}
	auto intersect_tbn = Intersect<FLOAT, IntersectTBN<FLOAT>>(origin, direction, pScene, intersect.Object);
	auto vector_point = origin + intersect_tbn.Lambda * direction + FLOAT(0.001) * intersect_tbn.Normal;
	FLOAT c = 0;
	for (int i = 0; i < hemisamples.Count(); ++i)
	{
		Vector3<FLOAT> hemisample = hemisamples.Sample(i);
		const Vector3<FLOAT> hemisample_mapped = hemisample.x * intersect_tbn.Tangent + hemisample.y * intersect_tbn.Bitangent + hemisample.z * intersect_tbn.Normal;
		if (!RayShadowInfinite(pScene, vector_point, hemisample_mapped)) c += 1; //Dot(hemisample_mapped, intersect_tbn.Normal);
	}
	return make_vector4<FLOAT>(c, c, c, FLOAT(hemisamples.Count()));
}

template <typename FLOAT, typename SETPIXEL = PixelSetARGB<FLOAT>, typename HEMISAMPLE>
DEVICE_PREFIX void ComputePixelAOC(const Scene<FLOAT>& pScene, Matrix44<FLOAT> inverse_mvp, const SETPIXEL& setpixel, HEMISAMPLE& hemisamples) DEVICE_SUFFIX {
	if (setpixel.GetX() >= setpixel.GetWidth() || setpixel.GetY() >= setpixel.GetHeight()) return;
	Vector3<FLOAT> origin;
	Vector3<FLOAT> direction;
	// Build a ray for this supersample.
	FLOAT vx = Lerp<FLOAT>(-1, +1, (setpixel.GetX() + FLOAT(0.5)) / setpixel.GetWidth());
	FLOAT vy = Lerp<FLOAT>(+1, -1, (setpixel.GetY() + FLOAT(0.5)) / setpixel.GetHeight());
	ComputeScreenRay(inverse_mvp, vx, vy, origin, direction);
	// Compute intersection with plane.
	setpixel.PutPixel(CalculateColorAOC<FLOAT>(pScene, origin, direction, hemisamples));
}
#pragma endregion