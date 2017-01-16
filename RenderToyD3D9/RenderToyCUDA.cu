#include <cuda_runtime.h>
#include <math.h>

////////////////////////////////////////////////////////////////////////////////
// Device Code.
////////////////////////////////////////////////////////////////////////////////

// Basic Math Primitives.
__device__ double3 operator-(const double3& val) { return make_double3(-val.x, -val.y, -val.z); }
__device__ double4 operator-(const double4& val) { return make_double4(-val.z, -val.y, -val.z, -val.w); }
__device__ double3 operator+(const double3& lhs, const double3& rhs) { return make_double3(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z); }
__device__ double4 operator+(const double4& lhs, const double4& rhs) { return make_double4(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w); }
__device__ double3 operator-(const double3& lhs, const double3& rhs) { return make_double3(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z); }
__device__ double4 operator-(const double4& lhs, const double4& rhs) { return make_double4(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w); }
__device__ double3 operator*(const double3& lhs, double rhs) { return make_double3(lhs.x * rhs, lhs.y * rhs, lhs.z * rhs); }
__device__ double3 operator*(double lhs, const double3& rhs) { return rhs * lhs; }
__device__ double4 operator*(const double4& lhs, double rhs) { return make_double4(lhs.x * rhs, lhs.y * rhs, lhs.z * rhs, lhs.w * rhs); }
__device__ double4 operator*(double lhs, const double4& rhs) { return rhs * lhs; }
__device__ double3 operator/(const double3 &lhs, double rhs) { return lhs * (1 / rhs); }
__device__ double4 operator/(const double4 &lhs, double rhs) { return lhs * (1 / rhs); }

// Common Math Primitives.
__device__ double Clamp(double min, double max, double val) { return val < min ? min : (val > max ? max : val); }
__device__ double Dot(const double3& lhs, const double3& rhs) { return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z; }
__device__ double Length(const double3& val) { return norm3d(val.x, val.y, val.z); }
__device__ double Lerp(double y1, double y2, double x) { return y1 + (y2 - y1) * x; }
__device__ double3 Normalize(const double3 &val) { return val * rnorm3d(val.x, val.y, val.z); }

// Matrix Math.
struct Matrix4D { double M[16]; };

__device__ Matrix4D CreateMatrixIdentity() {
	double m[] = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };
	return *(Matrix4D*)m;
}

__device__ Matrix4D CreateMatrixTranslate(double x, double y, double z) {
	double m[] = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1 };
	return *(Matrix4D*)m;
}

__device__ double3 TransformPoint(const Matrix4D& m, const double3& p) {
	return make_double3(
		m.M[0] * p.x + m.M[4] * p.y + m.M[8] * p.z + m.M[12],
		m.M[1] * p.x + m.M[5] * p.y + m.M[9] * p.z + m.M[13],
		m.M[2] * p.x + m.M[6] * p.y + m.M[10] * p.z + m.M[14]);
}

__device__ double3 TransformVector(const Matrix4D& m, const double3& p) {
	return make_double3(
		m.M[0] * p.x + m.M[4] * p.y + m.M[8] * p.z,
		m.M[1] * p.x + m.M[5] * p.y + m.M[9] * p.z,
		m.M[2] * p.x + m.M[6] * p.y + m.M[10] * p.z);
}

__device__ double4 Transform(const Matrix4D& m, const double4& p) {
	return make_double4(
		m.M[0] * p.x + m.M[4] * p.y + m.M[8] * p.z + m.M[12] * p.w,
		m.M[1] * p.x + m.M[5] * p.y + m.M[9] * p.z + m.M[13] * p.w,
		m.M[2] * p.x + m.M[6] * p.y + m.M[10] * p.z + m.M[14] * p.w,
		m.M[3] * p.x + m.M[7] * p.y + m.M[11] * p.z + m.M[15] * p.w);
}

__device__ double3 Reflect(const double3& incident, const double3& normal) {
	return incident - 2 * Dot(incident, normal) * normal;
}

__device__ double3 Refract(const double3& incident, const double3& normal, double ior) {
	double cosi = Clamp(-1, 1, Dot(incident, normal));
	double etai = 1, etat = ior;
	double3 n = normal;
	if (cosi < 0) { cosi = -cosi; }
	else { double tmp = etai; etai = etat; etat = tmp; n = -normal; }
	double eta = etai / etat;
	double k = 1 - eta * eta * (1 - cosi * cosi);
	return k < 0 ? make_double3(0, 0, 0) : (eta * incident + (eta * cosi - sqrt(k)) * n);
}

__device__ double SchlickApprox(const double3& incident, const double3 &normal, double ior1, double ior2) {
	double coeff = (ior1 - ior2) / (ior1 + ior2);
	coeff = coeff * coeff;
	return coeff + (1 - coeff) * pow((1 - Dot(-incident, normal)), 5);
}

// Geometric Math.
__device__ double IntersectPlane(const double3 &origin, const double3 &direction) {
	const double PLANE_DISTANCE = 0;
	const double3 PLANE_NORMAL = make_double3(0, 1, 0);
	return (PLANE_DISTANCE - Dot(PLANE_NORMAL, origin)) / Dot(PLANE_NORMAL, direction);
}

__device__ double IntersectSphere(const double3 &origin, const double3 &direction) {
	const double SPHERE_RADIUS = 1;
	double a = Dot(direction, direction);
	double b = 2 * Dot(origin, direction);
	double c = Dot(origin, origin) - SPHERE_RADIUS * SPHERE_RADIUS;
	// If the determinant is negative then there are no real roots and this will be NaN.
	double det = sqrt(b * b - 4 * a * c);
	// "a" cannot be negative so (worst case) these lambdas are +Inf.
	double den = 2 * a;
	double lambda1 = (-b - det) / den;
	double lambda2 = (-b + det) / den;
	double lambda_best = 1000000;
	if (lambda1 >= 0 && lambda1 < lambda_best) lambda_best = lambda1;
	if (lambda2 >= 0 && lambda2 < lambda_best) lambda_best = lambda2;
	return lambda_best;
}

enum GeometryType {
	GEOMETRY_NONE = 0,
	GEOMETRY_PLANE = 1,
	GEOMETRY_SPHERE = 2
};

enum MaterialType {
	MATERIAL_NONE = 0,
	MATERIAL_CHECKERBOARD_XZ = 1,
	MATERIAL_RED,
	MATERIAL_GREEN,
	MATERIAL_BLUE,
	MATERIAL_GLASS
};

__device__ double Intersect(const double3 &origin, const double3 &direction, GeometryType geometry) {
	switch (geometry) {
	case GEOMETRY_PLANE:
		return IntersectPlane(origin, direction);
	case GEOMETRY_SPHERE:
		return IntersectSphere(origin, direction);
	default: return 1000000;
	}
}

struct SceneObject {
	Matrix4D Transform;
	Matrix4D TransformInverse;
	GeometryType Geometry;
	MaterialType Material;
};

const int SCENE_COUNT = 4;

// Raytracer Ray Tests.
__device__ bool RayShadow(const double3 &origin, const double3 &direction) {
	const SceneObject TheScene[] = {
		SceneObject{ CreateMatrixIdentity(), CreateMatrixIdentity(), GEOMETRY_PLANE, MATERIAL_CHECKERBOARD_XZ },
		SceneObject{ CreateMatrixTranslate(-2, 1, 0), CreateMatrixTranslate(2, -1, 0), GEOMETRY_SPHERE, MATERIAL_RED },
		SceneObject{ CreateMatrixTranslate(0, 1, 0), CreateMatrixTranslate(0, -1, 0), GEOMETRY_SPHERE, MATERIAL_GREEN },
		SceneObject{ CreateMatrixTranslate(2, 1, 0), CreateMatrixTranslate(-2, -1, 0), GEOMETRY_SPHERE, MATERIAL_BLUE },
	};
	for (int i = 0; i < SCENE_COUNT; ++i) {
		const SceneObject &scene_object = TheScene[i];
		double3 transformed_origin = TransformPoint(scene_object.TransformInverse, origin);
		double3 transformed_direction = TransformVector(scene_object.TransformInverse, direction);
		double lambda = Intersect(transformed_origin, transformed_direction, scene_object.Geometry);
		if (lambda >= 0 && lambda < 1000000) return true;
	}
	return false;
}

template <int RECURSE>
__device__ double4 RayColor(const double3 &origin, const double3 &direction) {
	// Start intersecting objects.
	const SceneObject TheScene[] = {
		SceneObject{ CreateMatrixIdentity(), CreateMatrixIdentity(), GEOMETRY_PLANE, MATERIAL_CHECKERBOARD_XZ },
		SceneObject{ CreateMatrixTranslate(-2, 1, 0), CreateMatrixTranslate(2, -1, 0), GEOMETRY_SPHERE, MATERIAL_RED },
		SceneObject{ CreateMatrixTranslate(0, 1, 0), CreateMatrixTranslate(0, -1, 0), GEOMETRY_SPHERE, MATERIAL_GREEN },
		SceneObject{ CreateMatrixTranslate(2, 1, 0), CreateMatrixTranslate(-2, -1, 0), GEOMETRY_SPHERE, MATERIAL_BLUE },
	};
	double best_lambda = 1000000;
	int best_pobject = 0;
	for (int i = 0; i < SCENE_COUNT; ++i) {
		const SceneObject &scene_object = TheScene[i];
		double3 transformed_origin = TransformPoint(scene_object.TransformInverse, origin);
		double3 transformed_direction = TransformVector(scene_object.TransformInverse, direction);
		double lambda = Intersect(transformed_origin, transformed_direction, scene_object.Geometry);
		if (lambda >= 0 && lambda < best_lambda) {
			best_lambda = lambda;
			best_pobject = i;
		}
	}
	if (best_lambda == 1000000) {
		return make_double4(0, 0, 0, 0);
	}
	{
		double3 p = origin + best_lambda * direction;
		double3 l = Normalize(make_double3(10, 10, -10) - p);
		double3 n;
		switch (TheScene[best_pobject].Geometry) {
		case GEOMETRY_PLANE:
			n = make_double3(0, 1, 0);
			break;
		case GEOMETRY_SPHERE:
			n = TransformPoint(TheScene[best_pobject].TransformInverse, p);
			break;
		default:
			n = make_double3(0, 1, 0);
		}
		double3 i = Normalize(direction);
		double3 r = Reflect(i, n);
		double3 t = Refract(i, n, 1.5);
		double scale_diffuse = Dot(l, n);
		scale_diffuse = scale_diffuse > 0 ? scale_diffuse : 0;
		double scale_specular = Dot(l, r);
		scale_specular = scale_specular > 0 ? scale_specular : 0;
		scale_specular = pow(scale_specular, 100);
		if (RayShadow(p + 0.0001 * n, l)) scale_diffuse *= 0.5;
		double4 color_diffuse;
		double4 color_specular;
		double scale_reflect;
		double scale_refract;
		switch (TheScene[best_pobject].Material) {
		case MATERIAL_CHECKERBOARD_XZ:
		{
			int mx = (p.x - floor(p.x)) < 0.5 ? 0 : 1;
			int my = 0; // (space.Y - floor(space.Y)) < 0.5 ? 0 : 1;
			int mz = (p.z - floor(p.z)) < 0.5 ? 0 : 1;
			double c = (mx + my + mz) % 2;
			color_diffuse = make_double4(c, c, c, 0);
			color_specular = make_double4(1, 1, 1, 0);
			scale_reflect = 1;
			scale_refract = 0;
		}
		break;
		case MATERIAL_RED:
		{
			//color_diffuse = make_double4(1, 0, 0, 0);
			color_diffuse = make_double4(0, 0, 0, 0);
			color_specular = make_double4(1, 1, 1, 0);
			scale_reflect = 0;
			scale_refract = 1;
		}
		break;
		case MATERIAL_GREEN:
		{
			color_diffuse = make_double4(0, 0.5, 0, 0);
			color_specular = make_double4(1, 1, 1, 0);
			scale_reflect = 0.5;
			scale_refract = 0;
		}
		break;
		case MATERIAL_BLUE:
		{
			color_diffuse = make_double4(0, 0, 1, 0);
			color_specular = make_double4(1, 1, 1, 0);
			scale_reflect = 0.5;
			scale_refract = 0;
		}
		break;
		case MATERIAL_GLASS:
		{
			color_diffuse = make_double4(0, 0, 0, 0);
			color_specular = make_double4(1, 1, 1, 0);
			scale_reflect = 0;
			scale_refract = 1;
		}
		break;
		}
		double4 color = color_diffuse * scale_diffuse + color_specular * scale_specular;
		if (scale_refract > 0) {
			//Schlick's Approximation for reflectance.
			double R = SchlickApprox(i, n, 1, 1.5);
			scale_reflect = R;
			//Add refraction contribution.
			color = color + RayColor<RECURSE - 1>(p + t * 0.0001, t) * scale_refract;
		}
		if (scale_reflect > 0) {
			color = color + RayColor<RECURSE - 1>(p + r * 0.0001, r) * scale_reflect;
		}
		color.w = 1;
		return color;
	}
}

template <>
__device__ double4 RayColor<-1>(const double3 &origin, const double3 &direction) {
	return make_double4(0, 0, 0, 0);
}

__device__ double4 RayColor(const double3 &origin, const double3 &direction) {
	return RayColor<2>(origin, direction);
}

__device__ void ComputeRay(const Matrix4D &inverse_mvp, double clipx, double clipy, double3 &origin, double3 &direction) {
	double4 v41 = Transform(inverse_mvp, make_double4(clipx, clipy, 0, 1));
	double4 v42 = Transform(inverse_mvp, make_double4(clipx, clipy, 1, 1));
	double3 ray_p1 = make_double3(v41.x / v41.w, v41.y / v41.w, v41.z / v41.w);
	double3 ray_p2 = make_double3(v42.x / v42.w, v42.y / v42.w, v42.z / v42.w);
	origin = ray_p1;
	direction = ray_p2 - ray_p1;
}

__device__ unsigned int MakePixel(const double4 &color) {
	unsigned char r = color.x < 0 ? 0 : (color.x > 1 ? 1 : color.x) * 255;
	unsigned char g = color.y < 0 ? 0 : (color.y > 1 ? 1 : color.y) * 255;
	unsigned char b = color.z < 0 ? 0 : (color.z > 1 ? 1 : color.z) * 255;
	unsigned char a = color.w < 0 ? 0 : (color.w > 1 ? 1 : color.w) * 255;
	return (a << 24) | (r << 16) | (g << 8) | (b << 0);
}

__global__ void cudaRaytraceKernel(Matrix4D inverse_mvp, void *bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
{
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	// Generate untransformed ray.
	double3 origin;
	double3 direction;
	double4 color = make_double4(0, 0, 0, 0);
	const int X_SUPERSAMPLES = 1;
	const int Y_SUPERSAMPLES = 1;
	for (int y_supersample = 1; y_supersample <= Y_SUPERSAMPLES; ++y_supersample) {
		for (int x_supersample = 1; x_supersample <= X_SUPERSAMPLES; ++x_supersample) {
			// Build a ray for this supersample.
			double vx = Lerp(-1, +1, (x + x_supersample / (X_SUPERSAMPLES + 1.0)) / bitmap_width);
			double vy = Lerp(+1, -1, (y + y_supersample / (Y_SUPERSAMPLES + 1.0)) / bitmap_height);
			ComputeRay(inverse_mvp, vx, vy, origin, direction);
			// Compute intersection with plane.
			color = color + RayColor(origin, direction);
		}
	}
	color = color / (X_SUPERSAMPLES * Y_SUPERSAMPLES);
	// Fill in the pixel.
	void *pRaster = (unsigned char*)bitmap_ptr + bitmap_stride * y;
	void *pPixel = (unsigned char*)pRaster + 4 * x;
	*(unsigned int*)pPixel = MakePixel(color);
}

////////////////////////////////////////////////////////////////////////////////
// Host Code
////////////////////////////////////////////////////////////////////////////////

void CUDA_CALL(cudaError_t error) {
	if (error == 0) return;
	int test = 0;
}

#define TRY_CUDA(fn) CUDA_CALL(fn);

extern "C" bool cudaRaytrace(double* pMVP, void *bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
{
	void *device_buffer = nullptr;
	int buffer_stride = 4 * bitmap_width;
	TRY_CUDA(cudaMalloc((void **)&device_buffer, buffer_stride * bitmap_height));
	Matrix4D MVP = *(Matrix4D*)pMVP;
	dim3 grid(bitmap_width / 16, bitmap_height / 16, 1);
	dim3 threads(16, 16, 1);
	cudaRaytraceKernel<<<grid, threads>>>(MVP, device_buffer, bitmap_width, bitmap_height, 4 * bitmap_width);
	for (int y = 0; y < bitmap_height; ++y)
	{
		void* pDevice = (unsigned char*)device_buffer + buffer_stride * y;
		void* pHost = (unsigned char*)bitmap_ptr + bitmap_stride * y;
		TRY_CUDA(cudaMemcpy(pHost, pDevice, 4 * bitmap_width, cudaMemcpyDeviceToHost));
		int test = 0;
	}
	TRY_CUDA(cudaFree(device_buffer));
	device_buffer = nullptr;
	return true;
}
