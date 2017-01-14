#include <cuda_runtime.h>
#include <math.h>

void CUDA_CALL(cudaError_t error) {
	if (error == 0) return;
	int test = 0;
}

#define TRY_CUDA(fn) CUDA_CALL(fn);

struct Point3D { double X, Y, Z; };
struct Point4D { double X, Y, Z, W; };
struct Vector3D { double X, Y, Z; };
struct Matrix4D { double M[16]; };

__device__ Point3D CreatePoint3D(double x, double y, double z) { Point3D r; r.X = x; r.Y = y; r.Z = z; return r; }
__device__ Point4D CreatePoint4D(double x, double y, double z, double w) { Point4D r; r.X = x; r.Y = y; r.Z = z; r.W = w; return r; }
__device__ Vector3D CreateVector3D(double x, double y, double z) { Vector3D r; r.X = x; r.Y = y; r.Z = z; return r; }
__device__ Point3D Add(const Point3D &lhs, const Point3D &rhs) { Point3D r; r.X = lhs.X + rhs.X; r.Y = lhs.Y + rhs.Y; r.Z = lhs.Z + rhs.Z; return r; }
__device__ Point4D Add(const Point4D &lhs, const Point4D &rhs) { Point4D r; r.X = lhs.X + rhs.X; r.Y = lhs.Y + rhs.Y; r.Z = lhs.Z + rhs.Z; r.W = lhs.W + rhs.W; return r; }
__device__ Point3D Add(const Point3D &lhs, const Vector3D &rhs) { Point3D r; r.X = lhs.X + rhs.X; r.Y = lhs.Y + rhs.Y; r.Z = lhs.Z + rhs.Z; return r; }
__device__ Vector3D Add(const Vector3D &lhs, const Vector3D &rhs) { Vector3D r; r.X = lhs.X + rhs.X; r.Y = lhs.Y + rhs.Y; r.Z = lhs.Z + rhs.Z; return r; }
__device__ double Dot(const Point3D& lhs, const Point3D& rhs) { return lhs.X * rhs.X + lhs.Y * rhs.Y + lhs.Z * rhs.Z; }
__device__ double Dot(const Point3D& lhs, const Vector3D& rhs) { return lhs.X * rhs.X + lhs.Y * rhs.Y + lhs.Z * rhs.Z; }
__device__ double Dot(const Vector3D& lhs, const Point3D& rhs) { return lhs.X * rhs.X + lhs.Y * rhs.Y + lhs.Z * rhs.Z; }
__device__ double Dot(const Vector3D& lhs, const Vector3D& rhs) { return lhs.X * rhs.X + lhs.Y * rhs.Y + lhs.Z * rhs.Z; }
__device__ double Length(const Point3D& val) { return sqrt(Dot(val, val)); }
__device__ double Length(const Vector3D& val) { return sqrt(Dot(val, val)); }
__device__ Point3D Multiply(const Point3D& lhs, double rhs) { Point3D r; r.X = lhs.X * rhs; r.Y = lhs.Y * rhs; r.Z = lhs.Z * rhs; return r; }
__device__ Point4D Multiply(const Point4D& lhs, double rhs) { Point4D r; r.X = lhs.X * rhs; r.Y = lhs.Y * rhs; r.Z = lhs.Z * rhs; r.W = lhs.W * rhs; return r; }
__device__ Vector3D Multiply(const Vector3D& lhs, double rhs) { Vector3D r; r.X = lhs.X * rhs; r.Y = lhs.Y * rhs; r.Z = lhs.Z * rhs; return r; }
__device__ Point3D Negate(const Point3D& val) { Point3D r; r.X = -val.X; r.Y = -val.Y; r.Z = -val.Z; return r; }
__device__ Point4D Negate(const Point4D& val) { Point4D r; r.X = -val.X; r.Y = -val.Y; r.Z = -val.Z; r.W = -val.W; r; }
__device__ Vector3D Negate(const Vector3D& val) { Vector3D r; r.X = -val.X; r.Y = -val.Y; r.Z = -val.Z; return r; }
__device__ Point3D Normalize(const Point3D &val) { return Multiply(val, 1.0 / Length(val)); }
__device__ Vector3D Normalize(const Vector3D &val) { return Multiply(val, 1.0 / Length(val)); }
__device__ Vector3D Subtract(const Point3D& lhs, const Point3D& rhs) { Vector3D r; r.X = lhs.X - rhs.X; r.Y = lhs.Y - rhs.Y; r.Z = lhs.Z - rhs.Z; return r; }
__device__ Point3D Transform(const Matrix4D& m, const Point3D& p) {
	Point3D r;
	r.X = m.M[ 0] * p.X + m.M[ 4] * p.Y + m.M[ 8] * p.Z + m.M[12];
	r.Y = m.M[ 1] * p.X + m.M[ 5] * p.Y + m.M[ 9] * p.Z + m.M[13];
	r.Z = m.M[ 2] * p.X + m.M[ 6] * p.Y + m.M[10] * p.Z + m.M[14];
	return r;
}
__device__ Point4D Transform(const Matrix4D& m, const Point4D& p) {
	Point4D r;
	r.X = m.M[ 0] * p.X + m.M[ 4] * p.Y + m.M[ 8] * p.Z + m.M[12] * p.W;
	r.Y = m.M[ 1] * p.X + m.M[ 5] * p.Y + m.M[ 9] * p.Z + m.M[13] * p.W;
	r.Z = m.M[ 2] * p.X + m.M[ 6] * p.Y + m.M[10] * p.Z + m.M[14] * p.W;
	r.W = m.M[ 3] * p.X + m.M[ 7] * p.Y + m.M[11] * p.Z + m.M[15] * p.W;
	return r;
}
__device__ Vector3D Transform(const Matrix4D& m, const Vector3D& p) {
	Vector3D r;
	r.X = m.M[ 0] * p.X + m.M[ 4] * p.Y + m.M[ 8] * p.Z;
	r.Y = m.M[ 1] * p.X + m.M[ 5] * p.Y + m.M[ 9] * p.Z;
	r.Z = m.M[ 2] * p.X + m.M[ 6] * p.Y + m.M[10] * p.Z;
	return r;
}

__device__ Point3D operator-(const Point3D& val) { return Negate(val); }
__device__ Point4D operator-(const Point4D& val) { return Negate(val); }
__device__ Vector3D operator-(const Vector3D& val) { return Negate(val); }
__device__ Point3D operator+(const Point3D& lhs, const Point3D& rhs) { return Add(lhs, rhs); }
__device__ Point3D operator+(const Point3D& lhs, const Vector3D& rhs) { return Add(lhs, rhs); }
__device__ Point4D operator+(const Point4D& lhs, const Point4D& rhs) { return Add(lhs, rhs); }
__device__ Vector3D operator+(const Vector3D& lhs, const Vector3D& rhs) { return Add(lhs, rhs); }
__device__ Vector3D operator-(const Point3D& lhs, const Point3D& rhs) { return Subtract(lhs, rhs); }
__device__ Point3D operator*(const Point3D& lhs, double rhs) { return Multiply(lhs, rhs); }
__device__ Point3D operator*(double lhs, const Point3D& rhs) { return Multiply(rhs, lhs); }
__device__ Point4D operator*(const Point4D& lhs, double rhs) { return Multiply(lhs, rhs); }
__device__ Point4D operator*(double lhs, const Point4D& rhs) { return Multiply(rhs, lhs); }
__device__ Vector3D operator*(const Vector3D& lhs, double rhs) { return Multiply(lhs, rhs); }
__device__ Vector3D operator*(double lhs, const Vector3D& rhs) { return Multiply(rhs, lhs); }
__device__ Point3D operator/(const Point3D &lhs, double rhs) { return Multiply(lhs, 1.0 / rhs); }
__device__ Point4D operator/(const Point4D &lhs, double rhs) { return Multiply(lhs, 1.0 / rhs); }
__device__ Vector3D operator/(const Vector3D &lhs, double rhs) { return Multiply(lhs, 1.0 / rhs); }

__device__ double IntersectPlane(const Point3D &origin, const Vector3D &direction) {
	const double PLANE_DISTANCE = 0;
	const Vector3D PLANE_NORMAL = CreateVector3D(0, 1, 0);
	return (PLANE_DISTANCE - Dot(PLANE_NORMAL, origin)) / Dot(PLANE_NORMAL, direction);
}

__device__ double IntersectSphere(const Point3D &origin, const Vector3D &direction) {
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

__device__ bool RayShadow(const Point3D &origin, const Vector3D &direction) {
	// Start intersecting objects.
	double best_distance = 1000000;
	double distance;
	// Intersect with the ground plane
	distance = IntersectPlane(origin, direction);
	if (distance >= 0 && distance < best_distance) return true;
	// Intersect with the red sphere.
	{
		double xfrm_inv[] = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 2, -1 ,0, 1 };
		Matrix4D *matxfrm_inv = (Matrix4D*)xfrm_inv;
		Point3D origin2 = Transform(*matxfrm_inv, origin);
		Vector3D direction2 = Transform(*matxfrm_inv, direction);
		distance = IntersectSphere(origin2, direction2);
		if (distance >= 0 && distance < best_distance) return true;
	}
	// Intersect with the green sphere.
	{
		double xfrm_inv[] = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, -1 ,0, 1 };
		Matrix4D *matxfrm_inv = (Matrix4D*)xfrm_inv;
		Point3D origin2 = Transform(*matxfrm_inv, origin);
		Vector3D direction2 = Transform(*matxfrm_inv, direction);
		distance = IntersectSphere(origin2, direction2);
		if (distance >= 0 && distance < best_distance) return true;
	}
	// Intersect with the blue sphere.
	{
		double xfrm_inv[] = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -2, -1 ,0, 1 };
		Matrix4D *matxfrm_inv = (Matrix4D*)xfrm_inv;
		Point3D origin2 = Transform(*matxfrm_inv, origin);
		Vector3D direction2 = Transform(*matxfrm_inv, direction);
		distance = IntersectSphere(origin2, direction2);
		if (distance >= 0 && distance < best_distance) return true;
	}
	return false;
}

__device__ Point4D RayColor(const Point3D &origin, const Vector3D &direction, int recurse) {
	Point4D color = CreatePoint4D(0, 0, 0, 0);
	// Start intersecting objects.
	double best_distance = 1000000;
	int best_object = 0;
	double distance;
	// Intersect with the ground plane
	distance = IntersectPlane(origin, direction);
	if (distance >= 0 && distance < best_distance) {
		best_distance = distance;
		best_object = 1;
	}
	// Intersect with the red sphere.
	{
		double xfrm_inv[] = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 2, -1 ,0, 1 };
		Matrix4D *matxfrm_inv = (Matrix4D*)xfrm_inv;
		Point3D origin2 = Transform(*matxfrm_inv, origin);
		Vector3D direction2 = Transform(*matxfrm_inv, direction);
		distance = IntersectSphere(origin2, direction2);
		if (distance >= 0 && distance < best_distance) {
			best_distance = distance;
			best_object = 2;
		}
	}
	// Intersect with the green sphere.
	{
		double xfrm_inv[] = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, -1 ,0, 1 };
		Matrix4D *matxfrm_inv = (Matrix4D*)xfrm_inv;
		Point3D origin2 = Transform(*matxfrm_inv, origin);
		Vector3D direction2 = Transform(*matxfrm_inv, direction);
		distance = IntersectSphere(origin2, direction2);
		if (distance >= 0 && distance < best_distance) {
			best_distance = distance;
			best_object = 3;
		}
	}
	// Intersect with the blue sphere.
	{
		double xfrm_inv[] = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -2, -1 ,0, 1 };
		Matrix4D *matxfrm_inv = (Matrix4D*)xfrm_inv;
		Point3D origin2 = Transform(*matxfrm_inv, origin);
		Vector3D direction2 = Transform(*matxfrm_inv, direction);
		distance = IntersectSphere(origin2, direction2);
		if (distance >= 0 && distance < best_distance) {
			best_distance = distance;
			best_object = 4;
		}
	}
	// Best was the ground plane.
	if (best_object == 1) {
		Point3D p = origin + best_distance * direction;
		Vector3D l = Normalize(CreatePoint3D(10, 10, -10) - p);
		Vector3D n = CreateVector3D(0, 1, 0);
		Vector3D v = Normalize(-direction);
		Vector3D r = -v + 2 * Dot(n, v) * n;
		double lambert = Dot(l, n);
		lambert = lambert > 0 ? lambert : 0;
		double specular = Dot(l, r);
		specular = specular > 0 ? specular : 0;
		specular = pow(specular, 10);
		if (RayShadow(p + 0.0001 * n, l)) lambert *= 0.5;
		int mx = (p.X - floor(p.X)) < 0.5 ? 0 : 1;
		int my = 0; // (space.Y - floor(space.Y)) < 0.5 ? 0 : 1;
		int mz = (p.Z - floor(p.Z)) < 0.5 ? 0 : 1;
		int mod = (mx + my + mz) % 2;
		color.X = color.Y = color.Z = lambert * mod + specular;
		if (recurse > 0) {
			color = color + RayColor(p + n * 0.0001, r, recurse - 1) * 0.5;
		}
		color.W = 1;
	}
	// Best was the red sphere.
	if (best_object == 2) {
		Point3D p = origin + best_distance * direction;
		Vector3D l = Normalize(CreatePoint3D(10, 10, -10) - p);
		Vector3D n = Normalize(p - CreatePoint3D(-2, 1, 0));
		Vector3D v = Normalize(-direction);
		Vector3D r = -v + 2 * Dot(n, v) * n;
		double scale_diffuse = Dot(l, n);
		scale_diffuse = scale_diffuse > 0 ? scale_diffuse : 0;
		double scale_specular = Dot(l, r);
		scale_specular = scale_specular > 0 ? scale_specular : 0;
		scale_specular = pow(scale_specular, 10);
		Point4D color_diffuse = CreatePoint4D(1, 0, 0, 0);
		Point4D color_specular = CreatePoint4D(1, 1, 1, 0);
		color = color_diffuse * scale_diffuse + color_specular * scale_specular;
		if (recurse > 0) {
			color = color + RayColor(p + n * 0.0001, r, recurse - 1) * 0.5;
		}
		color.W = 1;
	}
	// Best was the green sphere.
	if (best_object == 3) {
		Point3D p = origin + best_distance * direction;
		Vector3D l = Normalize(CreatePoint3D(10, 10, -10) - p);
		Vector3D n = Normalize(p - CreatePoint3D(0, 1, 0));
		Vector3D v = Normalize(-direction);
		Vector3D r = -v + 2 * Dot(n, v) * n;
		double scale_diffuse = Dot(l, n);
		scale_diffuse = scale_diffuse > 0 ? scale_diffuse : 0;
		double scale_specular = Dot(l, r);
		scale_specular = scale_specular > 0 ? scale_specular : 0;
		scale_specular = pow(scale_specular, 10);
		Point4D color_diffuse = CreatePoint4D(0, 0.5, 0, 0);
		Point4D color_specular = CreatePoint4D(1, 1, 1, 0);
		color = color_diffuse * scale_diffuse + color_specular * scale_specular;
		if (recurse > 0) {
			color = color + RayColor(p + n * 0.0001, r, recurse - 1) * 0.5;
		}
		color.W = 1;
	}
	// Best was the blue sphere.
	if (best_object == 4) {
		Point3D p = origin + best_distance * direction;
		Vector3D l = Normalize(CreatePoint3D(10, 10, -10) - p);
		Vector3D n = Normalize(p - CreatePoint3D(2, 1, 0));
		Vector3D v = Normalize(-direction);
		Vector3D r = -v + 2 * Dot(n, v) * n;
		double scale_diffuse = Dot(l, n);
		scale_diffuse = scale_diffuse > 0 ? scale_diffuse : 0;
		double scale_specular = Dot(l, r);
		scale_specular = scale_specular > 0 ? scale_specular : 0;
		scale_specular = pow(scale_specular, 10);
		Point4D color_diffuse = CreatePoint4D(0, 0, 1, 0);
		Point4D color_specular = CreatePoint4D(1, 1, 1, 0);
		color = color_diffuse * scale_diffuse + color_specular * scale_specular;
		if (recurse > 0) {
			color = color + RayColor(p + n * 0.0001, r, recurse - 1) * 0.5;
		}
		color.W = 1;
	}
	return color;
}

__device__ Point4D RayColor(const Point3D &origin, const Vector3D &direction) {
	return RayColor(origin, direction, 2);
}

__device__ unsigned int Point4DToUint(const Point4D &color) {
	unsigned char r = color.X < 0 ? 0 : (color.X > 1 ? 1 : color.X) * 255;
	unsigned char g = color.Y < 0 ? 0 : (color.Y > 1 ? 1 : color.Y) * 255;
	unsigned char b = color.Z < 0 ? 0 : (color.Z > 1 ? 1 : color.Z) * 255;
	unsigned char a = color.W < 0 ? 0 : (color.W > 1 ? 1 : color.W) * 255;
	return (a << 24) | (r << 16) | (g << 8) | (b << 0);
}

__device__ void ComputeRay(const Matrix4D &inverse_mvp, double clipx, double clipy, Point3D &origin, Vector3D &direction) {
	Point4D v41 = Transform(inverse_mvp, CreatePoint4D(clipx, clipy, 0, 1));
	Point4D v42 = Transform(inverse_mvp, CreatePoint4D(clipx, clipy, 1, 1));
	Point3D ray_p1 = CreatePoint3D(v41.X / v41.W, v41.Y / v41.W, v41.Z / v41.W);
	Point3D ray_p2 = CreatePoint3D(v42.X / v42.W, v42.Y / v42.W, v42.Z / v42.W);
	origin = ray_p1;
	direction = ray_p2 - ray_p1;
}

__global__ void cudaRaytraceKernel(Matrix4D inverse_mvp, void *bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
{
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	// Generate untransformed ray.
	Point3D origin;
	Vector3D direction;
	Point4D color = CreatePoint4D(0, 0, 0, 0);
	const int SAMPLES_X = 2;
	const int SAMPLES_Y = 2;
	for (int aa_y = 0; aa_y < SAMPLES_Y; ++aa_y) {
		for (int aa_x = 0; aa_x < SAMPLES_X; ++aa_x) {
			double dx = (aa_x + 0.5) / SAMPLES_X;
			double dy = (aa_y + 0.5) / SAMPLES_Y;
			double vx = -1.0 + ((x * 2.0) + dx) / bitmap_width;
			double vy = 1.0 - ((y * 2.0) + dy) / bitmap_height;
			ComputeRay(inverse_mvp, vx, vy, origin, direction);
			// Compute intersection with plane.
			color = Add(color, RayColor(origin, direction));
		}
	}
	color = color / (SAMPLES_X * SAMPLES_Y);
	// Fill in the pixel.
	void *pRaster = (unsigned char*)bitmap_ptr + bitmap_stride * y;
	void *pPixel = (unsigned char*)pRaster + 4 * x;
	*(unsigned int*)pPixel = Point4DToUint(color);
}

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
