//==================================================================================================
// Written in 2016 by Peter Shirley <ptrshrl@gmail.com>
//
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is distributed
// without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication along
// with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//==================================================================================================

#include <iostream>
#include <cfloat>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <iomanip>
#include <chrono>
#include <string>
#include <algorithm>
#include <memory>
#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <exception>
#include <CL/sycl.hpp>

class timer
{
public:
	timer() = default;
	void start(const std::string& text_)
	{
		text = text_;
		begin = std::chrono::high_resolution_clock::now();
	}
	void stop()
	{
		auto end = std::chrono::high_resolution_clock::now();
		auto dur = end - begin;
		auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
		std::cout << std::setw(20) << text << " timing:" << std::setw(5) << ms << "ms" << std::endl;
	}

private:
	std::string text;
	std::chrono::high_resolution_clock::time_point begin;
};

class RandomNumGen
{
public:
	RandomNumGen()
		: gen(rd())
		, dis(0.0, 1.0)
	{
	}
	static double GetRand()
	{
		thread_local std::unique_ptr<RandomNumGen> randomNumGen;
		if (!randomNumGen)
			randomNumGen = std::unique_ptr<RandomNumGen>(new RandomNumGen());
		return randomNumGen->dis(randomNumGen->gen);
	}

	std::random_device rd;  //Will be used to obtain a seed for the random number engine
	std::mt19937 gen; //Standard mersenne_twister_engine seeded with rd()
	std::uniform_real_distribution<> dis;

};

class PreGenerated
{
public:
	PreGenerated(size_t size)
	{
		std::random_device rd;  //Will be used to obtain a seed for the random number engine
		std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
		std::uniform_real_distribution<> dis(0.0, 1.0);
		vec.resize(size);
		for (size_t i = 0; i < size; ++i)
			vec[i] = dis(gen);
	}
	const std::vector<float>& GetVector() const
	{
		return vec;
	}
private:
	std::vector<float> vec;
};

class RandAccessor
{
public:
	 RandAccessor(size_t offset_, const float* arr_, size_t size_)
		: offset(offset_)
		, arr(arr_)
		, size(size_)
		, index(0)
	{}
	 float Get() const
	{
		size_t i = (offset + index) % size;
		++index;
		return arr[i];
	}
private:
	size_t offset;
	const float* arr;
	size_t size;
	mutable size_t index;
};

class vec3 {

public:
	  vec3() {}
	  vec3(float e0, float e1, float e2) { e[0] = e0; e[1] = e1; e[2] = e2; }
	  inline float x() const { return e[0]; }
	  inline float y() const { return e[1]; }
	  inline float z() const { return e[2]; }
	  inline float r() const { return e[0]; }
	  inline float g() const { return e[1]; }
	  inline float b() const { return e[2]; }

	  inline const vec3& operator+() const { return *this; }
	  inline vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
	  inline float operator[](int i) const { return e[i]; }
	  inline float& operator[](int i) { return e[i]; }

	  inline vec3& operator+=(const vec3& v2);
	  inline vec3& operator-=(const vec3& v2);
	  inline vec3& operator*=(const vec3& v2);
	  inline vec3& operator/=(const vec3& v2);
	  inline vec3& operator*=(const float t);
	  inline vec3& operator/=(const float t);

	  inline float length() const { return sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]); }
	  inline float squared_length() const { return e[0] * e[0] + e[1] * e[1] + e[2] * e[2]; }
	  inline void make_unit_vector();

	float e[3];
};



inline std::istream& operator>>(std::istream& is, vec3& t) {
	is >> t.e[0] >> t.e[1] >> t.e[2];
	return is;
}

inline std::ostream& operator<<(std::ostream& os, const vec3& t) {
	os << t.e[0] << " " << t.e[1] << " " << t.e[2];
	return os;
}

inline void vec3::make_unit_vector() {
	float k = 1.0 / sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
	e[0] *= k; e[1] *= k; e[2] *= k;
}

inline vec3 operator+(const vec3& v1, const vec3& v2) {
	return vec3(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]);
}

inline vec3 operator-(const vec3& v1, const vec3& v2) {
	return vec3(v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2]);
}

inline vec3 operator*(const vec3& v1, const vec3& v2) {
	return vec3(v1.e[0] * v2.e[0], v1.e[1] * v2.e[1], v1.e[2] * v2.e[2]);
}

inline vec3 operator/(const vec3& v1, const vec3& v2) {
	return vec3(v1.e[0] / v2.e[0], v1.e[1] / v2.e[1], v1.e[2] / v2.e[2]);
}

inline vec3 operator*(float t, const vec3& v) {
	return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

inline vec3 operator/(vec3 v, float t) {
	return vec3(v.e[0] / t, v.e[1] / t, v.e[2] / t);
}

inline vec3 operator*(const vec3& v, float t) {
	return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

inline float dot(const vec3& v1, const vec3& v2) {
	return v1.e[0] * v2.e[0] + v1.e[1] * v2.e[1] + v1.e[2] * v2.e[2];
}

inline vec3 cross(const vec3& v1, const vec3& v2) {
	return vec3((v1.e[1] * v2.e[2] - v1.e[2] * v2.e[1]),
		(-(v1.e[0] * v2.e[2] - v1.e[2] * v2.e[0])),
		(v1.e[0] * v2.e[1] - v1.e[1] * v2.e[0]));
}

inline vec3& vec3::operator+=(const vec3& v) {
	e[0] += v.e[0];
	e[1] += v.e[1];
	e[2] += v.e[2];
	return *this;
}

inline vec3& vec3::operator*=(const vec3& v) {
	e[0] *= v.e[0];
	e[1] *= v.e[1];
	e[2] *= v.e[2];
	return *this;
}

inline vec3& vec3::operator/=(const vec3& v) {
	e[0] /= v.e[0];
	e[1] /= v.e[1];
	e[2] /= v.e[2];
	return *this;
}

inline vec3& vec3::operator-=(const vec3& v) {
	e[0] -= v.e[0];
	e[1] -= v.e[1];
	e[2] -= v.e[2];
	return *this;
}

inline vec3& vec3::operator*=(const float t) {
	e[0] *= t;
	e[1] *= t;
	e[2] *= t;
	return *this;
}

inline vec3& vec3::operator/=(const float t) {
	float k = 1.0 / t;

	e[0] *= k;
	e[1] *= k;
	e[2] *= k;
	return *this;
}

inline vec3 unit_vector(vec3 v) {
	return v / v.length();
}

class ray
{
public:
	  ray() {}
	  ray(const vec3& a, const vec3& b) { A = a; B = b; }
	  vec3 origin() const { return A; }
	  vec3 direction() const { return B; }
	  vec3 point_at_parameter(float t) const { return A + t * B; }

	vec3 A;
	vec3 B;
};

vec3 random_in_unit_disk(const RandAccessor& rand) {
	vec3 p;
	do {
		p = 2.0 * vec3(rand.Get(), rand.Get(), 0) - vec3(1, 1, 0);
	} while (dot(p, p) >= 1.0);
	return p;
}

class camera {
public:
	camera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect, float aperture, float focus_dist) { // vfov is top to bottom in degrees
		lens_radius = aperture / 2;
		float theta = vfov * M_PI / 180;
		float half_height = tan(theta / 2);
		float half_width = aspect * half_height;
		origin = lookfrom;
		w = unit_vector(lookfrom - lookat);
		u = unit_vector(cross(vup, w));
		v = cross(w, u);
		lower_left_corner = origin - half_width * focus_dist * u - half_height * focus_dist * v - focus_dist * w;
		horizontal = 2 * half_width * focus_dist * u;
		vertical = 2 * half_height * focus_dist * v;
	}
	ray get_ray(float s, float t, const RandAccessor& rand) {
		vec3 rd = lens_radius * random_in_unit_disk(rand);
		vec3 offset = u * rd.x() + v * rd.y();
		return ray(origin + offset, lower_left_corner + s * horizontal + t * vertical - origin - offset);
	}

	vec3 origin;
	vec3 lower_left_corner;
	vec3 horizontal;
	vec3 vertical;
	vec3 u, v, w;
	float lens_radius;
};

struct hit_record;

float schlick(float cosine, float ref_idx) {
	float r0 = (1 - ref_idx) / (1 + ref_idx);
	r0 = r0 * r0;
	return r0 + (1 - r0) * powf((1 - cosine), 5);
}

bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted) {
	vec3 uv = unit_vector(v);
	float dt = dot(uv, n);
	float discriminant = 1.0 - ni_over_nt * ni_over_nt * (1 - dt * dt);
	if (discriminant > 0) {
		refracted = ni_over_nt * (uv - n * dt) - n * sqrt(discriminant);
		return true;
	}
	else
		return false;
}


vec3 reflect(const vec3& v, const vec3& n) {
	return v - 2 * dot(v, n) * n;
}


vec3 random_in_unit_sphere(const RandAccessor& rand) {
	vec3 p;
	do {
		p = 2.0 * vec3(rand.Get(), rand.Get(), rand.Get()) - vec3(1, 1, 1);
	} while (p.squared_length() >= 1.0);
	return p;
}

class material;

struct hit_record
{
	float t;
	vec3 p;
	vec3 normal;
	const material* mat;
};

enum class material_type
{
	lambertian,
	metal,
	dielectric
};

class material {
public:
	material() = default;
	material(material_type mtype, const vec3& a, float f, float ri) : mat_type(mtype), albedo(a), ref_idx(ri) {
		if (f < 1)
			fuzz = f;
		else
			fuzz = 1;
	}
	bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, const RandAccessor& rand) const
	{
		switch (mat_type)
		{
		case material_type::lambertian:
			return lambertian_scatter(r_in, rec, attenuation, scattered, rand);
		case material_type::metal:
			return metal_scatter(r_in, rec, attenuation, scattered, rand);
		case material_type::dielectric:
			return dielectric_scatter(r_in, rec, attenuation, scattered, rand);
		}
		return lambertian_scatter(r_in, rec, attenuation, scattered, rand);
	}

	bool lambertian_scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, const RandAccessor& rand) const {
		vec3 target = rec.p + rec.normal + random_in_unit_sphere(rand);
		scattered = ray(rec.p, target - rec.p);
		attenuation = albedo;
		return true;
	}
	bool metal_scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, const RandAccessor& rand) const {
		vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
		scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere(rand));
		attenuation = albedo;
		return (dot(scattered.direction(), rec.normal) > 0);
	}
	bool dielectric_scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, const RandAccessor& rand) const {
		vec3 outward_normal;
		vec3 reflected = reflect(r_in.direction(), rec.normal);
		float ni_over_nt;
		attenuation = vec3(1.0, 1.0, 1.0);
		vec3 refracted;
		float reflect_prob;
		float cosine;
		if (dot(r_in.direction(), rec.normal) > 0) {
			outward_normal = -rec.normal;
			ni_over_nt = ref_idx;
			//         cosine = ref_idx * dot(r_in.direction(), rec.normal) / r_in.direction().length();
			cosine = dot(r_in.direction(), rec.normal) / r_in.direction().length();
			cosine = sqrt(1 - ref_idx * ref_idx * (1 - cosine * cosine));
		}
		else {
			outward_normal = rec.normal;
			ni_over_nt = 1.0 / ref_idx;
			cosine = -dot(r_in.direction(), rec.normal) / r_in.direction().length();
		}
		if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted))
			reflect_prob = schlick(cosine, ref_idx);
		else
			reflect_prob = 1.0;
		if (rand.Get() < reflect_prob)
			scattered = ray(rec.p, reflected);
		else
			scattered = ray(rec.p, refracted);
		return true;
	}

	material_type mat_type;
	vec3 albedo;
	float fuzz;
	float ref_idx;

};

class sphere {
public:
	sphere() = default;
	sphere(vec3 cen, float r, material m) : center(cen), radius(r), mat(m) {};
	bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
	vec3 center;
	float radius;
	material mat;
};

bool sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
	vec3 oc = r.origin() - center;
	float a = dot(r.direction(), r.direction());
	float b = dot(oc, r.direction());
	float c = dot(oc, oc) - radius * radius;
	float discriminant = b * b - a * c;
	if (discriminant > 0) {
		float temp = (-b - sqrt(discriminant)) / a;
		if (temp < t_max && temp > t_min) {
			rec.t = temp;
			rec.p = r.point_at_parameter(rec.t);
			rec.normal = (rec.p - center) / radius;
			rec.mat = &mat;
			return true;
		}
		temp = (-b + sqrt(discriminant)) / a;
		if (temp < t_max && temp > t_min) {
			rec.t = temp;
			rec.p = r.point_at_parameter(rec.t);
			rec.normal = (rec.p - center) / radius;
			rec.mat = &mat;
			return true;
		}
	}
	return false;
}

bool world_hit(sphere* world, size_t world_size, const ray& r, float tmin, float tmax, hit_record& rec) {
	hit_record temp_rec;
	bool hit_anything = false;
	double closest_so_far = tmax;
	for (int i = 0; i < world_size; i++) {
		if (world[i].hit(r, tmin, closest_so_far, temp_rec)) {
			hit_anything = true;
			closest_so_far = temp_rec.t;
			rec = temp_rec;
		}
	}
	return hit_anything;
}

/*
 vec3 color(const ray& r, sphere* world, size_t world_size, int depth, const RandAccessor& rand) {
	hit_record rec;
	if (world_hit(world, world_size, r, 0.001, FLT_MAX, rec)) {
		ray scattered;
		vec3 attenuation;
		if (depth < 50 && rec.mat->scatter(r, rec, attenuation, scattered, rand)) {
			return attenuation * color(scattered, world, world_size, depth + 1, rand);
		}
		else {
			return vec3(0, 0, 0);
		}
	}
	else {
		vec3 unit_direction = unit_vector(r.direction());
		float t = 0.5 * (unit_direction.y() + 1.0);
		return (1.0 - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
	}
}
*/

vec3 color_loop(const ray& r, sphere* world, size_t world_size, const RandAccessor& rand) {
	vec3 attenuation_result(1.0, 1.0, 1.0);
	bool assigned = false;
	ray temp = r;
	for (int i = 0; i < 50; ++i)
	{
		hit_record rec;
		if (world_hit(world, world_size, temp, 0.001, FLT_MAX, rec)) {
			ray scattered;
			vec3 attenuation;
			if (rec.mat->scatter(temp, rec, attenuation, scattered, rand)) {
				temp = scattered;
				if (assigned == false)
				{
					attenuation_result = attenuation;
					assigned = true;
				}
				else
					attenuation_result *= attenuation;
			}
			else
			{
				attenuation_result = vec3(0, 0, 0);
				break;
			}
		}
		else {
			vec3 unit_direction = unit_vector(temp.direction());
			float t = 0.5 * (unit_direction.y() + 1.0);
			attenuation_result *= (1.0 - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
			break;
		}
	}
	return attenuation_result;
}



std::vector<sphere>  random_scene() {
	std::vector<sphere> list;
	list.push_back(sphere(vec3(0, -1000, 0), 1000, material(material_type::lambertian, vec3(0.5, 0.5, 0.5), 0.0f, 0.0f)));
	for (int a = -11; a < 11; a++) {
		for (int b = -11; b < 11; b++) {
			float choose_mat = RandomNumGen::GetRand();
			vec3 center(a + 0.9 * RandomNumGen::GetRand(), 0.2, b + 0.9 * RandomNumGen::GetRand());
			if ((center - vec3(4, 0.2, 0)).length() > 0.9) {
				if (choose_mat < 0.8) {  // diffuse
					list.push_back(sphere(center, 0.2, material(material_type::lambertian, vec3(RandomNumGen::GetRand() * RandomNumGen::GetRand(), RandomNumGen::GetRand() * RandomNumGen::GetRand(), RandomNumGen::GetRand() * RandomNumGen::GetRand()), 0.0f, 0.0f)));
				}
				else if (choose_mat < 0.95) { // metal
					list.push_back(sphere(center, 0.2,
						material(material_type::metal, vec3(0.5 * (1 + RandomNumGen::GetRand()), 0.5 * (1 + RandomNumGen::GetRand()), 0.5 * (1 + RandomNumGen::GetRand())), 0.5 * RandomNumGen::GetRand(), 0.0f)));
				}
				else {  // glass
					list.push_back(sphere(center, 0.2, material(material_type::dielectric, vec3(), 0.0f, 1.5)));
				}
			}
		}
	}

	list.push_back(sphere(vec3(0, 1, 0), 1.0, material(material_type::dielectric, vec3(), 0.0f, 1.5)));
	list.push_back(sphere(vec3(-4, 1, 0), 1.0, material(material_type::lambertian, vec3(0.4, 0.2, 0.1), 0.0f, 0.0f)));
	list.push_back(sphere(vec3(4, 1, 0), 1.0, material(material_type::metal, vec3(0.7, 0.6, 0.5), 0.0, 0.0)));

	return list;
}

int main() {
	int nx = 256;
	int ny = 256;
	int ns = 600;

	timer stopwatch;
	stopwatch.start("ray_tracer_init");

	std::vector<sphere> world;
	float R = cos(M_PI / 4);
	world.push_back(sphere(vec3(0, 0, -1), 0.5, material(material_type::lambertian, vec3(0.1, 0.2, 0.5), 0.0f, 0.0f)));
	world.push_back(sphere(vec3(0, -100.5, -1), 100, material(material_type::lambertian, vec3(0.8, 0.8, 0.0), 0.0f, 0.0f)));
	world.push_back(sphere(vec3(1, 0, -1), 0.5, material(material_type::metal, vec3(0.8, 0.6, 0.2), 0.0, 0.0f)));
	world.push_back(sphere(vec3(-1, 0, -1), 0.5, material(material_type::dielectric, vec3(), 0.0f, 1.5)));
	world.push_back(sphere(vec3(-1, 0, -1), -0.45, material(material_type::dielectric, vec3(), 0.0f, 1.5)));
	world = random_scene();

	vec3 lookfrom(13, 2, 3);
	vec3 lookat(0, 0, 0);
	float dist_to_focus = 10.0;
	float aperture = 0.1;

	camera cam(lookfrom, lookat, vec3(0, 1, 0), 20, float(nx) / float(ny), aperture, dist_to_focus);

	std::vector<unsigned int> pixelsSrc;
	pixelsSrc.resize(nx * ny);
	int stride = nx;
	for (int j = ny - 1; j >= 0; j--)
	{
		for (int i = 0; i < nx; i++)
		{
			int index = ((ny - 1) - j) * stride + i;

			pixelsSrc[index] = ((i & 0xffff) << 16) | (j & 0xffff);
		}
	}

	PreGenerated preGenerated(20000);
	std::vector<float> vec = preGenerated.GetVector();

	stopwatch.stop();

	size_t arr_size = vec.size();
	size_t sphere_size = world.size();


	stopwatch.start("ray_tracer");

	try {
		using namespace sycl;
		queue Q(host_selector{}); // for cpu without OpenCL installed
		//queue Q(gpu_selector{});
		buffer<float, 1> dev_arr{ vec.data(), range<1>(vec.size()) };
		buffer<sphere, 1> dev_sphere{ world.data(), range<1>(world.size()) };
		buffer<unsigned int, 1> dev_pixelsSrc{ pixelsSrc.data(), range<1>(pixelsSrc.size()) };
		buffer<camera, 1> dev_camera{ &cam, range<1>(1) };

		Q.submit([&](handler& h) {
			auto acc_dev_arr = dev_arr.template get_access<access::mode::read>(h);
			auto acc_dev_sphere = dev_sphere.template get_access<access::mode::read>(h);
			auto acc_dev_pixelsSrc =  dev_pixelsSrc.template get_access<access::mode::read_write>(h);
			auto acc_dev_camera = dev_camera.template get_access<access::mode::read>(h);

			h.parallel_for<class nstream>(range<1>{pixelsSrc.size()}, [=](id<1> index) {

				unsigned int& pixel = acc_dev_pixelsSrc[index];
				unsigned int* dev_pixel = &pixel;
				int j = (*dev_pixel) & 0xffff;
				int i = ((*dev_pixel) & 0xffff0000) >> 16;

				RandAccessor rand(i + j * nx, acc_dev_arr.get_pointer(), arr_size);

				vec3 col(0, 0, 0);
				for (int s = 0; s < ns; s++) {
					float u = float(i + rand.Get()) / float(nx);
					float v = float(j + rand.Get()) / float(ny);
					ray r = acc_dev_camera.get_pointer()->get_ray(u, v, rand);
					col += color_loop(r, acc_dev_sphere.get_pointer(), sphere_size, rand);
				}
				col /= float(ns);
				col = vec3(sqrt(col[0]), sqrt(col[1]), sqrt(col[2]));
				int ir = int(255.99 * col[0]);
				int ig = int(255.99 * col[1]);
				int ib = int(255.99 * col[2]);

				acc_dev_pixelsSrc[index] = (0xff000000 | (ib << 16) | (ig << 8) | ir);

				});

			});
		Q.wait();

		dev_pixelsSrc.get_access<access::mode::read>(); // <--- Host Accessor to Synchronize Memory
	}
	catch (sycl::exception& ex)
	{
		std::cerr << "SYCL Exception thrown: " << ex.what() << std::endl;
	}
	catch (std::exception& ex)
	{
		std::cerr << "std Exception thrown: " << ex.what() << std::endl;
	}
	std::cout << "\nDone!\n";

	stopwatch.stop();

	int channels = 4;
	stbi_write_png("ray_trace.png", nx, ny, channels, pixelsSrc.data(), nx * channels);

	return 0;
}

