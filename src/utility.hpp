#pragma once

#include <cmath>

float unpack_ieee32_le(unsigned char b0, unsigned char b1, unsigned char b2, unsigned char b3)
{
    float f;
    unsigned char b[] = { b3, b2, b1, b0 };
    memcpy(&f, &b, sizeof(f));
    return f;
}

template<typename T>
struct vec3
{
    T x = 0;
    T y = 0;
    T z = 0;

    void normalize()
    {
        auto len = length();
        x/=len; y/=len; z/=len;
    }

    float length() const
    {
        return std::sqrt(x * x + y * y + z* z);
    }

    vec3<T> operator-(const vec3<T> & other)
    {
        return {x - other.x, y - other.y, z - other.z};
    }

    vec3<T> operator*(const vec3<T>& other)
    {
        return { x * other.x, y * other.y, z * other.z };
    }
};

template<typename T>
vec3<T> operator*(T scalar, const vec3<T>& vec) {
    return { vec.x * scalar, vec.y * scalar, vec.z * scalar };
}

template<typename T>
T normalize(T&& v)
{
    auto len = v.length();

    if(std::abs(len) < 1e-6)
    {
        return v;
    }

    return {v.x/len,v.y/len,v.z/len};
}

template<typename T>
std::decay_t<T> cross(T&& v1,T&& v2)
{
    return {v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x};
}

template<typename T>
auto dot(T&& v1,T&& v2)
{
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
            __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)