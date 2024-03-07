#pragma once

float unpack_ieee32_le(unsigned char b0, unsigned char b1, unsigned char b2, unsigned char b3)
{
    float f;
    unsigned char b[] = { b3, b2, b1, b0 };
    memcpy(&f, &b, sizeof(f));
    return f;
}