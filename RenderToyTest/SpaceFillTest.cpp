#include "gtest\gtest.h"

const uint32_t W = 511;
const uint32_t H = 255;

struct Rectangle { uint32_t x, y, w, h; };

uint32_t Min(uint32_t a, uint32_t b)
{
    return a < b ? a : b;
}

uint32_t HighestSetBit(uint32_t value)
{
    uint32_t count = -1;
    while (value)
    {
        value >>= 1;
        ++count;
    }
    return count;
}

void SpaceFillBinarySquare(const Rectangle& r, std::function<void(const Rectangle&)> fn)
{
    // Terminate if the rectangle has zero area. 
    if (r.w <= 0 || r.h <= 0)
    {
        return;
    }
    // Determine the largest binary edge square that fits.
    uint32_t l = 1 << Min(HighestSetBit(r.w), HighestSetBit(r.h));
    fn(Rectangle { r.x, r.y, l, l });
    SpaceFillBinarySquare(Rectangle { r.x + l, r.y, r.w - l, l }, fn);
    SpaceFillBinarySquare(Rectangle { r.x, r.y + l, l, r.h - l }, fn);
    SpaceFillBinarySquare(Rectangle { r.x + l, r.y + l, r.w - l, r.h - l }, fn);
}

void SpaceFillBisectMajor(const Rectangle& r, std::function<void(const Rectangle&)> fn)
{
    // Terminate if the rectangle has zero area. 
    if (r.w <= 0 || r.h <= 0)
    {
        return;
    }
    if (r.w <= 16 && r.h <= 16)
    {
        fn(Rectangle { r.x, r.y, r.w, r.h });
        return;
    }
    // Determine the major axis.
    if (r.w >= r.h)
    {
        // Bisect on X.
        SpaceFillBisectMajor(Rectangle { r.x, r.y, r.w / 2, r.h }, fn);
        SpaceFillBisectMajor(Rectangle { r.x + r.w / 2, r.y, r.w - r.w / 2, r.h }, fn);
    }
    else
    {
        // Bisect on Y.
        SpaceFillBisectMajor(Rectangle { r.x, r.y, r.w , r.h / 2 }, fn);
        SpaceFillBisectMajor(Rectangle { r.x, r.y + r.h / 2, r.w, r.h - r.h / 2 }, fn);
    }
}

struct Slicer
{
    void Slice()
    {
        // Terminate if the rectangle has zero area. 
        if (region.w <= 0 || region.h <= 0)
        {
            return;
        }
        // Terminate if there is nothing to process.
        if (contents.size() <= 0)
        {
            printf("Rectangle(%d, %d, %d, %d) is empty.\n", region.x, region.y, region.w, region.h);
            return;
        }
        if (region.w <= 8 && region.h <= 8)
        {
            printf("Rectangle(%d, %d, %d, %d) will be processed.\n", region.x, region.y, region.w, region.h);
            return;
        }
        // Determine the major axis.
        if (region.w >= region.h)
        {
            // Bisect on X.
            Slicer(BuildSubcontext({ region.x, region.y, region.w / 2, region.h })).Slice();
            Slicer(BuildSubcontext({ region.x + region.w / 2, region.y, region.w - region.w / 2, region.h })).Slice();
        }
        else
        {
            // Bisect on Y.
            Slicer(BuildSubcontext({ region.x, region.y, region.w, region.h / 2 })).Slice();
            Slicer(BuildSubcontext({ region.x, region.y + region.h / 2, region.w, region.h - region.h / 2 })).Slice();
        }
    }
    Slicer BuildSubcontext(const Rectangle& subregion)
    {
        uint32_t tx1 = subregion.x;
        uint32_t ty1 = subregion.y;
        uint32_t tx2 = tx1 + subregion.w;
        uint32_t ty2 = ty1 + subregion.h;
        std::vector<Rectangle> newthings;
        for (const Rectangle& check : contents)
        {
            uint32_t nx1 = check.x;
            uint32_t ny1 = check.y;
            uint32_t nx2 = nx1 + check.w;
            uint32_t ny2 = ny1 + check.h;
            if (nx2 < tx1) continue;
            if (ny2 < ty1) continue;
            if (nx1 > tx2) continue;
            if (ny1 > ty2) continue;
            newthings.push_back(check);
        }
        return Slicer { subregion, newthings };
    }
    Rectangle region;
    std::vector<Rectangle> contents;
};

////////////////////////////////////////////////////////////////////////////////

TEST(SpaceFillBinarySquare, DebugPrint)
{
    SpaceFillBinarySquare(Rectangle { 0, 0, W, H }, [](const Rectangle& r) {
        printf("Origin = (%d, %d), Size = (%d, %d)\n", r.x, r.y, r.w, r.h);
    });
}

TEST(SpaceFillBinarySquare, FillsArea)
{
    std::unique_ptr<uint8_t> pixels(new uint8_t[W * H]);
    memset(pixels.get(), 0, W * H);
    SpaceFillBinarySquare(Rectangle { 0, 0, W, H }, [&](const Rectangle& r) {
        for (uint32_t fy = 0; fy < r.h; ++fy)
        {
            for (uint32_t fx = 0; fx < r.w; ++fx)
            {
                ++pixels.get()[(r.x + fx) + W * (r.y + fy)];
            }
        }
    });
    // Verify that all pixels have been touched exactly once.
    for (uint32_t fy = 0; fy < H; ++fy)
    {
        for (uint32_t fx = 0; fx < W; ++fx)
        {
            ASSERT_EQ(pixels.get()[fx + W * fy], 1);
        }
    }
}

TEST(SpaceFillBisectMajor, DebugPrint)
{
    SpaceFillBisectMajor(Rectangle { 0, 0, W, H }, [&](const Rectangle& r) {
        printf("Origin = (%d, %d), Size = (%d, %d)\n", r.x, r.y, r.w, r.h);
    });
}

TEST(SpaceFillBisectMajor, FillsArea)
{
    std::unique_ptr<uint8_t> pixels(new uint8_t[W * H]);
    memset(pixels.get(), 0, W * H);
    SpaceFillBisectMajor(Rectangle { 0, 0, W, H }, [&](const Rectangle& r) {
        for (uint32_t fy = 0; fy < r.h; ++fy)
        {
            for (uint32_t fx = 0; fx < r.w; ++fx)
            {
                ++pixels.get()[(r.x + fx) + W * (r.y + fy)];
            }
        }
    });
    // Verify that all pixels have been touched exactly once.
    for (uint32_t fy = 0; fy < H; ++fy)
    {
        for (uint32_t fx = 0; fx < W; ++fx)
        {
            ASSERT_EQ(pixels.get()[fx + W * fy], 1);
        }
    }
}

TEST(Slicer, Slice)
{
    Slicer c;
    c.region = { 0, 0, W, H};
    c.contents.push_back(Rectangle { 3, 3, 2, 2 });
    c.Slice();
}