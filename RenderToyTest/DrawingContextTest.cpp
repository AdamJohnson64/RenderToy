#include "gtest\gtest.h"

#include "Arcturus/DrawingContextReference.h"

const int TEST_PAGE_WIDTH = 64;
const int TEST_PAGE_HEIGHT = 64;

const Arcturus::Vec2 DEFAULT_TOP_LEFT = { 0 + 8.5f, 0 + 8.5f };
const Arcturus::Vec2 DEFAULT_BOTTOM_RIGHT = { TEST_PAGE_WIDTH - 8.5f, TEST_PAGE_HEIGHT - 8.5f };

struct PixelRGBA32 { uint8_t R, G, B, A; };

bool IsSharp(const void* pixels, uint32_t width, uint32_t height, uint32_t stride)
{
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(pixels);
    // All pixels should be perfect black or perfect white for a sharp image.
    for (uint32_t y = 0; y < TEST_PAGE_HEIGHT; ++y)
    {
        for (uint32_t x = 0; x < TEST_PAGE_WIDTH; ++x)
        {
            const PixelRGBA32* p = reinterpret_cast<const PixelRGBA32*>(bytes + sizeof(uint32_t) * x + stride * y);
            if (!(p->R == 0 || p->R == 0xFF)) return false;
            if (!(p->G == 0 || p->G == 0xFF)) return false;
            if (!(p->B == 0 || p->B == 0xFF)) return false;
            if (!(p->A == 0 || p->A == 0xFF)) return false;
        }
    }
    return true;
}

TEST(DrawingContextReference, DrawLineSharp)
{
    Arcturus::DrawingContextReference dc;
    dc.moveTo({ DEFAULT_TOP_LEFT.X, DEFAULT_TOP_LEFT.Y });
    dc.lineTo({ DEFAULT_BOTTOM_RIGHT.X, DEFAULT_TOP_LEFT.Y });
    dc.lineTo({ DEFAULT_BOTTOM_RIGHT.X, DEFAULT_BOTTOM_RIGHT.Y });
    dc.lineTo({ DEFAULT_TOP_LEFT.X, DEFAULT_BOTTOM_RIGHT.Y });
    dc.lineTo({ DEFAULT_TOP_LEFT.X, DEFAULT_TOP_LEFT.Y });
    uint8_t pixels[sizeof(uint32_t) * TEST_PAGE_WIDTH * TEST_PAGE_HEIGHT];
    uint32_t stride = sizeof(uint32_t) * TEST_PAGE_WIDTH;
    dc.renderTo(pixels, TEST_PAGE_WIDTH, TEST_PAGE_HEIGHT, stride);
    // This sequence of lines should align perfectly to the half-pixel with a pen width of 1.
    // The lines in this rectangle should be white and perfectly sharp (black 0x00, or white 0xFF).
    ASSERT_TRUE(IsSharp(pixels, TEST_PAGE_WIDTH, TEST_PAGE_HEIGHT, stride));
}

TEST(DrawingContextReference, DrawRectangleSharp)
{
    Arcturus::DrawingContextReference dc;
    dc.drawRectangle(DEFAULT_TOP_LEFT, DEFAULT_BOTTOM_RIGHT);
    uint8_t pixels[sizeof(uint32_t) * TEST_PAGE_WIDTH * TEST_PAGE_HEIGHT];
    uint32_t stride = sizeof(uint32_t) * TEST_PAGE_WIDTH;
    dc.renderTo(pixels, TEST_PAGE_WIDTH, TEST_PAGE_HEIGHT, stride);
    // This rectangle should align perfectly to the half-pixel with a pen width of 1.
    // The lines in this rectangle should be white and perfectly sharp (black 0x00, or white 0xFF).
    ASSERT_TRUE(IsSharp(pixels, TEST_PAGE_WIDTH, TEST_PAGE_HEIGHT, stride));
}

TEST(DrawingContextReference, FillRectangleSharp)
{
    Arcturus::DrawingContextReference dc;
    dc.fillRectangle({ 8, 8 }, { TEST_PAGE_WIDTH - 8, TEST_PAGE_HEIGHT - 8 });
    uint8_t pixels[sizeof(uint32_t) * TEST_PAGE_WIDTH * TEST_PAGE_HEIGHT];
    uint32_t stride = sizeof(uint32_t) * TEST_PAGE_WIDTH;
    dc.renderTo(pixels, TEST_PAGE_WIDTH, TEST_PAGE_HEIGHT, stride);
    // This rectangle should align perfectly to fit within pixels.
    // All pixels should be white and perfectly sharp (black 0x00, or white 0xFF).
    ASSERT_TRUE(IsSharp(pixels, TEST_PAGE_WIDTH, TEST_PAGE_HEIGHT, stride));
}