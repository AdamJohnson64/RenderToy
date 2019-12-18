#include "gtest\gtest.h"

#include "Arcturus/DrawingContextReference.h"

const int TEST_PAGE_WIDTH = 64;
const int TEST_PAGE_HEIGHT = 64;

const Arcturus::Vec2 DEFAULT_TOP_LEFT = { 0 + 8.5f, 0 + 8.5f };
const Arcturus::Vec2 DEFAULT_BOTTOM_RIGHT = { TEST_PAGE_WIDTH - 8.5f, TEST_PAGE_HEIGHT - 8.5f };

TEST(DrawingContextReference, Rectangle)
{
    Arcturus::DrawingContextReference dc;
    dc.drawRectangle(DEFAULT_TOP_LEFT, DEFAULT_BOTTOM_RIGHT);
    uint8_t pixels[sizeof(uint32_t) * TEST_PAGE_WIDTH * TEST_PAGE_HEIGHT];
    dc.renderTo(pixels, TEST_PAGE_WIDTH, TEST_PAGE_HEIGHT, sizeof(uint32_t) * TEST_PAGE_WIDTH);
}