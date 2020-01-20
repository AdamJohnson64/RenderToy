#include "RenderToyCLI\Arcturus\DrawingContextReference.h"

unsigned char charcolor(unsigned int pixel)
{
	int intensity = pixel & 0xFF;
	if (intensity < 0x20) return 32;
	if (intensity < 0x40) return 176;
	if (intensity < 0x80) return 177;
	if (intensity < 0xC0) return 178;
	return 219;
}

int main()
{
	const int WIDTH = 64;
	const int HEIGHT = 64;
	Arcturus::DrawingContextReference ctx;
	ctx.moveTo({2, 2});
	ctx.lineTo({8, 8});
	ctx.drawCircle({20, 10}, 8);
	ctx.setWidth(8);
	ctx.drawCircle({10, 50}, 40);
	unsigned int buffer[WIDTH * HEIGHT];
	ctx.renderTo(buffer, WIDTH, HEIGHT, sizeof(unsigned int) * WIDTH);
	for (int y = 0; y < HEIGHT; ++y)
	{
		for (int x = 0; x < WIDTH; ++x)
		{
			printf("%c", charcolor(buffer[x + WIDTH * y]));
		}
		printf("\n");
	}
	return -1;
}