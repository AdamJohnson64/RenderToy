#include <memory>
#include <msclr\marshal_cppstd.h>
#include <png.h>
#include <pngstruct.h>
#include <pnginfo.h>

extern "C" static void PrintPNG(png_structp readstruct, const char *error)
{
	int test = 0;
}

namespace RenderToy
{
	public ref class ImageBGRA32
	{
	public:
		int Width;
		int Height;
		cli::array<byte>^ Data;
	};
	public ref class LibPNG
	{
	public:
		static ImageBGRA32^ Open(System::String ^path)
		{
			msclr::interop::marshal_context ctx;
			const char *szPath = ctx.marshal_as<const char*>(path);
			FILE *fp = fopen(szPath, "rb");
			if (fp == nullptr)
			{
				throw gcnew System::Exception("PNG: Unable to open file '" + path + "'.");
			}
			auto readstruct = ::png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
			::png_init_io(readstruct, fp);
			::png_info info = { 0 };
			::png_read_info(readstruct, &info);
			auto bitsperchannel = ::png_get_bit_depth(readstruct, &info);
			if (bitsperchannel != 8)
			{
				throw gcnew System::Exception("PNG: This library only supports 8 bits per channel.");
			}
			auto channelsperpixel = ::png_get_channels(readstruct, &info);
			if (channelsperpixel != 3 && channelsperpixel != 4)
			{
				throw gcnew System::Exception("PNG: This library only supports 3 channel (RGB) or 4 channel (RGBA) pixel formats.");
			}
			auto width = ::png_get_image_width(readstruct, &info);
			auto height = ::png_get_image_height(readstruct, &info);
			auto imagedata = gcnew cli::array<byte>(4 * width * height);
			{
				auto rastersize = ::png_get_rowbytes(readstruct, &info);
				auto raster = std::unique_ptr<png_byte[]>(new png_byte[rastersize]);
				for (int y = 0; y < height; ++y)
				{
					::png_read_row(readstruct, raster.get(), nullptr);
					if (channelsperpixel == 3 && bitsperchannel == 8)
					{
						for (int x = 0; x < width; ++x)
						{
							imagedata[0 + 4 * (x + width * y)] = raster[0 + 3 * x];
							imagedata[1 + 4 * (x + width * y)] = raster[1 + 3 * x];
							imagedata[2 + 4 * (x + width * y)] = raster[2 + 3 * x];
							imagedata[3 + 4 * (x + width * y)] = 0xFF; // Opaque alpha.
						}
					}
					if (channelsperpixel == 4 && bitsperchannel == 8)
					{
						for (int x = 0; x < width; ++x)
						{
							imagedata[0 + 4 * (x + width * y)] = raster[0 + 4 * x];
							imagedata[1 + 4 * (x + width * y)] = raster[1 + 4 * x];
							imagedata[2 + 4 * (x + width * y)] = raster[2 + 4 * x];
							imagedata[3 + 4 * (x + width * y)] = raster[3 + 4 * x];
						}
					}
				}
			}
			fclose(fp);
			ImageBGRA32 ^output = gcnew ImageBGRA32();
			output->Width = width;
			output->Height = height;
			output->Data = imagedata;
			return output;
		}
	};
}