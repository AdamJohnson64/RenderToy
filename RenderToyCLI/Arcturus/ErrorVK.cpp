#include "ErrorVK.h"

#include <exception>
#include <string>

namespace Arcturus
{
    static std::string ConvertToErrorString(const char* line, VkResult error)
    {
        return std::string(line);
    }

    void HandleErrorVK(const char* line, VkResult error)
    {
        if (VK_SUCCESS != error) throw std::exception(ConvertToErrorString(line, error).c_str());
    }
}