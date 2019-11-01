#pragma once

#include "VulkanInclude.h"

namespace Arcturus
{

    void HandleErrorVK(const char* line, VkResult error);

    #define TRYVK(FN) HandleErrorVK(#FN, ##FN);

}