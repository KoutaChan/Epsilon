#pragma once

#if __has_include(<backend/function_attributes.h>)
#include <backend/function_attributes.h>
#endif

#ifndef PRISM_DEVICE_INLINE
#define PRISM_HOST_DEVICE
#define PRISM_DEVICE_INLINE inline
#endif
