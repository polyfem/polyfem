#pragma once

////////////////////////////////////////////////////////////////////////////////

#define POLYFEM_DEFAULT_MOVE_COPY(Base)     \
    Base(Base&&) = default;                 \
    Base& operator=(Base&&) = default;      \
    Base(const Base&) = default;            \
    Base& operator=(const Base&) = default; \

#define POLYFEM_DELETE_MOVE_COPY(Base)     \
    Base(Base&&) = delete;                 \
    Base& operator=(Base&&) = delete;      \
    Base(const Base&) = delete;            \
    Base& operator=(const Base&) = delete; \

////////////////////////////////////////////////////////////////////////////////
// External libraries
////////////////////////////////////////////////////////////////////////////////

// Json
#include "json.hpp"
using json = nlohmann::json;
