#pragma once
#include "session.hpp"
#include "node.hpp"

namespace utils
{
    void clean_session()
    {
        Session& session = Session::get_session();
        for(const auto& node : session.get_session_nodes())
        {
            delete node;
        }
    } 
} // namespace utils


