#pragma once

#include "node.hpp"
#include "tensor.hpp"
#include "graph.hpp"
#include "session.hpp"
#include "utils.hpp"
#include "model.hpp"
#include "optim.hpp"
#include "loss_functions.hpp"

namespace czy{
namespace utils{

void clean_session(){
    Session& session = Session::get_session();
        
    for(auto* x : session.get_session_nodes())
    {
        delete x;
    }
}

}//namespace utils
}//namespace czy
