#pragma once

#include "tensor/tensor.hpp"
#include "tensor/tensor_ops.hpp"
#include "tensor/tensor_operators.hpp"

#include "autograd/node.hpp"
#include "autograd/graph.hpp"
#include "autograd/session.hpp"

// #include "nn/model.hpp"
// #include "nn/optim.hpp"
// #include "nn/loss_functions.hpp"
#include "utils.hpp"


namespace czy{
namespace utils{

template <typename T>
void clean_session(){
    Session<T>& session = Session<T>::get_session();
    
    auto& active_nodes = session.get_session_nodes();
    while(!active_nodes.empty())
    {
        delete active_nodes.back();;
        active_nodes.pop_back();
    }
}

}//namespace utils
}//namespace czy
