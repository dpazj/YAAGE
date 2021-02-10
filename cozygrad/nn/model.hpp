#pragma once

#include "optim.hpp"
#include "../tensor/tensor.hpp"
#include "../autograd/node.hpp"
#include "../autograd/graph.hpp"
#include "../autograd/session.hpp"

#include <vector>
#include <functional>
#include <exception>

namespace czy{

template<typename T>
class model
{
    public: 
        ~model();

        void train(tensor<T>& x_train, tensor<T>& y_train, optimizer<T>& optim, size_t BATCH_SIZE, unsigned int epochs, std::function<node<T>&(node<T>&,node<T>&)> loss_fn);
        //void evaluate(std::vector<tensor>& x_test, std::vector<tensor>& y_test);


        virtual node<T>& create_model() = 0;

    protected:
        //model builder api
        node<T>& create_input_node();
        node<T>& create_model_param(tensor_shape); //shape

    private:
        std::vector<tensor<T>*> m_model_parameters;
        node<T>* m_input_node = nullptr;
        node<T>* m_model = nullptr;

        std::function<node<T>&(node<T>&,node<T>&)> m_loss_fn;
};

template<typename T>
model<T>::~model()
{
    for(const auto& x : m_model_parameters)
    {
        delete x;
    }
}

template<typename T>
node<T>& model<T>::create_input_node()
{
    node<T>* input = new node<T>();

    auto& session = Session<T>::get_session();
    session.add_node(input);

    m_input_node = input;
    input->name = "input";

    return *input;
}

template<typename T>
node<T>& model<T>::create_model_param(tensor_shape shape)
{
    tensor<T>* param = new tensor<T>(shape);
    param->random(-1.0f,1.0f); // initialises the parameters to random (-1, 1) 
    m_model_parameters.push_back(param);

    node<T>* out = new node<T>(param);

    auto& session = Session<T>::get_session();
    session.add_node(out);

    return *out;
}

template<typename T>
void model<T>::train(tensor<T>& x_train, tensor<T>& y_train, optimizer<T>& optim, size_t BATCH_SIZE, unsigned int EPOCHS, std::function<node<T>&(node<T>&,node<T>&)> loss_fn)
{
    std::cout << "started training" << std::endl;


    auto& model = create_model();
    node<T> label;

    m_model = &model;
    m_loss_fn = loss_fn;

    auto& loss = loss_fn(label, model);
    graph<T> g(loss);

    size_t end;
  
    size_t samples = x_train.shape()[0];


    for(size_t k=0; k < EPOCHS; k++){
        std::cout << "start epoch " << k + 1 << std::endl;
        tensor<T> av_loss = {0};

        end = 0;
        for(size_t i=0; i < samples; i+=BATCH_SIZE)
        {

            end+=BATCH_SIZE;
            if(end > samples){end = samples;}

            auto batch_x = x_train.slice(i, end); 
            auto batch_y = y_train.slice(i, end); 
            
            m_input_node->set_data(&batch_x);
            label.set_data(&batch_y);

            g.forwards();
            g.backwards();
            optim.step(g.nodes());
            g.zero_gradients();

            //stats
            av_loss = av_loss + loss.data();

            std::cout << "Batch " << (i/BATCH_SIZE) << "/" << samples/BATCH_SIZE << " Loss: " << loss.data() << std::endl;

        }
    }
}

// void model::evaluate(std::vector<tensor>& x_test, std::vector<tensor>& y_test)
// {
//     if(m_model == nullptr)
//     {
//         throw std::runtime_error("Model has not been trained yet : )");
//     }

//     auto& model = *m_model;
//     node label;
//     tensor av_loss = {0};


//     auto& loss = m_loss_fn(label, model);

//     graph g(loss);


//     for(size_t i=0; i < x_test.size();i++)
//     {
//         m_input_node->set_data(&x_test[i]);
//         label.set_data(&y_test[i]);
//         g.forwards();

//         av_loss = av_loss + *loss.data();
        
//         //stats
//     }

//     std::cout << "Evaluation average loss: " << av_loss.data()[0] / x_test.size() << std::endl;
        
// }

}//namespace czy


