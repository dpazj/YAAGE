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
        void evaluate(tensor<T>& x_test, tensor<T>& y_test, std::function<node<T>&(node<T>&,node<T>&)> loss_fn);

        virtual node<T>& create_model() = 0;

    protected:
        //model builder api
        node<T>& create_input_node();
        node<T>& create_model_param(tensor_shape); //shape

    private:
        std::vector<tensor<T>*> m_model_parameters;
        node<T>* m_input_node = nullptr;
        node<T>* m_model = nullptr;
};

template<typename T>
model<T>::~model()
{
    for(const auto& x : m_model_parameters)
    {
        delete x;
    }
    m_input_node = nullptr;
    m_model = nullptr;
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

    if(m_model == nullptr)
    {
        m_model = &create_model();
    }
    auto& model = *m_model;
    


    node<T> label;

    auto& loss = loss_fn(label, model);
    graph<T> g(loss);

    size_t end;
    size_t samples = x_train.shape()[0];


    for(size_t k=0; k < EPOCHS; k++){
        std::cout << "Epoch " << k + 1 << "/" << EPOCHS << std::endl;
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

            std::cout << "\rBatch " << (i/BATCH_SIZE) << "/" << samples/BATCH_SIZE << std::flush;
        }

        std::cout << std::endl << "Epoch " << k+1 << " average loss :" << av_loss / (samples / BATCH_SIZE) << std::endl; 
    }
}

template<typename T>
void model<T>::evaluate(tensor<T>& x_test, tensor<T>& y_test, std::function<node<T>&(node<T>&,node<T>&)> loss_fn)
{
    std::cout << "Evaluating model..." << std::endl;
    if(m_model == nullptr)
    {
        m_model = &create_model();
    }
    auto& model = *m_model;

    node<T> label;
    tensor<T> av_loss = {0};


    auto& loss = loss_fn(label, model);

    graph<T> g(loss);

    double accuracy = 0.0;

    size_t samples = x_test.shape()[0];

    for(size_t i=0; i<samples;i++)
    {

        auto item_x = x_test.slice(i, i+1); 
        auto item_y = y_test.slice(i, i+1); 

        m_input_node->set_data(&item_x);
        label.set_data(&item_y);
        g.forwards();

        av_loss = av_loss + loss.data();
        

        tensor<T> prediction = model.data();
       

        T* prediction_data_ptr = prediction.data();
        size_t max_idx = 0;
        T max = 0;
        for(size_t i=0; i<prediction.size();i++)
        {
            if(prediction_data_ptr[i] > max)
            {
                max = prediction_data_ptr[i];
                max_idx = i;
            }
        }
        prediction.zeros();
        prediction_data_ptr[max_idx] = 1.0;

        //stats
        if(prediction == item_y)
        {
            accuracy +=1;
        }
        
        
    }

    std::cout << "Evaluation average loss: " << av_loss / samples << std::endl;
    std::cout << "Evaluation accuracy: " << accuracy/ samples * 100 << "%" << std::endl;

        
}

}//namespace czy


