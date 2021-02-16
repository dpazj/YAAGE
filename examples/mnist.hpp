#include "../cozygrad/cozygrad.h"
#include <vector>
#include <fstream>
#include <sstream>


using namespace czy;

template <typename T>
class MnistNet : public model<T>
{
    node<T>& create_model()
    {
        auto& input = this->create_input_node();

        size_t xx = 50;


        auto& w1 = this->create_model_param({784,xx});
        auto& w2 = this->create_model_param({xx,xx});
        auto& w3 = this->create_model_param({xx,10});


        auto& b1 = this->create_model_param({1,xx});
        auto& b2 = this->create_model_param({1,xx});
        auto& b3 = this->create_model_param({1,10});


        auto& l1 = (input.dot(w1) + b1).sigmoid(); //layer 1
        auto& l2 = (l1.dot(w2) + b2).sigmoid(); //layer 2
        auto& l3 = (l2.dot(w3) + b3).sigmoid(); //layer 3
        
        return l3;
    }
};



void do_mnist()
{

    std::ifstream train_x_file("./datasets/mnist/train-images-idx3-ubyte", std::ios::binary);
    std::ifstream train_y_file("./datasets/mnist/train-labels-idx1-ubyte", std::ios::binary);

    std::cout << "loading dataset..." << std::endl;
    std::vector<unsigned char> train_x_bytes((std::istreambuf_iterator<char>(train_x_file)), (std::istreambuf_iterator<char>()));
    std::vector<unsigned char> train_y_bytes((std::istreambuf_iterator<char>(train_y_file)), (std::istreambuf_iterator<char>()));

    //get rid of format bytes
    train_x_bytes.erase(train_x_bytes.begin(), train_x_bytes.begin() + 16); 
    train_y_bytes.erase(train_y_bytes.begin(), train_y_bytes.begin() + 8);

    train_x_file.close();
    train_y_file.close();

    //need to convert bytes to double values
    std::vector<double> train_x_double(train_x_bytes.begin(), train_x_bytes.end());
    std::vector<double> train_y_double;

    for(const auto& byte : train_y_bytes)
    {
        size_t idx = (size_t) byte;
        std::vector<double> tmp(10,0.0);
        tmp[idx] = 1.0;
        train_y_double.insert(train_y_double.end(), tmp.begin(), tmp.end());
    }

    size_t img_count = train_x_bytes.size() / 784;

    std::cout << "done loading dataset!" << std::endl;
    std::cout << "traning samples: " << img_count << std::endl;

    tensor<double> X(train_x_double, {img_count, 784});
    tensor<double> y(train_y_double, {img_count, 10} );

    //scale to 0-1
    X.map([](double x){ return x / 255;});

    //visulaise dataset
    // X = X > 0;
    // size_t c = 0;
    // for(size_t i=0; i < X.size(); i+=784)
    // {
    //     std::cout << y.slice(c,c+1) << std::endl;
    //     c++;
    //     for(size_t j=0; j < 28; j++)
    //     {
    //         for(size_t k=0;k<28;k++)
    //         {
    //             std::cout << X[i + j * 28 + k] << "";
    //         }
    //         std::cout << std::endl;
    //     }
    //     std::cout << std::endl;
    // }
    // X = X.slice(0,10000);    
    // y = y.slice(0,10000);    


    double learning_rate = 10;
    unsigned int epochs = 30;
    size_t batch_size = 128;
    SDG<double> optim(learning_rate);
    MnistNet<double> model;


    auto X_train = X.slice(0,55000);
    auto y_train = y.slice(0,55000);

    auto X_test = X.slice(55000);
    auto y_test = y.slice(55000);
    
    model.train(X_train,y_train, optim, batch_size, epochs, loss::mean_squared_error<double>);
    model.evaluate(X_test,y_test, loss::mean_squared_error<double>);


}