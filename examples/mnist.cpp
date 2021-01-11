class MnistNet : public model
{
    node& create_model()
    {
        auto& input = create_input_node();

        auto& w1 = create_model_param(784,128);
        //auto& w2 = create_model_param(128,128);
        auto& w2 = create_model_param(128,10);

        auto& b1 = create_model_param(1,128);
        //auto& b2 = create_model_param(1,128);
        auto& b2 = create_model_param(1,10);

        //return input.dot(w1).relu().dot(w2).logsoftmax();

        auto& l1 = (input.dot(w1) + b1).relu(); //layer 1
        auto& l2 = (l1.dot(w2) + b2).sigmoid(); //layer 2
        //auto& l3 = (l2.dot(w3) + b3).sigmoid(); //layer 3
        
        return l2;
    }
};



void do_mnist()
{
    std::vector<tensor> X;
    std::vector<tensor> y;

    std::ifstream train_x_file("./datasets/mnist/train-images-idx3-ubyte", std::ios::binary);
    std::ifstream train_y_file("./datasets/mnist/train-labels-idx1-ubyte", std::ios::binary);

    std::cout << "loading dataset..." << std::endl;
    std::vector<char> train_x_bytes((std::istreambuf_iterator<char>(train_x_file)), (std::istreambuf_iterator<char>()));
    std::vector<char> train_y_bytes((std::istreambuf_iterator<char>(train_y_file)), (std::istreambuf_iterator<char>()));

    //get rid format bytes
    train_x_bytes.erase(train_x_bytes.begin(), train_x_bytes.begin() + 16); 
    train_y_bytes.erase(train_y_bytes.begin(), train_y_bytes.begin() + 8);

    train_x_file.close();
    train_y_file.close();

    size_t img_count = (train_x_bytes.size()) / 784;

    img_count = 4000;


    std::cout << "done loading dataset" << std::endl;
    std::cout << "traning samples: " << img_count << std::endl;

    std::cout << train_x_bytes.size() << std::endl;


    //get labels
    for(size_t i = 0; i < img_count; i++) //train_y_bytes.size()
    {
        size_t index = (size_t) train_y_bytes[i];
        tensor tmp(1,10);
        tmp.zeros();
        tmp[index] = 1.0f;
        y.push_back(tmp);
    }

    //get inputs
    
    size_t img_size = 28 * 28;
    for(size_t i = 0; i < img_count; i++) //train_x_bytes.size()
    {
        tensor tmp(1,img_size);
        size_t offset = i * img_size;
        for(size_t j = 0; j < img_size; j++)
        {
            tmp[j] = (double) (unsigned char) train_x_bytes[offset + j] / 255; //scale to 0-1;
        }
        X.push_back(tmp);
    }

    double learning_rate = 0.0001;
    unsigned int epoch = 50;
    SDG optim(learning_rate);
    MnistNet model;
    
    model.train(X,y, optim, epoch, loss::mean_squared_error);
}