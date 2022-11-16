#ifndef LR_H
#define LR_H


#include <ndarray.h>
#include <iostream>


template<typename T>
class LinearRegression {

public:
    LinearRegression() = default;
    LinearRegression(ndarray<T> x, ndarray<T> y);
    ndarray<T> predict(ndarray<T> x);
    ndarray<T> getWeights();
    ndarray<T> getBias();
    void fit(ndarray<T> x, ndarray<T> y, int epochs, T lr);
    void fit(int epochs, T lr);
    void fit(int epochs);
    void fit();
    void setWeights(ndarray<T> w);
    void setBias(ndarray<T> b);
    void setX(ndarray<T> x);
    void setY(ndarray<T> y);
    void setLearningRate(T lr);
    void setEpochs(int epochs);
    void setLoss(ndarray<T> loss);
    ndarray<T> getLoss();
    ndarray<T> getLossDerivative();
    ndarray<T> predict();

private:
    ndarray<T> x;
    ndarray<T> y;
    ndarray<T> w;
    ndarray<T> b;
    ndarray<T> loss;
    ndarray<T> loss_derivative;
    T lr;
    int epochs;
    void updateWeights();
    void updateBias();
    void updateLoss();
    void updateLossDerivative();
};

template<typename T>
LinearRegression<T>::LinearRegression(ndarray<T> x, ndarray<T> y) {
    this->x = x;
    this->y = y;
    this->w = ndarray<T>({x.shape()[0], 1});
    this->b = ndarray<T>({1, 1});
    this->w.random(-1, 1);
    this->b.random(-1, 1);
    this->lr = 0.01;
    this->epochs = 100;
    this->loss = ndarray<T>({1, 1});
    this->loss_derivative = ndarray<T>({1, 1});
}

template<typename T>
ndarray<T> LinearRegression<T>::predict(ndarray<T> x) {
    return  this->b + x.dot(this->w);
}

template<typename T>
ndarray<T> LinearRegression<T>::getWeights() {
    return this->w;
}

template<typename T>
ndarray<T> LinearRegression<T>::getBias() {
    return this->b;
}

template<typename T>
void LinearRegression<T>::fit(ndarray<T> x, ndarray<T> y, int epochs, T lr) {
    this->x = x;
    this->y = y;
    this->lr = lr;
    this->epochs = epochs;
    this->fit();
}

template<typename T>
void LinearRegression<T>::fit(int epochs, T lr) {
    this->lr = lr;
    this->epochs = epochs;
    this->fit();
}

template<typename T>
void LinearRegression<T>::fit(int epochs) {
    this->epochs = epochs;
    this->fit();
}

template<typename T>
void LinearRegression<T>::fit() {
    for (int i = 0; i < this->epochs; i++) {
        std::cout << "Epoch: " << i << std::endl;
        this->updateWeights();
        this->updateBias();
        this->updateLoss();
        this->updateLossDerivative();
    }
}

template<typename T>
void LinearRegression<T>::setWeights(ndarray<T> w) {
    this->w = w;
}

template<typename T>
void LinearRegression<T>::setBias(ndarray<T> b) {
    this->b = b;
}

template<typename T>
void LinearRegression<T>::setX(ndarray<T> x) {
    this->x = x;
}

template<typename T>
void LinearRegression<T>::setY(ndarray<T> y) {
    this->y = y;
}

template<typename T>
void LinearRegression<T>::setLearningRate(T lr) {
    this->lr = lr;
}

template<typename T>
void LinearRegression<T>::setEpochs(int epochs) {
    this->epochs = epochs;
}

template<typename T>
void LinearRegression<T>::setLoss(ndarray<T> loss) {
    this->loss = loss;
}

template<typename T>
ndarray<T> LinearRegression<T>::getLoss() {
    return this->loss;
}

template<typename T>
ndarray<T> LinearRegression<T>::getLossDerivative() {
    return this->loss_derivative;
}

template<typename T>
ndarray<T> LinearRegression<T>::predict() {
    return this->predict(this->x);
}

template<typename T>
void LinearRegression<T>::updateWeights() {
    std::cout << this->x.transpose() << std::endl;
    std::cout << this->loss_derivative << std::endl;
    this->w = this->w - this->lr * this->x.transpose().dot(this->loss_derivative);
}

template<typename T>
void LinearRegression<T>::updateBias() {
    this->b -= this->loss_derivative.sum() * this->lr;
}

template<typename T>
void LinearRegression<T>::updateLoss() {
    this->loss = this->predict() - this->y;
}

template<typename T>
void LinearRegression<T>::updateLossDerivative() {
    this->loss_derivative = this->loss;
}




#endif
