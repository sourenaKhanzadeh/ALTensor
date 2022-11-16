#ifndef LR_H
#define LR_H


#include <ndarray.h>
#include <iostream>
#include <math.h>

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
    float MSE();

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
    this->w = ndarray<T>({x.shape()[1], 1});
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
    return x.matMult(this->w) + this->b.flatten()[0];
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
        // flush the buffer
        std::cout << std::flush;
        //clear line
        std::cout << "\r";
        // print progress
        std::cout << "Epoch: " << i << "/" << this->epochs << std::flush;
        this->updateLoss();
        this->updateLossDerivative();
        this->updateWeights();
        this->updateBias();
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
    ndarray<T> dw = this->x.transpose().matMult(this->loss_derivative);
    this->w -= dw * this->lr;
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


template<typename T>
float LinearRegression<T>::MSE() {
    return (this->loss * this->loss).sum() / this->loss.shape()[0];
}



///////////////////////////////////////////
// Logistic Regression
///////////////////////////////////////////
template<typename T>
class LogisticRegression {
public:
    LogisticRegression(ndarray<T> x, ndarray<T> y);
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
    ndarray<T> predict();
    ndarray<T> sigmoid(ndarray<T> x);
    float accuracy();
    float accuracy(ndarray<T> x, ndarray<T> y);

private:
    ndarray<T> x;
    ndarray<T> y;
    ndarray<T> w;
    ndarray<T> b;
    T lr;
    int epochs;
    ndarray<T> loss;
    void updateWeights();
    void updateBias();
    void updateLoss();
    void SGD();
};

template<typename T>
LogisticRegression<T>::LogisticRegression(ndarray<T> x, ndarray<T> y) {
    this->x = x;
    this->y = y;
    this->w = ndarray<T>({x.shape()[1], 1});
    this->b = ndarray<T>({1, 1});
    this->w.random();
    this->b.random();
    this->loss = ndarray<T>({x.shape()[0], 1});
}

template<typename T>
ndarray<T> LogisticRegression<T>::getWeights() {
    return this->w;
}

template<typename T>
ndarray<T> LogisticRegression<T>::getBias() {
    return this->b;
}

template<typename T>
void LogisticRegression<T>::fit(ndarray<T> x, ndarray<T> y, int epochs, T lr) {
    this->x = x;
    this->y = y;
    this->lr = lr;
    this->epochs = epochs;
    this->fit();
}

template<typename T>
void LogisticRegression<T>::fit(int epochs, T lr) {
    this->lr = lr;
    this->epochs = epochs;
    this->fit();
}

template<typename T>
void LogisticRegression<T>::fit(int epochs) {
    this->epochs = epochs;
    this->fit();
}

template<typename T>
void LogisticRegression<T>::fit() {
    for (int i = 0; i < this->epochs; i++) {
        // flush the buffer
        std::cout << std::flush;
        //clear the line
        std::cout << "\r";
        // print the progress
        std::cout << "Epoch: " << i + 1 << "/" << this->epochs << " - " << (float)(i + 1) / this->epochs * 100 << "%";
        this->updateLoss();
        this->SGD();
        this->updateWeights();
        this->updateBias();
    }
}

template<typename T>
void LogisticRegression<T>::setWeights(ndarray<T> w) {
    this->w = w;
}

template<typename T>
void LogisticRegression<T>::setBias(ndarray<T> b) {
    this->b = b;
}

template<typename T>
void LogisticRegression<T>::setX(ndarray<T> x) {
    this->x = x;
}

template<typename T>
void LogisticRegression<T>::setY(ndarray<T> y) {
    this->y = y;
}

template<typename T>
void LogisticRegression<T>::setLearningRate(T lr) {
    this->lr = lr;
}

template<typename T>
void LogisticRegression<T>::setEpochs(int epochs) {
    this->epochs = epochs;
}

template<typename T>
void LogisticRegression<T>::setLoss(ndarray<T> loss) {
    this->loss = loss;
}

template<typename T>
ndarray<T> LogisticRegression<T>::getLoss() {
    return this->loss;
}


template<typename T>
ndarray<T> LogisticRegression<T>::predict() {
    return this->predict(this->x);
}

template<typename T>
float LogisticRegression<T>::accuracy() {
    return this->accuracy(this->x, this->y);
}

template<typename T>
float LogisticRegression<T>::accuracy(ndarray<T> x, ndarray<T> y) {
    return this->predict(x).round().eq(y).sum() / x.shape()[0];
}

template<typename T>
void LogisticRegression<T>::updateWeights() {
    ndarray<T> x_transpose = this->x.transpose();
    ndarray<T> y_pred = this->predict();
    ndarray<T> y_pred_minus_y = y_pred - this->y;
    ndarray<T> x_transpose_dot_y = x_transpose.matMult(y_pred_minus_y);
    ndarray<T> x_transpose_dot_y_div_x_shape = x_transpose_dot_y / this->x.shape()[0];
    ndarray<T> x_transpose_dot_y_div = x_transpose_dot_y_div_x_shape * this->lr;
    this->w = this->w - x_transpose_dot_y_div;
}

template<typename T>
void LogisticRegression<T>::updateBias() {
    ndarray<T> y_pred = this->predict();
    ndarray<T> y_pred_minus_y = y_pred - this->y;
    ndarray<T> y_pred_minus_y_sum = y_pred_minus_y.sum(0);
    ndarray<T> y_pred_minus_y_sum_div = y_pred_minus_y_sum / this->x.shape()[0];
    ndarray<T> y_pred_minus_y_sum_div_lr = y_pred_minus_y_sum_div * this->lr;
    this->b = this->b.flatten() - y_pred_minus_y_sum_div_lr;
}

template<typename T>
void LogisticRegression<T>::updateLoss() {
    ndarray<T> y_pred = this->predict();
    this->loss = this->y - y_pred;
}

template<typename T>
void LogisticRegression<T>::SGD() {
    ndarray<T> y_pred = this->predict();
    ndarray<T> y_pred_minus_y = y_pred - this->y;
    ndarray<T> y_pred_minus_y_square = y_pred_minus_y * y_pred_minus_y;
    ndarray<T> y_pred_minus_y_square_sum = y_pred_minus_y_square.sum(0);
    ndarray<T> y_pred_minus_y_square_sum_divide_2 = y_pred_minus_y_square_sum / 2;
    ndarray<T> y_pred_minus_y_square_sum_divide_2_divide_x_shape_0 = y_pred_minus_y_square_sum_divide_2 / this->x.shape()[0];
    ndarray<T> loss = y_pred_minus_y_square_sum_divide_2_divide_x_shape_0;
    this->loss = loss;
}

template<typename T>
ndarray<T> LogisticRegression<T>::predict(ndarray<T> x) {
    ndarray<T> x_dot_w = x.matMult(this->w);
    ndarray<T> x_dot_w_plus_b = x_dot_w + this->b.flatten()[0];
    ndarray<T> x_dot_w_plus_b_sigmoid = sigmoid(x_dot_w_plus_b);
    return x_dot_w_plus_b_sigmoid;
}

template<typename T>
ndarray<T> LogisticRegression<T>::sigmoid(ndarray<T> x) {
    return ((x * -1).exp() + 1).inv();
}


#endif
