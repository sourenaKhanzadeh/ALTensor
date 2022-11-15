#ifndef DERIVATIVES_H
#define DERIVATIVES_H

#include <ndarray.h>

// Derivative of a function
template <typename T>
ndarray<T> deriv(ndarray<T> (*func)(ndarray<T>), ndarray<T> x, T h = 1e-4) {
    return func(x + h) - func(x - h) / (2 * h);
}

template<typename T>
ndarray<T> square(ndarray<T> x) {
    std::vector<T> x_data = x.toVector();
    std::vector<T> y_data(x_data.size());
    for (int i = 0; i < x_data.size(); i++) {
        y_data[i] = x_data[i] * x_data[i];
    }
    return ndarray<T>(x.shape(), y_data);
}

template<typename T>
ndarray<T> relu(ndarray<T> x) {
    std::vector<T> x_data = x.toVector();
    std::vector<T> y_data(x_data.size());
    for (int i = 0; i < x_data.size(); i++) {
        y_data[i] = x_data[i] > 0 ? x_data[i] : 0;
    }
    return ndarray<T>(x.shape(), y_data);
}







#endif