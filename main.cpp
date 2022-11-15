#include <iostream>
#include <deriv.h>
#include <ndarray.h>

int main() {

    NDArray<int> x({2, 2, 1});
    NDArray<int> y({2, 3, 1});

    y.fill(1);
    x.random(10, 20);
    x.set({1, 1, 0}, 100);

    std::cout << "x = " << x << std::endl;
    std::cout << x.tensProd(y) << std::endl;

    return 0;
}