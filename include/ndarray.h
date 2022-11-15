#ifndef NDARRAY_H
#define NDARRAY_H

#include <vector>
#include <sstream>
#include <random>

template <typename T>
class NDArray {
    public:
        NDArray(std::vector<int> shape);
        NDArray(std::vector<int> shape, std::vector<T> data);
        ~NDArray();
        const T& operator[](const std::vector<int> index) const;
        NDArray<T> operator[](int index) const;
        // print the array
        friend std::ostream& operator<<(std::ostream& os, const NDArray<T>& arr) {
            os << arr.str(0);
            return os;
        }

        // equals
        NDArray<T>& operator=(const NDArray<T>& other);

        // set a value
        void set(const std::vector<int> index, T value);

        // add two arrays
        NDArray<T> operator+(const NDArray<T>& arr);
        // subtract two arrays
        NDArray<T> operator-(const NDArray<T>& arr);

        // multiply two arrays
        NDArray<T> operator*(const NDArray<T>& arr);

        // tensor product of two arrays
        NDArray<T> matMult(const NDArray<T>& arr);

        // tensor product of two arrays
        NDArray<T> tensProd(const NDArray<T>& arr);

        // expand the array by one dimension
        NDArray<T> expandDims(int axis);

        int size();
        int size(int dim);
        int rank();
        std::vector<int> shape();
        std::vector<int> strides();
        void reshape(std::vector<int> shape);
        void resize(std::vector<int> shape);
        void resize(int size);
        void resize(int size, int dim);
        void resize(int size, int dim, T value);
        void resize(int size, int dim, T value, bool copy);
        void resize(int size, int dim, bool copy);
        void resize(int size, bool copy);
        void resize(std::vector<int> shape, T value);
        void resize(std::vector<int> shape, T value, bool copy);
        void resize(std::vector<int> shape, bool copy);
        void resize(bool copy);
        void fill(T value);
        void fill(T value, bool copy);
        void fill(bool copy);
        void copy(NDArray<T> &other);

        // return a string representation of the array
        std::string str(int index) const;
        void random();
        void random(T min, T max);
        void transpose();
        void flatten();

    private:
        std::vector<T> data;
        std::vector<int> shape_;
        std::vector<int> strides_;
        int size_;
        int rank_;
};

// implementation
template <typename T>
NDArray<T>::NDArray(std::vector<int> shape) {
    shape_ = shape;
    rank_ = shape.size();
    strides_.resize(rank_);
    strides_[rank_ - 1] = 1;
    size_ = 1;
    for (int i = rank_ - 1; i > 0; i--) {
        strides_[i - 1] = strides_[i] * shape_[i];
        size_ *= shape_[i];
    }
    size_ *= shape_[0];
    data.resize(size_);
}

template <typename T>
NDArray<T>::NDArray(std::vector<int> shape, std::vector<T> data) {
    shape_ = shape;
    rank_ = shape.size();
    strides_.resize(rank_);
    strides_[rank_ - 1] = 1;
    size_ = 1;
    for (int i = rank_ - 1; i > 0; i--) {
        strides_[i - 1] = strides_[i] * shape_[i];
        size_ *= shape_[i];
    }
    size_ *= shape_[0];
    this->data = data;
    
}

template <typename T>
NDArray<T>::~NDArray() {
    data.clear();
    shape_.clear();
    strides_.clear();
}


template <typename T>
const T& NDArray<T>::operator[](const std::vector<int> index) const{
    int offset = 0;
    for (int i = 0; i < rank_; i++) {
        offset += index[i] * strides_[i];
    }
    return data[offset];
}

template <typename T>
NDArray<T> NDArray<T>::operator[](int index) const{
    std::vector<int> new_shape(shape_.begin() + 1, shape_.end());
    std::vector<T> new_data(data.begin() + index * strides_[0], data.begin() + (index + 1) * strides_[0]);
    return NDArray<T>(new_shape, new_data);
}

template <typename T>
NDArray<T>& NDArray<T>::operator=(const NDArray<T>& other) {
    this->data = other.data;
    this->shape_ = other.shape_;
    this->strides_ = other.strides_;
    this->size_ = other.size_;
    this->rank_ = other.rank_;
    return *this;
}


template <typename T>
NDArray<T> NDArray<T>::operator+(const NDArray<T>& arr) {
    // check if the shapes are the same
    if (shape_ != arr.shape_) {
        throw std::invalid_argument("Shapes are not the same");
    }
    std::vector<T> new_data(size_);
    for (int i = 0; i < size_; i++) {
        new_data[i] = data[i] + arr.data[i];
    }
    return NDArray<T>(shape_, new_data);
}

template <typename T>
NDArray<T> NDArray<T>::operator-(const NDArray<T>& arr) {
    // check if the shapes are the same
    if (shape_ != arr.shape_) {
        throw std::invalid_argument("Shapes are not the same");
    }
    std::vector<T> new_data(size_);
    for (int i = 0; i < size_; i++) {
        new_data[i] = data[i] - arr.data[i];
    }
    return NDArray<T>(shape_, new_data);
}

template <typename T>
NDArray<T> NDArray<T>::operator*(const NDArray<T>& arr) {
    // check if the shapes are the same
    if (shape_ != arr.shape_) {
        throw std::invalid_argument("Shapes are not the same");
    }
    std::vector<T> new_data(size_);
    for (int i = 0; i < size_; i++) {
        new_data[i] = data[i] * arr.data[i];
    }
    return NDArray<T>(shape_, new_data);
}

template <typename T>
void NDArray<T>::set(std::vector<int> index, T value) {
    int offset = 0;
    for (int i = 0; i < rank_; i++) {
        offset += index[i] * strides_[i];
    }
    data[offset] = value;
}

template <typename T>
NDArray<T> NDArray<T>::matMult(const NDArray<T>& arr) {
    // Matrix multiplication
    // check if the shapes make sense
    if (shape_[1] != arr.shape_[0]) {
        throw std::invalid_argument("Shapes are not compatible");
    }
    std::vector<int> new_shape = {shape_[0], arr.shape_[1]};
    std::vector<T> new_data(new_shape[0] * new_shape[1]);
    for (int i = 0; i < new_shape[0]; i++) {
        for (int j = 0; j < new_shape[1]; j++) {
            T sum = 0;
            for (int k = 0; k < shape_[1]; k++) {
                sum += (*this)[{i, k}] * arr[{k, j}];
            }
            new_data[i * new_shape[1] + j] = sum;
        }
    }
    return NDArray<T>(new_shape, new_data);
}

template <typename T>
NDArray<T> NDArray<T>::tensProd(const NDArray<T>& arr) {
    // Tensor product
    // recursively call the function
    if (rank_ == 1) {
        std::vector<int> new_shape = {shape_[0], arr.shape_[0]};
        std::vector<T> new_data(new_shape[0] * new_shape[1]);
        for (int i = 0; i < new_shape[0]; i++) {
            for (int j = 0; j < new_shape[1]; j++) {
                new_data[i * new_shape[1] + j] = data[i] * arr.data[j];
            }
        }
        return NDArray<T>(new_shape, new_data);
    }
    else {
        std::vector<int> new_shape = {shape_[0], arr.shape_[0]};
        std::vector<T> new_data(new_shape[0] * new_shape[1]);
        NDArray<T> temp = (*this)[0].tensProd(arr[0]);
        for (int i = 0; i < new_shape[0]; i++) {
            for (int j = 0; j < new_shape[1]; j++) {
                temp = (*this)[i].tensProd(arr[j]);
            }
        }
        return temp;
    }
}

template <typename T>
NDArray<T> NDArray<T>::expandDims(int axis) {
    // expand the dimension of the array
    std::vector<int> new_shape = shape_;
    new_shape.insert(new_shape.begin() + axis, 1);
    std::vector<T> new_data(size_);
    for (int i = 0; i < size_; i++) {
        new_data[i] = data[i];
    }
    return NDArray<T>(new_shape, new_data);
}


template <typename T>
std::string NDArray<T>::str(int index) const {
    // recursively print the array depending on the index
    std::stringstream str;
    if (rank_ == 1) {
        str << "[";
        // print data in the array until index
        for (int i = 0; i < data.size(); i++) {
            if(i == data.size() - 1) {
                str << data[i];
            }
            else {
                str << data[i] << ", ";
            }
        }
        str << "]";
    }
    else {
        str << "[";
        for (int i = 0; i < shape_[0]; i++) {
            if (i == shape_[0] - 1) {
                str << (*this)[i].str(index + 1);
            }
            else {
                str << (*this)[i].str(index + 1) << ", ";
            }
        }
        str << "]";
    }
    return str.str();
}

template <typename T>
int NDArray<T>::size() {
    return size_;
}

template <typename T>
int NDArray<T>::size(int dim) {
    return shape_[dim];
}

template <typename T>
int NDArray<T>::rank() {
    return rank_;
}

template <typename T>
std::vector<int> NDArray<T>::shape() {
    return shape_;
}

template <typename T>
std::vector<int> NDArray<T>::strides() {
    return strides_;
}

template <typename T>
void NDArray<T>::reshape(std::vector<int> shape) {
    if (shape.size() != rank_) {
        throw "Shape size does not match rank";
    }
    int size = 1;
    for (int i = 0; i < rank_; i++) {
        size *= shape[i];
    }
    if (size != size_) {
        throw "Shape size does not match array size";
    }
    shape_ = shape;
    strides_[rank_ - 1] = 1;
    for (int i = rank_ - 1; i > 0; i--) {
        strides_[i - 1] = strides_[i] * shape_[i];
    }
}

template <typename T>
void NDArray<T>::resize(std::vector<int> shape) {
    if (shape.size() != rank_) {
        throw "Shape size does not match rank";
    }
    int size = 1;
    for (int i = 0; i < rank_; i++) {
        size *= shape[i];
    }
    shape_ = shape;
    strides_[rank_ - 1] = 1;
    for (int i = rank_ - 1; i > 0; i--) {
        strides_[i - 1] = strides_[i] * shape_[i];
    }
    data.resize(size);
    size_ = size;
}

template <typename T>
void NDArray<T>::resize(int size) {
    data.resize(size);
    size_ = size;
}

template <typename T>
void NDArray<T>::resize(int size, int dim) {
    if (dim >= rank_) {
        throw "Dimension out of range";
    }
    shape_[dim] = size;
    strides_[rank_ - 1] = 1;
    for (int i = rank_ - 1; i > 0; i--) {
        strides_[i - 1] = strides_[i] * shape_[i];
    }
    data.resize(size);
    size_ = size;
}

template <typename T>
void NDArray<T>::resize(int size, int dim, T value) {
    if (dim >= rank_) {
        throw "Dimension out of range";
    }
    shape_[dim] = size;
    strides_[rank_ - 1] = 1;
    for (int i = rank_ - 1; i > 0; i--) {
        strides_[i - 1] = strides_[i] * shape_[i];
    }
    data.resize(size, value);
    size_ = size;
}

template <typename T>
void NDArray<T>::resize(int size, int dim, T value, bool copy) {
    if (dim >= rank_) {
        throw "Dimension out of range";
    }
    shape_[dim] = size;
    strides_[rank_ - 1] = 1;
    for (int i = rank_ - 1; i > 0; i--) {
        strides_[i - 1] = strides_[i] * shape_[i];
    }
    data.resize(size, value);
    size_ = size;
}

template <typename T>
void NDArray<T>::resize(int size, int dim, bool copy) {
    if (dim >= rank_) {
        throw "Dimension out of range";
    }
    shape_[dim] = size;
    strides_[rank_ - 1] = 1;
    for (int i = rank_ - 1; i > 0; i--) {
        strides_[i - 1] = strides_[i] * shape_[i];
    }
    data.resize(size);
    size_ = size;
}



template <typename T>
void NDArray<T>::resize(bool copy) {
    data.resize(size_);
}

template <typename T>
void NDArray<T>::resize(std::vector<int> shape, T value) {
    if (shape.size() != rank_) {
        throw "Shape size does not match rank";
    }
    int size = 1;
    for (int i = 0; i < rank_; i++) {
        size *= shape[i];
    }
    shape_ = shape;
    strides_[rank_ - 1] = 1;
    for (int i = rank_ - 1; i > 0; i--) {
        strides_[i - 1] = strides_[i] * shape_[i];
    }
    data.resize(size, value);
    size_ = size;
}

template <typename T>
void NDArray<T>::resize(std::vector<int> shape, T value, bool copy) {
    if (shape.size() != rank_) {
        throw "Shape size does not match rank";
    }
    int size = 1;
    for (int i = 0; i < rank_; i++) {
        size *= shape[i];
    }
    shape_ = shape;
    strides_[rank_ - 1] = 1;
    for (int i = rank_ - 1; i > 0; i--) {
        strides_[i - 1] = strides_[i] * shape_[i];
    }
    data.resize(size, value);
    size_ = size;
}

template <typename T>
void NDArray<T>::resize(std::vector<int> shape, bool copy) {
    if (shape.size() != rank_) {
        throw "Shape size does not match rank";
    }
    int size = 1;
    for (int i = 0; i < rank_; i++) {
        size *= shape[i];
    }
    shape_ = shape;
    strides_[rank_ - 1] = 1;
    for (int i = rank_ - 1; i > 0; i--) {
        strides_[i - 1] = strides_[i] * shape_[i];
    }
    data.resize(size);
    size_ = size;
}


template <typename T>
void NDArray<T>::resize(int size, bool copy) {
    data.resize(size);
}

template <typename T>
void NDArray<T>::fill(T value) {
    std::fill(data.begin(), data.end(), value);
}

template <typename T>
void NDArray<T>::fill(T value, bool copy) {
    std::fill(data.begin(), data.end(), value);
}

template <typename T>
void NDArray<T>::fill(bool copy) {
    std::fill(data.begin(), data.end(), 0);
}

template <typename T>
void NDArray<T>::copy(NDArray<T> &other) {
    if (other.size_ != size_) {
        throw "Size mismatch";
    }
    data = other.data;
}

template <typename T>
void NDArray<T>::flatten() {
    rank_ = 1;
    shape_ = {size_};
    strides_ = {1};
}

template <typename T>
void NDArray<T>::transpose() {
    // reverse the shape and strides
    std::reverse(shape_.begin(), shape_.end());
    std::reverse(strides_.begin(), strides_.end());

}

template <typename T>
void NDArray<T>::random() {
    // fill with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);
    for (int i = 0; i < size_; i++) {
        data[i] = dis(gen);
    }
}

template <typename T>
void NDArray<T>::random(T min, T max) {
    // fill with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(min, max);
    for (int i = 0; i < size_; i++) {
        data[i] = dis(gen);
    }
}

template <typename T>
using ndarray = NDArray<T>;

#endif