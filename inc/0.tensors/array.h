#pragma once
#include "predefs.h"
#include "scalar.h"
#include <vector>
#include <initializer_list>
#include <algorithm>
#include <functional>
//#include "debug_util.h"

/*
Array<T> is basic array class, now it has std::vector<T>, in long run we could implement our own efficient one
*/
template <class T>
class Array
{
private:
    std::vector<T> _v;
protected:
    inline T get(uint index) const
    {
        assert(index < this->size());
        return _v[index];
    }

    inline void set(const T &e, uint index)
    {
        assert(index < this->size());
        _v[index] = e;
    }

public:
    Array(){}
    explicit Array(size_t t){
        _v.resize(t);
    }
    Array(std::initializer_list<T> l){
        if (l.size() > 0)
            _v = l;
    }
    /*Array(T* start, size_t size)
    {
        _v.assign(start, start + size);
    }*/

    virtual ~Array() {clear();} 

    // TODO: initial goal for this func is to allocate memory but not init
    // TODO: perf is not good
    inline void reserve(uint len)
    {
        clear();
        for (uint i = 0; i < len; ++i)
        {
            push_back(T());
        }
    }

    /***************************************************************************************************************/
    // set functions
    inline typename std::vector<T>::reference operator[](uint index)
    {
        if (index >= size())
        {
            assert(index < size());
        }
        return _v[index];
    }

    inline typename std::vector<T>::iterator begin()
    {
        return _v.begin();
    }

    inline typename std::vector<T>::iterator end()
    {
        return _v.end();
    }

    void set(uint a1_start, const Array<T> &a2, uint a2_start = 0, int len = -1)
    {
        if (len < 0)
            len = a2.size() - a2_start;

        assert(a1_start + len <= size() && a2_start + len <= a2.size());

        for (uint i = 0; i < len; ++i)
        {
            set(a2[i + a2_start], i + a1_start);
        }
    }

    void set_each(const T &e, uint start = 0, int len = -1)
    {
        if (len < 0)
            len = size() - start;

        assert(start + len <= size());
        for (uint i = 0; i < len; ++i)
        {
            set(e, i + start);
        }
    }

    Array<T> &copy(const Array<T> &a2, uint start = 0, int len = -1)
    {
        assert(start <= a2.size());
        if (len < 0) len = a2.size() - start;

        clear();
        reserve(len);
        set(0, a2, start, len);
        return *this;
    }

    inline Array<T>& push_back(const T &e)
    {
        _v.push_back(e);
        return *this;
    }

    inline void insert(uint start, const Array<T>& arr)
    {
        _v.insert(this->begin() + start, arr.begin(), arr.end());
    }

    inline Array<T>& insert(uint start, const T& e)
    {
        _v.insert(this->begin() + start, e);
        return *this;
    }

    inline Array<T> clone() const
    {
        Array<T> arr;
        arr.copy(*this);
        return arr;
    }

    void append(const Array<T> &a2, uint start = 0, int len = -1)
    {
        assert(start <= a2.size());
        if (len < 0) len = a2.size() - start;
        assert(start + len <= a2.size());

        for (uint i = 0; i < len; ++i)
        {
            T v;
            v = a2[i + start];

            push_back(v);
        }
    }

    inline virtual void clear()
    {
        _v.clear();
    }

    inline void erase(uint start, int len = -1)
    {
        if (len < 0)
        len = size() - start;

        _v.erase(this->begin() + start, this->begin() + start + len);
    }

    /***************************************************************************************************************/
    // get functions, all const

    inline T* ptr() 
    {
        return _v.data();
    }

    inline const T* ptr()  const
    {
        return _v.data();
    }

    inline const T &operator[](uint index) const
    {
        assert(index < size());
        return _v[index];
    }

    inline typename std::vector<T>::const_iterator begin() const
    {
        return _v.cbegin();
    }

    inline typename std::vector<T>::const_iterator end() const
    {
        return _v.cend();
    }

    inline T front() const
    {
        assert(size() > 0);
        return get(0);
    }

    inline T back() const
    {
        assert(size() > 0);
        return get(size() - 1);
    }

    inline Array<T> subset(uint start, int len = -1) const
    {
        if (len < 0) len = size() - start;
        assert(start + len <= size());
        Array<T> out;
        for (uint i = 0; i < len; ++i)
        {
            out.push_back(get(i + start));
        }

        return out;
    }

    inline bool contains(const T &t) const
    {
        return std::find(_v.begin(), _v.end(), t) != _v.end();
    }

    inline uint find(const T &t) const
    {
        auto iter = std::find(_v.begin(), _v.end(), t);
        return iter - _v.begin();
    }

    inline uint find(const std::function<bool(const T&)> &func) const
    {
        auto iter = std::find_if(_v.begin(), _v.end(), func);
        return iter - _v.begin();
    }

    // note: only support numeric T
    bool equals_to(const Array<T> &a2, T noise_level = 0.00001) const
    {
        if (size() != a2.size())
            return false;

        for (uint i = 0; i < size(); ++i)
        {
            if (get(i) == a2.get(i))
                continue;
            if (!Math<float>::almost_equal(get(i), a2.get(i), noise_level))
                return false;
        }

        return true;
    }

    inline bool operator==(const Array<T> &a2) const
    {
        return equals_to(a2);
    }

    inline bool operator!=(const Array<T> &a2) const
    {
        return !equals_to(a2);
    }

    inline uint size() const
    {
        return _v.size();
    }

    void print(std::ostream& os = std::cout) const
    {
        for (uint i = 0; i < size(); ++i)
        {
            os << get(i) << " ";
        }

        os << "\n";
    }

    /****************************************************************************************************************/
    /* below are array update functions */
    void swap(uint from, uint to)
    {
        assert(from < size() && to < size());
        T tmp = get(from);
        set(get(to), from);
        set(tmp, to);
    }

    // [0, 1, 2, 3, 4, 5], move_forward(3, 2, 1) => [0, 3, 4, 1, 2, 5]
    Array<T> &move_forward(uint move_from, uint move_len, uint move_to)
    {
        assert(move_len > 0);
        assert(move_from + move_len <= size());
        assert(move_from > move_to);

        Array<T> tmp(size());
        for (uint i = 0; i < move_to; ++i)
            tmp.set(get(i), i);
        for (uint i = 0; i < move_len; ++i)
            tmp.set(get(i + move_from), i + move_to);
        for (uint i = 0; i < move_from - move_to; ++i)
            tmp.set(get(i + move_to), i + move_to + move_len);
        for (uint i = 0; i < size() - move_from - move_len; ++i)
            tmp.set(get(i + move_from + move_len), i + move_from + move_len);

        copy(tmp);
        return *this;
    }

    void loop(const std::function<void(T)> &func, uint a1_start = 0, int len = -1)
    {
        if (len < 0)
            len = size() - a1_start;
        for (uint i = 0; i < len; ++i)
        {
            func(get(a1_start + i));
        }
    }
};

/*template <>
inline std::vector<bool>::reference Array<bool>::operator[](uint index)
{
    return _v[index];
}*/