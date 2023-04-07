#pragma once

#include "../0.tensors/vector.h"

class Metrics
{
public:
    double accuracy(const Vector<double>& y, const Vector<double>& z) const
    {
        assert(y.size() == z.size() && y.size() > 0);
        double correct = 0;
        for (uint i = 0; i < y.size(); ++i)
        {
            if (y[i] == z[i])
                ++correct;
        }
        return correct / y.size();
    }

    double precision(const Vector<bool>& y, const Vector<bool>& z) const
    {
        assert(y.size() == z.size() && y.size() > 0);
        double tp = 0;
        double fp = 0;
        for (uint i = 0; i < y.size(); ++i)
        {
            if (y[i] == 1 && z[i] == 1)
                ++tp;
            else if (y[i] == 1 && z[i] == 0)
                ++fp;
        }
        return tp + fp > 0 ? tp / (tp + fp) : 0;
    }   

    double recall(const Vector<bool>& y, const Vector<bool>& z) const
    {
        assert(y.size() == z.size() && y.size() > 0);
        double tp = 0;
        double fn = 0;
        for (uint i = 0; i < y.size(); ++i)
        {
            if (y[i] == 1 && z[i] == 1)
                ++tp;
            else if (y[i] == 0 && z[i] == 1)
                ++fn;
        }
        return tp + fn ? tp / (tp + fn) : 0;
    }   

    double f1(const Vector<bool>& y, const Vector<bool>& z) const
    {
        double p = precision(y, z);
        double r = recall(y, z);
        return p + r > 0 ? 2 * p * r / (p + r) : 0;
    }

    double pr_auc(const Vector<double>& y, const Vector<double>& z) const
    {
        assert(y.size() == z.size() && y.size() > 0);
        Vector<double> y_sorted = y;
        Vector<double> z_sorted = z;
        y_sorted.sort(z_sorted);
        double auc = 0;
        double tp = 0;
        double fp = 0;
        for (uint i = 0; i < y_sorted.size(); ++i)
        {
            if (z_sorted[i] == 1)
                ++tp;
            else
                ++fp;
            auc += tp / y_sorted.size() / (tp + fp);
        }
        return auc;
    }   
};