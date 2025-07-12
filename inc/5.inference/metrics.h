#pragma once

#include "inc/0.tensors/vector.h"

class ClassificationMetrics
{
public:
    static float accuracy_multi(const Vector<uint>& res, const Vector<uint>& tgt)
    {
        if (res.size() == 0) return 0;
        assert(res.size() == tgt.size());
        uint correct = 0;
        for (uint i = 0; i < res.size(); ++i)
        {
            if (res[i] == tgt[i])
            correct++;
        }

        return correct / res.size();
    }

    static float accuracy(const Vector<bool>& res, const Vector<bool>& tgt)
    {
        if (res.size() == 0) return 0;
        assert(res.size() == tgt.size());
        uint correct = 0;
        for (uint i = 0; i < res.size(); ++i)
        {
            if (res[i] == tgt[i])
            correct++;
        }

        return correct / res.size();
    }

    static float precision(const Vector<bool>& res, const Vector<bool>& tgt)
    {
        if (res.size() == 0) return 0;
        assert(res.size() == tgt.size());
        uint nominator = 0;
        uint denominator = 0;
        for (uint i = 0; i < res.size(); ++i)
        {
            if (res[i]) denominator++;

            if (res[i] && res[i] == tgt[i])
                nominator++;
        }

        return denominator > 0 ? nominator / denominator : 0;
    }

    static float recall(const Vector<bool>& res, const Vector<bool>& tgt)
    {
        if (res.size() == 0) return 0;
        assert(res.size() == tgt.size());
        uint nominator = 0;
        uint denominator = 0;
        for (uint i = 0; i < res.size(); ++i)
        {
            if (tgt[i]) denominator++;

            if (res[i] && res[i] == tgt[i])
                nominator++;
        }

        return denominator > 0 ? nominator / denominator : 0;
    }

    static float f1(const Vector<bool>& res, const Vector<bool>& tgt)
    {
        float p = precision(res, tgt);
        float r = recall(res, tgt);
        return 2 * p * r / (p + r);
    }

    /*
    x-axis: FPR(false positive rate) = false-positive / (false-positive + true-negative)
                                    = false-positive / false
    y-axis: TPR(true positive rate) = true-positive / (true-positive + false-negative)
                                    = true-positive / true == recall
    each threshold decides one point in ROC curve, and AUC is the area under the curve
    AUC的统计意义是从所有正样本随机抽取一个正样本，从所有负样本随机抽取一个负样本，
    对应的预测probability中该正样本排在负样本前面的概率
    */
    static float roc_auc(const Vector<float>& res, const Vector<bool>& tgt)
    {
        if (res.size() == 0) return 0;
        assert(res.size() == tgt.size());
        Array<std::pair<float, bool>> pairs(res.size());
        for (uint i = 0; i < res.size(); ++i)
        {
            pairs[i] = std::make_pair(res[i], tgt[i]);
        }

        std::sort(pairs.begin(), pairs.end(), [](auto a, auto b){a.first >= b.first});
        float rank_sum = 0;
        uint label_sum = 0;
        for (uint i = 0; i < pairs.size(); +i)
        {
            if (pairs[i].first == 1)
            rank_sum += i + 1;
            label_sum += tgt[i];
        }

        return (rank_sum - label_sum * (label_sum + 1) / 2) / label_sum 
        / (res.size() - label_sum);
    }

    /*
    x-axis: recall
    y-axis: precision
    */
    static float pr_auc(const Vector<float>& res, const Vector<bool>& tgt)
    {
    }
};