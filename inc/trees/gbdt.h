#pragma once
#include "../0.tensors/tensor.h"
#include <queue>

class Gbdt
{
public:
    struct GbdtNode
    {
    public:
        int _feature_id = -1;
        double _threshold = NAN; // note: only supports binary tree now
        double _split_gain = 0;
        double _score = NAN;
        double _entropy = NAN;
        int _left_child = -1, _right_child = -1;
        uint _layer;
        GbdtNode(uint layer) : _layer(layer){}

        //bool is_leaf() const { return _left_child == -1 && _right_child == -1; }
    };

    struct GbdtTree
    {
    public:
        Array<GbdtNode> _nodes;
        int _root_node = -1;
    };

    struct GbdtEnsemble
    {
    public:
        Array<GbdtTree> _trees;
    };

    enum class GbdtTrainingTargets : uint
    {
        Regression, // not used
        // BinaryClassification, not supported yet
        // Rank
    };

    struct GbdtTrainingConfig
    {
    public:
        GbdtTrainingTargets _training_targets = GbdtTrainingTargets::Regression;
        uint _max_trees = 100;
        uint _max_leaves; // not used
        uint _max_layers = 5;
        uint _min_samples_per_leave = 100;
        uint _max_bins = 10;
        double _min_bin_size = 0.00001;
        double _feature_fraction = 1; // not implemented
        double _data_fraction = 1; // not implemented
        double _min_split_gain = 0.1;
        double _min_node_entropy = 0; // not used
        double _min_avg_residual = 0.1;
    };

public:
    bool train(const TensorDArray<double> training_data, const GbdtTrainingConfig &config, GbdtEnsemble &model)
    {
        // x is 2-dim feature array, z is 1-dim labels;
        assert(training_data.size() == 2);
        const TensorD<double> &x = training_data[0];
        const TensorD<double> &z = training_data[1];
        assert(x.shape() == 2); // note: don't consider batch so far
        assert(z.shape() == 1);
        assert(x.dim()[0] == z.dim()[0]);
        uint m = x.dim()[0]; // # of samples
        uint n = x.dim()[1]; // # of features

        model._trees.clear();     // make the output model to be empty
        TensorD<double> y({config._max_trees, m}, TensorInit_Types::Zero); // model outputs per tree, note: we may use 2-dim tensor
        TensorD<double> residual({config._max_trees, m});

        for (uint i = 0; i < config._max_trees; ++i)
        {   
            GbdtTree tree;
            bool success = build_tree(x, z, residual, config, tree, i);
            model._trees.push_back(tree);

            // inference this tree, and store model outputs to y[i]
            run_tree(tree, x, y, i);

            // note: to make it productive, we just need to use previous residual - current tree's y
            // note: we only need to store previous tree's residual, but store for every tree now for debugging purpose
            calc_residual(y, z, residual, i);
            double avg_residual = residual.vector().avg(i * m, m);
            if (avg_residual < config._min_avg_residual)
            {
                return true;
            }
        }

        return true;
    }

    void run_tree(const GbdtTree &tree, const TensorD<double> &x, TensorD<double> &y, uint tree_id)
    {
        assert(x.shape() == 2);
        uint m = x.dim()[0];
        uint y_start = m * tree_id;
        // run for every sample in x
        for (uint i = 0; i < m; ++i) // note: this could be in parrallel
        {
            y[y_start + i] += run_tree(tree, x, i);
        }
    }

    // note: shall we learn one complete boolean expression, seems tree is one special case
    // note: shall we pre-cache all the tree paths and speed up this?
    double run_tree(const GbdtTree &tree, const TensorD<double> &x, uint sample_id)
    {
        assert(x.shape() == 2 && sample_id < x.dim()[0]);
        uint n = x.dim()[1];
        uint curr_node = tree._root_node;
        uint parent_node = -1;
        uint x_start = sample_id * n;
        while (curr_node >= 0)
        {
            uint curr_feature_id = tree._nodes[curr_node]._feature_id;
            assert(curr_feature_id < n);
            // sample_id represents one sample, sample_id * n is start offset of current feature
            double feature_value = x.vector()[x_start + curr_feature_id];
            // treat == as left child move
            parent_node = curr_node;
            if (feature_value <= tree._nodes[curr_node]._threshold)
            {
                curr_node = tree._nodes[curr_node]._left_child;
            }
            else
            {
                curr_node = tree._nodes[curr_node]._right_child;
            }
        }

        if (parent_node == -1)
            return 0;
        else
            return tree._nodes[parent_node]._score;
    }

    void calc_residual(const TensorD<double>& y, const TensorD<double> &z, TensorD<double> &residual, uint tree_id)
    {
        assert(y.shape() == 2 && y.dim().equals_to(residual.dim()));
        assert(z.shape() == 1 && z.dim()[0] == y.dim()[1]);
        assert(tree_id < y.dim()[0]);
        uint m = residual.dim()[1];
        uint residual_start = m * tree_id;
        uint y_start = m * tree_id;

        // seems below is gradient of regression loss, other task may be different
        if (tree_id == 0) // residual[0] = z - y[0];
        {
            z.vector().add(y.vector(), residual.vector(), 0, 0, 0, m, false, 1, -1, 0);
        }
        else // residual[i] = residual[i-1] - y[i];
        {
            uint residual_last_start = m * (tree_id - 1);
            residual.vector().add(y.vector(), residual.vector(), residual_last_start, y_start, residual_start, m, false, 1, -1, 0);
        }
    }

    bool build_tree(const TensorD<double>& x, const TensorD<double>& z, const TensorD<double>& residual, const GbdtTrainingConfig& config, GbdtTree& tree, uint tree_id)
    {
        uint m = x.dim()[0];
        uint n = x.dim()[1];

        TensorD<uint> sorts;
        // TODO: we can pre-sort all features across all samples first, so that no need to rerun sorts
        //pre_sort(x, sorts);

        std::queue<std::pair<Vector<uint>, uint>> q;
        Vector<uint> samples(x.dim()[0], TensorInit_Types::Ordinal);

        tree._nodes.push_back(GbdtNode(0));
        q.push(std::make_pair(samples, 0));
        while (!q.empty())
        {
            auto pair = q.front();
            q.pop();
            samples = pair.first; uint node_id = pair.second;


            // prepare labels for selected samples
            Vector<double> curr_labels;
            for (uint sample_id = 0; sample_id < samples.size(); ++sample_id)
            {
                double label;
                if (tree_id == 0)
                    label = z.vector()[sample_id];
                else
                    label = residual.vector()[tree_id * m + sample_id];
                curr_labels.push_back(label);
            }

            Vector<uint> left_samples, right_samples;
            bool success = split_node(samples, x, curr_labels, config, tree._nodes[node_id], left_samples, right_samples);

            if (success && tree._nodes[node_id]._layer < config._max_layers - 1) // this means node is not leaf node, below is to create 2 new child nodes
            {
                tree._nodes.push_back(GbdtNode(tree._nodes[node_id]._layer + 1)); // left child
                tree._nodes.push_back(GbdtNode(tree._nodes[node_id]._layer + 1)); // right child
                tree._nodes[node_id]._left_child = tree._nodes.size() - 2;
                tree._nodes[node_id]._right_child = tree._nodes.size() - 1;

                q.push(std::make_pair(left_samples, tree._nodes.size() - 2));
                q.push(std::make_pair(left_samples, tree._nodes.size() - 1));
            }
        }

        return true;
    }

    bool split_node(const Vector<uint>& samples, const TensorD<double>& x, const Vector<double>& curr_labels, 
        const GbdtTrainingConfig& config, GbdtNode& node, Vector<uint>& left_samples, Vector<uint>& right_samples)
    {
        uint m = x.dim()[0];
        uint n = x.dim()[1];

        node._entropy = curr_labels.var(); // this is for regression loss only
        if (samples.size() < config._min_samples_per_leave) // no need to split furthur
        {
            node._score = calc_node_score(curr_labels);
            return false;
        }

        double min_entropy = 1e300;
        uint best_feature_id = -1;
        double best_threshold;
        for (uint feature_id = 0; feature_id < n; feature_id++)
        {
            // to speed up, cache features into one continous memory
            Vector<double> curr_features;
            //Vector<uint> curr_sorts;
            double min_feature, max_feature;
            for (uint samples_id = 0; samples_id < samples.size(); ++samples_id)
            {
                uint sample_id = samples[samples_id];
                assert(sample_id < m);
                double feature_value = x.vector()[sample_id * n + feature_id];
                curr_features.push_back(feature_value);
                //curr_sorts.push_back(sorts.vector()[sample_id * n + feature_id]);
                if (samples_id == 0)
                {
                    max_feature = min_feature = feature_value;
                }
                else
                {
                    if (feature_value < min_feature)
                    {
                        min_feature = feature_value;
                    }
                    else
                    {
                        max_feature = feature_value;
                    }
                }
            }

            double best_feature_threshold, min_feature_entropy;
            bool success = find_threshold(curr_features, curr_labels, config._max_bins, config._min_bin_size, min_feature, 
                max_feature, best_feature_threshold, min_feature_entropy);
            if (success && min_feature_entropy < min_entropy)
            {
                min_entropy = min_feature_entropy;
                best_feature_id = feature_id;
                best_threshold = best_feature_threshold;
            }
        }

        if (node._entropy - min_entropy >= config._min_split_gain)
        {
            node._feature_id = best_feature_id;
            node._threshold = best_threshold;
            node._split_gain = node._entropy - min_entropy;

            // note: do samples split here to save perf, cost is to re-build features vector once
            left_samples.clear();
            right_samples.clear();

            for (uint samples_id = 0; samples_id < samples.size(); ++samples_id)
            {
                uint sample_id = samples[samples_id];
                double feature_value = x.vector()[sample_id * n + best_feature_id];
                if (feature_value <= best_threshold)
                {
                    left_samples.push_back(sample_id);
                }
                else
                {
                    right_samples.push_back(sample_id);
                }
            }

            return true;
        }
        else
        {
            node._score = calc_node_score(curr_labels);
            return false;
        }
    }

    double calc_node_score(const Vector<double>& labels)
    {
        return labels.avg(); // note: this is for regression loss only
    }

    // TODO: draw histogram/distribution of features, and don't make bins same size
    // TODO: left child histogram could build from parent histogram?
    // TODO: right child histogram could be parent histogram - left child histogram?
    bool find_threshold(const Vector<double>& features, const Vector<double>& labels, uint max_bins, double min_bin_size,
        double min_feature, double max_feature, double& best_threshold, double& min_feature_entropy)
    {
        // build histogram of this features vector, may fail if all features are the same
        Vector<Vector<double>> bins_vector; // each bin stores the vector of labels
        build_histogram(features, labels, max_bins, min_bin_size, min_feature, max_feature, bins_vector);
        if (bins_vector.size() <= 1) // no need to split
        {
            return false;
        }
        
        // flatten this bins_vector and stores the size of each so that we can use one continuous vector to calculate variance
        Vector<double> flatten_vector; 
        Vector<double>::flatten(bins_vector, flatten_vector);

        // loop histogram to calculate feature gains per each bin
        double bin_size = (max_feature - min_feature) / bins_vector.size();
        uint left_size = 0, right_size = flatten_vector.size();
        for (uint i = 1; i < bins_vector.size(); ++i)
        {
            if (bins_vector[i - 1].size() == 0) // no need to process empty bin
                continue;
            
            left_size += bins_vector[i - 1].size();
            right_size -= bins_vector[i - 1].size();
            assert(left_size > 0 && right_size > 0);

            double feature_entropy = calc_entropy(flatten_vector, left_size);
            double threshold = min_feature + bin_size * i;
            if (i == 1)
            {
                min_feature_entropy = feature_entropy;
                best_threshold = threshold;
            }
            else if (feature_entropy < min_feature_entropy)
            {
                min_feature_entropy = feature_entropy;
                best_threshold = threshold;
            }
        }

        return true;
    }

    // can we pre-calculate each bin's variance and then do speed up? => seems not
    // TODO: for binary classification or ranking loss, we may need different feature gain function, so far only use variance
    double calc_entropy(const Vector<double>& flatten_vector, uint left_size)
    {
        double left_variance = flatten_vector.var(0, left_size);
        double right_variance = flatten_vector.var(left_size, flatten_vector.size() - left_size);
        return left_variance + right_variance;
    }

    // TODO: we shall use global data to do histogram, instead of each selected subsets of samples
    void build_histogram(const Vector<double>& features, const Vector<double>& labels, uint max_bins, double min_bin_size, double min_feature, double max_feature, Vector<Vector<double>>& bins_vector)
    {
        assert(max_bins >= 1);
        uint bins = max_bins;
        // this means we can't create max_bins as too many features are very similar
        if ((max_feature - min_feature) / min_bin_size < max_bins)
        {
            bins = (max_feature - min_feature) / min_bin_size;
        }

        if (bins > features.size()) // bins should not be more than samples count
        {
            bins = features.size();
        }

        if (bins == 0)
        {
            bins = 1;
        }

        double bin_size = (max_feature - min_feature) / bins;

        bins_vector.reserve(bins);
        for (uint i = 0; i < features.size(); ++i)
        {
            uint bin_id = (uint)((features[i] - min_feature) / bin_size);
            bins_vector[bin_id].push_back(labels[i]);
        }
    }

    TensorD<double>& inference(const TensorD<double> &x, TensorD<double>& y, const GbdtEnsemble& model)
    {
        assert(x.shape() == 2);
        uint m = x.dim()[0];
        uint n = x.dim()[1];
        y.reset({m}, TensorInit_Types::Zero);

        for (uint i = 0; i < model._trees.size(); ++i)
        {
            run_tree(model._trees[i], x, y, 0);
        }

        return y;
    }
};