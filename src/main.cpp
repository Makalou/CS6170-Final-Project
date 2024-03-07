#include <iostream>
#include <random>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <unordered_map>

#include "stb_image_write.h"
#include "stb_image.h"

#include "utility.hpp"

void generate_and_store_random_img2D(const char* name, uint32_t width, uint32_t height)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distribution(0, 1000000);

    auto* pixel_data = new unsigned char[width * height * 4];
    memset(pixel_data, 0, width * height * 4);

    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            float rand = std::pow(float(j - height / 2), 2) + std::pow(float(i - width / 2), 2);//distribution(gen);
            char* rand_ptr = reinterpret_cast<char*>(&rand);
            auto idx1 = j * width * 4 + i * 4 + 0;
            auto idx2 = j * width * 4 + i * 4 + 1;
            auto idx3 = j * width * 4 + i * 4 + 2;
            auto idx4 = j * width * 4 + i * 4 + 3;

            pixel_data[idx1] = *rand_ptr;
            pixel_data[idx2] = *(rand_ptr+1);
            pixel_data[idx3] = *(rand_ptr+2);
            pixel_data[idx4] = *(rand_ptr+3);
        }
    }
    
    stbi_write_png(name, width, height, 4, pixel_data, 4 * width);
}

float* load_float32_img2D(const char* name, int * width, int * height)
{
    int channel = 0;
    auto* raw_data = stbi_load(name, width, height, &channel, 0);
    assert(raw_data != nullptr);
    assert(channel == 4);
    assert(*width > 0);
    assert(*height > 0);
    float* data = new float[(*width) * (*height)];
    for (int j = 0; j < *height; ++j) {
        for (int i = 0; i < *width; ++i) {
            unsigned char r = raw_data[j * (*width) * 4 + i * 4 + 0];
            unsigned char g = raw_data[j * (*width) * 4 + i * 4 + 1];
            unsigned char b = raw_data[j * (*width) * 4 + i * 4 + 2];
            unsigned char a = raw_data[j * (*width) * 4 + i * 4 + 3];
            float unpacked = unpack_ieee32_le(a, b, g, r);
            data[j * (*width) + i] = unpacked;
        }
    }
    stbi_image_free(raw_data);
    return data;
}

struct DisJointSet
{
    DisJointSet* parent;
    uint32_t rank = 0;
};

template<typename T>
struct UnionFind
{
    DisJointSet* make_set(T v)
    {
        auto it = components.find(v);
        if (it != components.end())
        {
            return it->second.get();
        }
        DisJointSet* newSet= new DisJointSet;
        newSet->parent = newSet;
        newSet->rank = 0;
        components.emplace(v, newSet);
        return newSet;
    }

    DisJointSet* find(DisJointSet* x)
    {
        // Use path compression
        if (x->parent != x)
        {
            x->parent = find(x->parent); // Path compression
        }
        return x->parent;
    }

    bool is_connected(DisJointSet* x, DisJointSet* y)
    {
        x = find(x);
        y = find(y);
        return x == y;
    }

    // Return false if no actual merge happen
    // Return true otherwise
    bool union_two(DisJointSet * x, DisJointSet * y)
    {
        DisJointSet* x_root = find(x);
        DisJointSet* y_root = find(y);

        if (x_root == y_root) {
            return false;
        }
        // Union by rank
        if (x_root->rank < y_root->rank) {
            x_root->parent = y_root;
        }
        else if (x_root->rank > y_root->rank) {
            y_root->parent = x_root;
        }
        else {
            y_root->parent = x_root;
            x_root->rank++;
        }
        return true;
    }

    std::unordered_map<T, std::unique_ptr<DisJointSet>> components;
};

template<typename T>
struct MergeTreeNode
{
    T val;
    std::vector <MergeTreeNode<T>> children;
};

bool is_neighbor(uint32_t index1, uint32_t index2, int width, int height)
{
    int y1 = index1 / width;
    int x1 = index1 % width;

    int y2 = index2 / width;
    int x2 = index2 % width;

    if(y1 == y2 && std::abs( x1 - x2) == 1)
    {
        return true;
    }

    if(x1 == x2 && std::abs( y1 - y2) == 1)
    {
        return true;
    }

    if ((x1 - x2) * (y1 - y2) == -1)
    {
        return true;
    }

    return false;
}

std::vector<uint32_t> get_neighbors(uint32_t index, int width, int height)
{
    int y = index / width;
    int x = index % width;

    std::vector<uint32_t> res;
    if(x > 0)
    {
        res.push_back(index - 1);
        if (y < height - 1)
        {
            res.push_back(index - 1 + width);
        }
    }

    if (x < width - 1)
    {
        res.push_back(index + 1);
        if (y < height - 1)
        {
            res.push_back(index - 1 + width);
        }
    }

    if (y > 0)
    {
        res.push_back(index - width);
    }

    if (y < height - 1)
    {
        res.push_back(index + width);
    }

    return res;
}

int main()
{
    //generate_and_store_random_img2D("paraboloid256x256.png", 256, 256);

    //return 0;
    int width, height;
    float* data = load_float32_img2D("random256x256.png",&width,&height);
    /*int width = 1024;
    int height = 1024;
    float* data = new float[width * height];
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            data[j * width + i]= std::pow(float(j - height / 2), 2) + std::pow(float(i - width / 2), 2);//distribution(gen);
        }
    }*/

    // Extract domain...
    // Build simplical complex upon a 2D image
    // In this case we guaranteed that critical points only exist on vertices point,
    // So it's sufficient to only check vertices
    uint32_t vertices_count = width * height;
    uint32_t * indices = new uint32_t[vertices_count];
    std::iota(indices,indices + vertices_count, 0);

    // Global sorting ascending order
    std::cout << "Pass 1 sorting\n";
    std::sort(indices, indices + vertices_count, [data](uint32_t l, uint32_t r) ->bool{
        return data[l] < data[r];
    });
    std::cout << "Pass 1 sorting finished\n";

    using MergeTreeNodeType = MergeTreeNode<uint32_t>;

    std::list<MergeTreeNodeType> current_critical_nodes;

    {
        UnionFind<uint32_t> unionfind1;

        // Build Join tree
        // Exam the vertices in ascending order
        for (int i = 0; i < vertices_count; i++)
        {
            uint32_t idx = indices[i];
            float v = data[idx];
            auto* cc = unionfind1.make_set(idx);

            MergeTreeNodeType new_critical_node;
            new_critical_node.val = idx;

            std::vector<typename std::list<MergeTreeNodeType>::iterator> merged_with;

            std::cout << current_critical_nodes.size() << std::endl;

            for (auto neigh : get_neighbors(idx, width, height))
            {
                unionfind1.union_two(unionfind1.make_set(neigh), cc);
            }

            for (auto critical_node = current_critical_nodes.begin(); critical_node != current_critical_nodes.end(); critical_node++)
            {
                if (unionfind1.is_connected(unionfind1.make_set(critical_node->val), cc))
                {
                    merged_with.push_back(critical_node);
                    new_critical_node.children.push_back(*critical_node);
                }
            }

            if (merged_with.empty())
            {
                //local extrema
                //printf("Found local extrema %f\n",v);
                current_critical_nodes.push_back(new_critical_node);
                continue;
            }
            else if (merged_with.size() > 1)
            {
                //saddle point
                printf("Found saddle %f\n", v);
                current_critical_nodes.push_back(new_critical_node);
                for (auto it : merged_with)
                {
                    current_critical_nodes.erase(it);
                }
                continue;
            }
        }
    }

    // Global sorting descending order
    std::cout << "Pass 2 sorting\n";
    std::sort(indices, indices + vertices_count, [data](uint32_t l, uint32_t r) ->bool {
        return data[l] > data[r];
    });
    std::cout << "Pass 2 sorting finished\n";
    // Build Split tree
    // Exam the vertices in descending order
    std::list<MergeTreeNodeType> current_critical_nodes2;

    {
        UnionFind<uint32_t> unionfind1;

        // Build Join tree
        // Exam the vertices in ascending order
        for (int i = 0; i < vertices_count; i++)
        {
            uint32_t idx = indices[i];
            float v = data[idx];
            auto* cc = unionfind1.make_set(idx);

            MergeTreeNodeType new_critical_node;
            new_critical_node.val = idx;

            std::vector<typename std::list<MergeTreeNodeType>::iterator> merged_with;

            std::cout << current_critical_nodes2.size() << std::endl;

            for (auto neigh : get_neighbors(idx, width, height))
            {
                unionfind1.union_two(unionfind1.make_set(neigh), cc);
            }

            for (auto critical_node = current_critical_nodes2.begin(); critical_node != current_critical_nodes2.end(); critical_node++)
            {
                if (unionfind1.is_connected(unionfind1.make_set(critical_node->val), cc))
                {
                    merged_with.push_back(critical_node);
                    new_critical_node.children.push_back(*critical_node);
                }
            }

            if (merged_with.empty())
            {
                //local extrema
                //printf("Found local extrema %f\n",v);
                current_critical_nodes2.push_back(new_critical_node);
                continue;
            }
            else if (merged_with.size() > 1)
            {
                //saddle point
                printf("Found saddle %f\n", v);
                current_critical_nodes2.push_back(new_critical_node);
                for (auto it : merged_with)
                {
                    current_critical_nodes2.erase(it);
                }
                continue;
            }
        }
    }
    
    // Build Contour tree

}