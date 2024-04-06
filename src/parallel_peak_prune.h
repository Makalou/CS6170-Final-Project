#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

struct PPPVertex
{
    float val;
    int peak_label;
};

struct PPPEdge
{
    int v1;
    int v2;
    int label;
};

struct PPPLabelIndex
{
    int label;
    int idx;
};

struct PPPLabelIndexComp
{
    __device__ bool operator()(const PPPLabelIndex& lhs, const PPPLabelIndex& rhs)
    {
        return lhs.label > rhs.label;
    }
};

struct PPPEdgeLess
{
    const PPPVertex* vertices;
    PPPEdgeLess(const PPPVertex* vert) : vertices(vert) {}

    __device__ bool operator()(const PPPEdge& lhs, const PPPEdge& rhs)
    {
        if (lhs.label == rhs.label)
        {
            // Sort it into descending order
            // We guarantee now v1 is the lower end of edge
            return vertices[lhs.v1].val > vertices[rhs.v1].val;
        }
        else {
            return lhs.label < rhs.label;
        }
    }
};

void assign_init_label(PPPEdge* edges, PPPVertex* vertices, int size);

void pointer_jump(PPPVertex* d_vertices, int* d_aux_labels, int size, int * d_counter);

void count_peaks(PPPVertex* vertices, PPPLabelIndex* d_label_idx, int* d_flags1, int* d_flags2, int size, int* peak_count);

void assign_compact_peak_label(PPPVertex* vertices, PPPLabelIndex* d_label_idx, int* d_flags1, int* d_compact_labels_map, int size);

void identify_saddle_candidate(PPPEdge* d_edges, const PPPVertex* d_vertices, int* d_neighbor_labels, int* is_candidate, int size);

void partition_saddle_candidate_edges(const PPPEdge* d_edges, PPPEdge* out_edges, int* is_candidate, int size, int* selected_out);

void sort_saddle_candidate_edges(PPPEdge* d_edges, const PPPVertex* d_vertices,int size);

void identify_governing_saddle(const PPPEdge* d_candidate_edges, const PPPVertex* d_vertices, int* peak_saddle_pairs, int size, int * saddle_count);

void mark_delete_region(PPPVertex* d_vertices, PPPEdge* d_edges, const int* d_peak_saddle_pairs, int* d_vet_should_remain, int* d_edge_should_remain, int v_size, int e_size);

void flatten_vertices_and_edges(const PPPVertex* d_vertices, const PPPEdge* d_edges,
    PPPVertex* d_new_vertices, PPPEdge* d_new_edges,
    const int* d_vet_should_remain, const int* d_edge_should_remain,
    int* d_vet_remain_scan, int* d_vet_reorder_map,
    int v_size, int e_size, int* remain_v_size, int* remain_e_size);

void redirect_edges(PPPEdge* d_edges, int* d_vet_should_remain, int* d_vet_new_idx_map, const int* d_peak_saddle_pairs, int size);