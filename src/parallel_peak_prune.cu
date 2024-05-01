#include "parallel_peak_prune.h"
#include <cub/cub.cuh>
#include <cub/device/device_partition.cuh>
#include <vector>

__global__ 
void assign_init_label_kernel(PPPEdge* edges, PPPVertex* vertices, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        PPPEdge e = edges[index];
        int lower_end;
        int higher_end;
        if (vertices[e.v1].val > vertices[e.v2].val)
        {
            higher_end = e.v1;
            lower_end = e.v2;
        }
        else {
            higher_end = e.v2;
            lower_end = e.v1;
        }//todo what if vertices[e.v1].val == vertices[e.v2].val?

        //https://stackoverflow.com/questions/52848426/how-to-execute-atomic-write-in-cuda
        //We don't need atomic operation here
     
        vertices[lower_end].peak_label = higher_end;

        edges[index].v1 = lower_end;
        edges[index].v2 = higher_end;
    }
}

void assign_init_label(PPPEdge* d_edges, PPPVertex* d_vertices, int size)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    assign_init_label_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_edges, d_vertices,size);
    cudaDeviceSynchronize();
}

__global__
void pointer_jump_kernel1(const PPPVertex* vertices, int* peak_labels, int size)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        peak_labels[index] = vertices[index].peak_label;
    }
}

__global__
void pointer_jump_kernel2(PPPVertex* vertices, const int* peak_labels, int size, int* d_counter)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        int peak = peak_labels[index];
        if (peak != vertices[peak].peak_label)
        {
            vertices[index].peak_label = vertices[peak].peak_label;
            atomicAdd(d_counter, 1);
        }
    }
}

void pointer_jump(PPPVertex* d_vertices, int* d_aux_labels, int size, int* d_counter)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    // Read the preceder's label to labels buffer
    pointer_jump_kernel1 <<<blocksPerGrid, threadsPerBlock >>> (d_vertices,d_aux_labels, size);
    // Write preceder's label to vertex label, perform one pass pointer jump.
    // The separation of these two procedure is neccessary to ensure correct synchronization.
    cudaDeviceSynchronize();
    //std::vector<int> h_debug_peak_labels(size,-1);
    //cudaMemcpy(h_debug_peak_labels.data(), d_aux_labels, sizeof(int) * size, cudaMemcpyDeviceToHost);
    pointer_jump_kernel2<<<blocksPerGrid, threadsPerBlock>>>(d_vertices,d_aux_labels, size, d_counter);
    cudaDeviceSynchronize();
}

__global__
void compact_peak_label_kernel1(const PPPVertex* vertices, PPPLabelIndex* labelIdx, int size)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        labelIdx[index].label = vertices[index].peak_label;
        labelIdx[index].idx = index;
    }
}

__global__
void compact_peak_label_kernel2(const PPPLabelIndex* labelIdx,int* flags,int size)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        if (index != 0)
        {
            if (labelIdx[index - 1].label != labelIdx[index].label)
            {
                flags[index] = 1;
            }
            else {
                flags[index] = 0;
            }
        }
        else {
            flags[0] = 1;
        }
    }
}

__global__
void compact_peak_label_kernel3(PPPVertex* vertices, const PPPLabelIndex* labelIdx, const int* flags, int* compact_labels_map, int size)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        if (index == 0 || flags[index - 1] != flags[index])
        {
            // Store orginal peak_label to compacted label idx
            compact_labels_map[flags[index] - 1] = vertices[labelIdx[index].idx].peak_label;
        }
        vertices[labelIdx[index].idx].peak_label = flags[index] - 1;
    }
}

void count_peaks(PPPVertex* vertices, PPPLabelIndex* d_label_idx, int* d_flags1, int* d_flags2, int size, int* peak_count)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    compact_peak_label_kernel1 << <blocksPerGrid, threadsPerBlock >> > (vertices, d_label_idx, size);
    cudaDeviceSynchronize();
    // sort label idx
    PPPLabelIndexComp compOp;
    cub::DeviceMergeSort::SortKeys(nullptr, temp_storage_bytes, d_label_idx, size, compOp);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceMergeSort::SortKeys(d_temp_storage, temp_storage_bytes, d_label_idx, size, compOp);

    //std::vector<PPPLabelIndex> h_debug_labels(size);
    //cudaMemcpy(h_debug_labels.data(), d_label_idx, sizeof(PPPLabelIndex) * size, cudaMemcpyDeviceToHost);

    compact_peak_label_kernel2 << <blocksPerGrid, threadsPerBlock >> > (d_label_idx, d_flags1, size);
    cudaDeviceSynchronize();
    // inclusive scan on flags
    size_t temp_storage_bytes2 = 0;
    cub::DeviceScan::InclusiveSum(nullptr, temp_storage_bytes2, d_flags1, d_flags2, size);
    if (temp_storage_bytes2 > temp_storage_bytes)
    {
        //Need to reallocate temp storage
        cudaFree(d_temp_storage);
        cudaMalloc(&d_temp_storage, temp_storage_bytes2);
    }
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes2, d_flags1, d_flags2, size);

    //std::vector<int> h_debug_flags(size);
    //cudaMemcpy(h_debug_flags.data(), d_flags1, sizeof(int) * size, cudaMemcpyDeviceToHost);

    cudaMemcpy(peak_count, &d_flags2[size - 1], sizeof(int), cudaMemcpyDeviceToHost);
}

void assign_compact_peak_label(PPPVertex* vertices, PPPLabelIndex* d_label_idx, int* d_flags, int* d_compact_labels_map, int size)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    compact_peak_label_kernel3 << <blocksPerGrid, threadsPerBlock >> > (vertices, d_label_idx, d_flags, d_compact_labels_map, size);
    cudaDeviceSynchronize();
}

__global__
void identify_saddle_candidate_kernel(PPPEdge* edges, const PPPVertex* vertices, int* neighbor_labels, int* is_candidate, int size)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        const PPPEdge e = edges[index];
        int lower_end = e.v1;
        int higher_end = e.v2;
        // Remember the peak label here is compacted label
        int high_end_label = vertices[higher_end].peak_label;
        edges[index].label = high_end_label;
        int old = atomicExch(&neighbor_labels[lower_end], high_end_label);
        if (old != -1 && old != high_end_label)
        {
            is_candidate[lower_end] = 1;
        }
    }
}

void identify_saddle_candidate(PPPEdge* d_edges, const PPPVertex* d_vertices, int* d_neighbor_labels, int* is_candidate, int size)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    identify_saddle_candidate_kernel << <blocksPerGrid, threadsPerBlock >> > (d_edges, d_vertices, d_neighbor_labels, is_candidate, size);
    cudaDeviceSynchronize();
}

__global__
void identify_saddle_candidate_edge_kernel(PPPEdge* edges,const int* is_candidate_vertex, int* is_candidate_edge, int size)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        if (is_candidate_vertex[edges[index].v1] == 1)
        {
            is_candidate_edge[index] = 1;
        }
        else {
            is_candidate_edge[index] = 0;
        }
    }
}

void identify_saddle_candidate_edge(PPPEdge* d_edges, const int* is_candidate_vertex, int* is_candidate_edge, int size)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    identify_saddle_candidate_edge_kernel << <blocksPerGrid, threadsPerBlock >> > (d_edges,is_candidate_vertex,is_candidate_edge, size);
    cudaDeviceSynchronize();
}

void partition_saddle_candidate_edges(const PPPEdge* edges,PPPEdge* out_edges, int* is_candidate, int size, int * selected_out)
{
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DevicePartition::Flagged(nullptr, temp_storage_bytes, edges, is_candidate, out_edges, selected_out, size);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DevicePartition::Flagged(d_temp_storage, temp_storage_bytes, edges, is_candidate, out_edges, selected_out, size);
    cudaFree(d_temp_storage);
}

void sort_saddle_candidate_edges(PPPEdge* d_edges, const PPPVertex* d_vertices, int size)
{
    // sort candidate edges by (peak label, lower end)
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    PPPEdgeLess less_op(d_vertices);
    cub::DeviceMergeSort::SortKeys(nullptr, temp_storage_bytes, d_edges, size, less_op);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceMergeSort::SortKeys(d_temp_storage, temp_storage_bytes, d_edges, size, less_op);
    cudaFree(d_temp_storage);
}

__global__
void identify_saddle_kernel(const PPPEdge* d_edges, const PPPVertex* d_vertices, int* peak_saddle_pairs, int size, int* d_saddle_count)
{
    // Todo : can we make sure that each peak only has one governing saddle so that no write conflict would happen?
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        // Remember here the peak label is compacted label
        if (index == 0)
        {
            // The lower end of d_edges[0] is governing saddle point
            peak_saddle_pairs[d_edges[0].label] = d_edges[0].v1;;
            atomicAdd(d_saddle_count, 1);
        }
        else {
            if (d_edges[index - 1].label != d_edges[index].label)
            {
                // The lower end of d_edges[index] is governing saddle point
                peak_saddle_pairs[d_edges[index].label] = d_edges[index].v1;
                atomicAdd(d_saddle_count, 1);
            }
        }
    }
}

void identify_governing_saddle(const PPPEdge* d_candidate_edges, const PPPVertex* d_vertices, int* peak_saddle_pairs, int size, int* saddle_count)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    int* d_saddle_count;
    cudaMalloc((void**)&d_saddle_count, sizeof(int));
    cudaMemset(d_saddle_count, 0, sizeof(int));
    identify_saddle_kernel <<<blocksPerGrid, threadsPerBlock>>> (d_candidate_edges, d_vertices, peak_saddle_pairs, size, d_saddle_count);
    cudaDeviceSynchronize();
    cudaMemcpy(saddle_count, d_saddle_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_saddle_count);
}

__global__ 
void collect_saddle_pairs_kernel(const PPPVertex* d_vertices, const int* compacted_labels_map,const int* peak_saddle_pairs, PPPPeakSadlePair* pairs, int size, int offset)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        int peak_idx = compacted_labels_map[index];
        int saddle_idx = peak_saddle_pairs[index];
        pairs[offset + index].peak_id = d_vertices[peak_idx].identifier;
        pairs[offset + index].saddle_id = d_vertices[saddle_idx].identifier;
    }
}

void collect_saddle_pairs(const PPPVertex* d_vertices, const int* compacted_labels_map, const int* peak_saddle_pairs, PPPPeakSadlePair* pairs, int size, int offset)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    collect_saddle_pairs_kernel << <blocksPerGrid, threadsPerBlock >> > (d_vertices,compacted_labels_map,peak_saddle_pairs,pairs,size,offset);
    cudaDeviceSynchronize();
}

__global__
void mark_delete_vertices_kernel(const PPPVertex* d_vertices, const int* peak_saddle_pairs, int* vet_should_remain, int size)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        const auto v = d_vertices[index];
        if (v.val < d_vertices[peak_saddle_pairs[v.peak_label]].val || index == peak_saddle_pairs[v.peak_label])
        {
            vet_should_remain[index] = 1;
        }
        else {
            vet_should_remain[index] = 0;
        }
    }
}

__global__
void get_vet_reorder_map_kernel(const int * d_vet_remain_scan, const int* vet_should_remain, int* d_vet_reorder_map, int size)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        if (vet_should_remain[index] > 0)
        {
            d_vet_reorder_map[index] = d_vet_remain_scan[index] - 1;
        }
    }
}

__global__
void mark_delete_edges_kernel(const PPPEdge* d_edges,const int* vet_should_remain, int* edge_should_remain, int size)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        const auto e = d_edges[index];
        if (vet_should_remain[e.v1] > 0 || vet_should_remain[e.v2] > 0)
        {
            edge_should_remain[index] = 1;
        }
        else {
            edge_should_remain[index] = 0;
        }
    }
}

__global__
void edges_redirect_kernel(PPPEdge* d_edges, const int* vet_should_remain, 
                           const int* vet_new_idx_map, const int* peak_saddle_pairs, int size)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        auto e = d_edges[index];
        e.v1 = vet_new_idx_map[e.v1];
        if (vet_should_remain[e.v2] == 0)
        {
            // If the higher end of the edge is deleted, redirect it to the governing saddle
            e.v2 = vet_new_idx_map[peak_saddle_pairs[e.label]];
        }
        else {
            e.v2 = vet_new_idx_map[e.v2];
        }
        d_edges[index] = e;
    }
}

__global__
void transfer_leaves_step_kernel(MTNode* t1_nodes, MTNode* t2_nodes,
    const int* t1_leaves, const int t1_leaves_size, int* leaves_successful_proceed,
                            int* new_t1_leaves, int* new_t1_leaves_tail_pointer,
                            CTEdge* ct_edges, int* ct_edges_tail_pointer)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < t1_leaves_size) {
        int t1_leaf_id = t1_leaves[index]; // leaf index is the identifier for leafnodes of t1
        auto t1_leaf_node = t1_nodes[t1_leaf_id];
        if (t2_nodes[t1_leaf_id].degree <= 2)
        {
            // leaf can be removed
            t1_nodes[t1_leaf_id].is_deleted = true;
            // transfer the edge to contour tree
            ct_edges[atomicAdd(ct_edges_tail_pointer, 1)] = CTEdge{ t1_leaf_id,t1_leaf_node.parent_id };
            // also remove the vertex from another tree
            t2_nodes[t1_leaf_id].is_deleted = true;

            // Found new candidate leaves
            atomicAdd(&t1_nodes[t1_leaf_node.parent_id].degree, -1);
            new_t1_leaves[atomicAdd(new_t1_leaves_tail_pointer, 1)] = t1_leaf_node.parent_id;

            atomicAdd(leaves_successful_proceed, 1);
        }
    }
}

__global__
void merge_tree_edges_redirect_step_kernel(MTNode* nodes, int size, int * counter)
{
    //Todo : it's better to collect all the remain vertices(by partition)
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size && !nodes[index].is_deleted) {
        auto node = nodes[index];
        if (nodes[node.parent_id].is_deleted)
        {
            atomicAdd(counter, 1);
            // pointer jump
            nodes[index].parent_id = nodes[node.parent_id].parent_id;
        }
    }
}

// Gate functions

void mark_delete_region(PPPVertex* d_vertices, PPPEdge* d_edges,
                        const int* d_peak_saddle_pairs,
                        int* d_vet_should_remain, int* d_edge_should_remain,
                        int v_size, int e_size)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (v_size + threadsPerBlock - 1) / threadsPerBlock;

    mark_delete_vertices_kernel << <blocksPerGrid, threadsPerBlock >> > (d_vertices, d_peak_saddle_pairs, d_vet_should_remain, v_size);
    cudaDeviceSynchronize();

    blocksPerGrid = (e_size + threadsPerBlock - 1) / threadsPerBlock;
    mark_delete_edges_kernel << <blocksPerGrid, threadsPerBlock >> > (d_edges, d_vet_should_remain, d_edge_should_remain, e_size);
    cudaDeviceSynchronize();
}

__global__ 
void stable_partition_kernel(const PPPVertex* d_vertices, PPPVertex* d_new_vertices, 
                                   const int* d_vet_should_remain, const int* d_vet_remain_scan, int size)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        if (d_vet_should_remain[index] == 1)
        {
            d_new_vertices[d_vet_remain_scan[index] - 1] = d_vertices[index];
        }
    }
}

void flatten_vertices_and_edges(const PPPVertex* d_vertices, const PPPEdge* d_edges,
                                PPPVertex* d_new_vertices, PPPEdge* d_new_edges,
                                const int* d_vet_should_remain, const int* d_edge_should_remain,
                                int* d_vet_remain_scan, int* d_vet_reorder_map,
                                int v_size, int e_size, int* remain_v_size, int* remain_e_size)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (v_size + threadsPerBlock - 1) / threadsPerBlock;

    int* d_remain_v_size;
    int* d_remain_e_size;
    cudaMalloc((void**)&d_remain_v_size, sizeof(int));
    cudaMalloc((void**)&d_remain_e_size, sizeof(int));
    // Reorder the vertices, get vertex new index map, get the size of new vertices set
    // We have to assume cub::DevicePartition is stable
    // d_vet_reorder_map stores the target position for *remained* vertices after partition

    blocksPerGrid = (v_size + threadsPerBlock - 1) / threadsPerBlock;
    size_t temp_storage_bytes = 0;
    void** d_temp_storage;
    cub::DeviceScan::InclusiveSum(nullptr, temp_storage_bytes, d_vet_should_remain, d_vet_remain_scan, v_size);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_vet_should_remain, d_vet_remain_scan, v_size);
    cudaMemcpy(remain_v_size, &d_vet_remain_scan[v_size - 1], sizeof(int),cudaMemcpyDeviceToHost);

    stable_partition_kernel << <blocksPerGrid, threadsPerBlock >> > (d_vertices, d_new_vertices, d_vet_should_remain, d_vet_remain_scan, v_size);
    get_vet_reorder_map_kernel <<<blocksPerGrid, threadsPerBlock >>> (d_vet_remain_scan, d_vet_should_remain, d_vet_reorder_map, v_size);
    cudaDeviceSynchronize();

    // Reorder the edges, get the size of new edge set
    size_t temp_storage_bytes2 = 0;
    cub::DevicePartition::Flagged(nullptr, temp_storage_bytes2, d_edges, d_edge_should_remain, d_new_edges, d_remain_e_size, e_size);
    if (temp_storage_bytes2 > temp_storage_bytes)
    {
        cudaFree(d_temp_storage);
        cudaMalloc(&d_temp_storage, temp_storage_bytes2);
    }
    cub::DevicePartition::Flagged(d_temp_storage, temp_storage_bytes2, d_edges, d_edge_should_remain, d_new_edges, d_remain_e_size, e_size);
    cudaMemcpy(remain_e_size, d_remain_e_size, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_temp_storage);
    cudaFree(d_remain_v_size);
    cudaFree(d_remain_e_size);
}

void redirect_edges(PPPEdge* d_edges, int* d_vet_should_remain, int* d_vet_new_idx_map, const int* d_peak_saddle_pairs, int size)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    edges_redirect_kernel << <blocksPerGrid, threadsPerBlock >> > (d_edges, d_vet_should_remain, d_vet_new_idx_map, d_peak_saddle_pairs, size);
    cudaDeviceSynchronize();
}

void transfer_leaves_step(MTNode* t1_nodes, MTNode* t2_nodes,
    const int* t1_leaves, const int t1_leaves_size, int* leaves_successful_proceed,
    int* new_t1_leaves, int* new_t1_leaves_tail_pointer,
    CTEdge* ct_edges, int* ct_edges_tail_pointer)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (t1_leaves_size + threadsPerBlock - 1) / threadsPerBlock;
    transfer_leaves_step_kernel << <blocksPerGrid, threadsPerBlock >> > (t1_nodes,t2_nodes,
        t1_leaves, t1_leaves_size, leaves_successful_proceed,
        new_t1_leaves, new_t1_leaves_tail_pointer,
        ct_edges, ct_edges_tail_pointer);
    cudaDeviceSynchronize();
}

void identify_new_leaves()
{

}

void merge_tree_edges_redirect_step(MTNode* nodes, int size, int* counter)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (size+ threadsPerBlock - 1) / threadsPerBlock;
    merge_tree_edges_redirect_step_kernel << <blocksPerGrid, threadsPerBlock >> > (nodes,size,counter);
    cudaDeviceSynchronize();
}
