#include <iostream>
#include <random>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <unordered_map>

#include "stb_image_write.h"
#include "stb_image.h"

#include "utility.hpp"

#include <glad/glad.h>
#include "GLFW/glfw3.h"

#include "parallel_peak_prune.h"

#if _WIN32
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_PI_2
#define M_PI_2 1.57079632679489661923
#endif
#endif

const int DEFAULT_WINDOW_WIDTH = 1600;
const int DEFAULT_WINDOW_HEIGHT = 1200;

#pragma region GL_RESOURCE
struct GLVertex
{
    float vx, vy, vz;
};

GLuint terrainVBO;
GLuint terrainIBO;
GLuint terrainVAO;
GLuint terrainShaderProgram;
GLuint terrainVertexShader;
GLuint terrainFragmentShader;
#pragma endregion

struct Camera
{
public:

    void setPos(float x, float y, float z)
    {
        position.x = x; position.y = y; position.z = z;
        front = normalize(look_at - position);
        updateViewMatrix();
    }

    void setPos(const vec3<float> pos)
    {
        setPos(pos.x, pos.y, pos.z);
    }

    auto getPos()
    {
        return position;
    }

    void setLookAt(float x, float y, float z)
    {
        look_at.x = x; look_at.y = y; look_at.z = z;
        front = normalize(look_at - position);
        updateViewMatrix();
    }

    void setFront(float x, float y, float z)
    {
        front.x = x; front.y = y; front.z = z;
        updateViewMatrix();
    }

    auto getFront()
    {
        return front;
    }

    void setAspect(float asp)
    {
        aspect = asp;
        updateProjectionMatrix();
    }

    Camera()
    {
        position = { 1.0,0.0,0.0 };
        look_at = { 0.0,0.0,0.0 };
        updateViewMatrix();
        updateProjectionMatrix();
    }

    //recompute view matrix
    void updateViewMatrix()
    {
        vec3<float> up{ 0.0f, 1.0f, 0.0f };

        // here is a problem : up and front should not be co-linear!
        auto right = normalize(cross(up, front));

        up = cross(front, right);

        view_mat[0] = right.x; view_mat[1] = up.x; view_mat[2] = -front.x, view_mat[3] = 0.0f;
        view_mat[4] = right.y; view_mat[5] = up.y; view_mat[6] = -front.y, view_mat[7] = 0.0f;
        view_mat[8] = right.z; view_mat[9] = up.z; view_mat[10] = -front.z, view_mat[11] = 0.0f;
        view_mat[12] = -dot(right, position); view_mat[13] = -dot(up, position); view_mat[14] = dot(front, position), view_mat[15] = 1.0f;
    }

    //recompute projection matrix
    void updateProjectionMatrix()
    {
        float fov_rad = (fov_deg * M_PI) / 180.0f;
        float t = 1.0f / std::tan(fov_rad * 0.5f);
        float inv_frustum_dis = 1.0f / (far_plane - near_plane);
        proj_mat[0] = t * 1.0f / aspect; proj_mat[1] = 0; proj_mat[2] = 0; proj_mat[3] = 0;
        proj_mat[4] = 0;          proj_mat[5] = t; proj_mat[6] = 0; proj_mat[7] = 0;
        proj_mat[8] = 0;          proj_mat[9] = 0; proj_mat[10] = -(far_plane + near_plane) * inv_frustum_dis; proj_mat[11] = -1;
        proj_mat[12] = 0;         proj_mat[13] = 0; proj_mat[14] = -2 * far_plane * near_plane * inv_frustum_dis; proj_mat[15] = 0;
    }

    const float* getViewMatrix() const
    {
        return view_mat;
    }

    const float* getProjectionMatrix() const
    {
        return proj_mat;
    }

private:
    vec3<float> position;
    vec3<float> look_at;
    vec3<float> front;

    float view_mat[16]{};
    float proj_mat[16]{};

    //params for perspective camera
    float fov_deg = 30.f;
    float aspect = float(DEFAULT_WINDOW_WIDTH) / float(DEFAULT_WINDOW_HEIGHT);
    float far_plane = 10000.0f;
    float near_plane = 0.01f;
};

Camera main_cam;
float cam_distance = 10.0f;
//radians
float cam_theta = 0.0f;
float cam_phi = 0.0f;

#pragma region GL_Helper_Functions
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) 
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        // When user press ESC, close the window.
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
}

void frameBufferResizeCallBack(GLFWwindow* window, int width, int height) 
{

}

bool left_pressing = false;
bool right_pressing = false;
double lastCamX, lastCamY;

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
        left_pressing = true;
        glfwGetCursorPos(window, &lastCamX, &lastCamY);
    }
    else if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE) {
        left_pressing = false;
    }

    if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS) {
        right_pressing = true;
        glfwGetCursorPos(window, &lastCamX, &lastCamY);
    }
    else if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_RELEASE) {
        right_pressing = false;
    }
}

vec3<float> getCamPosition(float distance, float phi, float theta)
{
    float x = distance * std::cos(phi) * std::cos(theta);
    float y = distance * std::sin(phi);
    float z = distance * std::cos(phi) * std::sin(theta);
    return vec3<float>{ x, y, z };
}

void cursorPosCallback(GLFWwindow* window, double xpos, double ypos) 
{
    if (left_pressing) {
        double offsetX = xpos - lastCamX;
        double offsetY = lastCamY - ypos;

        cam_phi += offsetY * 0.01f;
        cam_theta += offsetX * 0.01f;

        // Avoid up vector flip
        cam_phi = std::max(std::min((float)(M_PI_2) * 0.999f, cam_phi), (float)(-M_PI_2) * 0.999f);

        lastCamX = xpos;
        lastCamY = ypos;

        main_cam.setPos(getCamPosition(cam_distance, cam_phi, cam_theta));
    }

    if (right_pressing)
    {
        double offsetX = xpos - lastCamX;
        double offsetY = lastCamY - ypos;

        cam_distance += offsetY * 0.01f;

        lastCamX = xpos;
        lastCamY = ypos;
        main_cam.setPos(getCamPosition(cam_distance, cam_phi, cam_theta));
    }
}

void updateMatrix4fv(const char* name, const float* matrix_ptr)
{
    GLuint loc = glGetUniformLocation(terrainShaderProgram, name);
    glUniformMatrix4fv(loc, 1, GL_FALSE, matrix_ptr);
}

void updateMatrix4fv(const char* name, int count, const float* matrix_ptr)
{
    GLuint loc = glGetUniformLocation(terrainShaderProgram, name);
    glUniformMatrix4fv(loc, count, GL_FALSE, matrix_ptr);
}

void updateVec3fv(const char* name, const float* vec_ptr)
{
    GLuint loc = glGetUniformLocation(terrainShaderProgram, name);
    glUniform3fv(loc, 1, vec_ptr);
}

void updateUint(const char* name, unsigned int val)
{
    GLuint loc = glGetUniformLocation(terrainShaderProgram, name);
    glUniform1ui(loc, val);
}
#pragma endregion

void generate_and_store_random_img2D(const char* name, uint32_t width, uint32_t height)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distribution(0, 100);

    auto* pixel_data = new unsigned char[width * height * 4];
    memset(pixel_data, 0, width * height * 4);

    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            float rand = distribution(gen);//std::pow(float(j - height / 2), 2) + std::pow(float(i - width / 2), 2);//distribution(gen);
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
    //generate_and_store_random_img2D("random4096x4096_scaled.png", 4096, 4096);
    //return 0;
    int width, height;
    float* data = load_float32_img2D("random128x128_scaled.png", &width, &height);
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
    uint32_t* indices = new uint32_t[vertices_count];
    std::iota(indices, indices + vertices_count, 0);

    std::vector<PPPVertex> ppp_vertices;
    for (int i = 0; i < vertices_count; i++)
    {
        PPPVertex v;
        v.peak_label = i;
        v.val = data[i];
        ppp_vertices.push_back(v);
    }

    std::vector<PPPEdge> ppp_edges;
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width - 1; j++)
        {
            int start = i * width;
            ppp_edges.push_back({ start + j, start + j + 1 });
        }
    }

    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < height - 1; j++)
        {
            int start = i + j * width;
            ppp_edges.push_back({ start, start + width });
        }
    }

    for (int i = 0; i + width + 1 < vertices_count; i++)
    {
        ppp_edges.push_back({ i,i + width + 1 });
    }

    int* unified_counter;
    cudaMallocManaged(&unified_counter, 1);
    *unified_counter = 0;

    PPPVertex* d_ppp_vertices;
    PPPEdge* d_ppp_edges;
    int* d_aux_labels;

    CUDA_CHECK(cudaMalloc((void**)&d_ppp_vertices,ppp_vertices.size() * sizeof(PPPVertex)));
    CUDA_CHECK(cudaMalloc((void**)&d_ppp_edges,ppp_edges.size() * sizeof(PPPEdge)));
    CUDA_CHECK(cudaMalloc((void**)&d_aux_labels, ppp_vertices.size() * sizeof(int)));

    cudaMemcpy(d_ppp_vertices, ppp_vertices.data(), ppp_vertices.size() * sizeof(PPPVertex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ppp_edges, ppp_edges.data(), ppp_edges.size() * sizeof(PPPEdge), cudaMemcpyHostToDevice);

    assign_init_label(d_ppp_edges, d_ppp_vertices, ppp_edges.size());

    pointer_jump(d_ppp_vertices, d_aux_labels, ppp_vertices.size(), unified_counter);

    while (*unified_counter != 0)
    {
        *unified_counter = 0;
        pointer_jump(d_ppp_vertices, d_aux_labels, ppp_vertices.size(), unified_counter);
        printf("%d\n", *unified_counter);
    }

    int* d_neighbor_lables;
    int* d_is_saddle_candidate;

    CUDA_CHECK(cudaMalloc((void**)&d_neighbor_lables,ppp_vertices.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_is_saddle_candidate, ppp_vertices.size() * sizeof(int)));
    cudaMemset(d_is_saddle_candidate, 0, ppp_vertices.size() * sizeof(int));
    cudaMemset(d_neighbor_lables, -1, ppp_vertices.size() * sizeof(int));
    identify_saddle_candidate(d_ppp_edges, d_ppp_vertices, d_neighbor_lables, d_is_saddle_candidate, ppp_edges.size());
    //partition all the edges with candidate saddle point as the lower end
    int* d_num_selected_out = NULL;
    cudaMalloc((void**)&d_num_selected_out,sizeof(int));
    PPPEdge* d_compacted_edges = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_compacted_edges, ppp_edges.size() * sizeof(PPPEdge)));
    partition_saddle_candidate_edges(d_ppp_edges, d_compacted_edges, d_is_saddle_candidate, ppp_edges.size(),d_num_selected_out);
    int num_selected_out;
    cudaMemcpy(&num_selected_out, d_num_selected_out, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Found candidate saddle edge num : %d\n", num_selected_out);
    sort_saddle_candidate_edges(d_compacted_edges, d_ppp_vertices, num_selected_out);

    std::vector<PPPEdge> sorted_edges;
    sorted_edges.resize(num_selected_out);
    cudaMemcpy(sorted_edges.data(), d_compacted_edges, num_selected_out * sizeof(PPPEdge), cudaMemcpyDeviceToHost);

    return 0;
#pragma region Init_OpenGL_Context
        if (!glfwInit())
        {
            return -1;
        }
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        GLFWwindow* window = glfwCreateWindow(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT, "", nullptr, nullptr);

        if (!window) {
            glfwTerminate();
            return -1;
        }
        glfwMakeContextCurrent(window);

        if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
        {
            std::cout << "Failed to initialize GLAD" << std::endl;
            return -1;
        }

        glfwSetKeyCallback(window, keyCallback);
        glfwSetFramebufferSizeCallback(window, frameBufferResizeCallBack);
        glfwSetMouseButtonCallback(window, mouseButtonCallback);
        glfwSetCursorPosCallback(window, cursorPosCallback);
#pragma endregion 
    

#pragma region Generate_OpenGL_Resource
    std::vector<GLVertex> renderVertices;
    renderVertices.resize(vertices_count);

    float scaled_width = 10.0;
    float scaled_height = 10.0;

    float scaled_width_start = -(scaled_width / 2.0);
    float scaled_height_start = -(scaled_height / 2.0);

    float dw = scaled_width / width;
    float dh = scaled_height / height;

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            renderVertices[i * width + j].vx = scaled_width_start  + dw * j;
            renderVertices[i * width + j].vy = data[i * width + j] / 100.0;
            renderVertices[i * width + j].vz = scaled_height_start + dh * i;
        }
    }

    std::vector<unsigned int> renderIndices;
    renderIndices.reserve((height - 1) * (width - 1) * 6);
    for (int i = 0; i < vertices_count - width - 1; i++)
    {
        renderIndices.push_back(i);
        renderIndices.push_back(i + 1);
        renderIndices.push_back(i + 1 + width);
        renderIndices.push_back(i);
        renderIndices.push_back(i + 1 + width);
        renderIndices.push_back(i + width);
    }

    glGenVertexArrays(1, &terrainVAO);
    glGenBuffers(1, &terrainVBO);
    glGenBuffers(1, &terrainIBO);
    glBindVertexArray(terrainVAO);
    glBindBuffer(GL_ARRAY_BUFFER, terrainVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLVertex) * renderVertices.size(), renderVertices.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, terrainIBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * renderIndices.size(), renderIndices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(GLVertex), (void*)offsetof(GLVertex, vx));
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);

    terrainVertexShader = glCreateShader(GL_VERTEX_SHADER);
    terrainFragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    terrainShaderProgram = glCreateProgram();

    const char* vs = "#version 410 core\n"
    "layout(location = 0) in vec3 aPos;"
    "layout(location = 0) out vec3 wPos;"
    "uniform mat4 view;"
    "uniform mat4 projection;"
    "void main() {"
        "vec4 worldPosition = vec4(aPos, 1.0);"
        "gl_Position = projection * view * worldPosition;"
        "wPos = worldPosition.xyz;"
    "}";

    const char* fs = "#version 410 core\n"
        "layout(location = 0) in vec3 fragPos;"
        "out vec4 color;"
        "void main() {"
        " color = vec4(vec3(fragPos.y),1.0);"
        " if(fragPos.y > 0.99) { color = vec4(1.0,0.0,0.0,1.0);} ;"
        "}";

    glShaderSource(terrainVertexShader, 1, &vs, NULL);
    glShaderSource(terrainFragmentShader, 1, &fs, NULL);
    glCompileShader(terrainVertexShader);
    GLint success;
    glGetShaderiv(terrainVertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        GLchar infoLog[1024];
        glGetShaderInfoLog(terrainVertexShader, 1024, NULL, infoLog);
        std::cerr << "Vertex shader compilation error: " << infoLog << std::endl;
    }

    glCompileShader(terrainFragmentShader);
    glGetShaderiv(terrainFragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        GLchar infoLog[1024];
        glGetShaderInfoLog(terrainFragmentShader, 1024, NULL, infoLog);
        std::cerr << "Fragment shader compilation error: " << infoLog << std::endl;
    }
    glAttachShader(terrainShaderProgram, terrainVertexShader);
    glAttachShader(terrainShaderProgram, terrainFragmentShader);
    glLinkProgram(terrainShaderProgram);
    glGetProgramiv(terrainShaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        GLchar infoLog[1024];
        glGetProgramInfoLog(terrainShaderProgram, 1024, NULL, infoLog);
        std::cerr << "Shader program linking error: " << infoLog << std::endl;
    }
    std::cout << "Shaders successfully compiled .\n";
#pragma endregion

    glUseProgram(terrainShaderProgram);
    glBindVertexArray(terrainVAO);
    glBindBuffer(GL_ARRAY_BUFFER, terrainVBO);
    glPointSize(5.0);

    main_cam.setPos(getCamPosition(cam_distance, cam_phi, cam_theta));

    while (!glfwWindowShouldClose(window))
    {
        glClearColor(.0f, .0f, .0f, .0f);
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);

#pragma region Update_Render_Context
        updateMatrix4fv("view", main_cam.getViewMatrix());
        updateMatrix4fv("projection", main_cam.getProjectionMatrix());
#pragma endregion

        glDrawElements(GL_TRIANGLES, renderIndices.size(), GL_UNSIGNED_INT, 0);
        //Double buffering
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;

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