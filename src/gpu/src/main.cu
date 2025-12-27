#include "gpu_autoencoder.h"
#include "gpu_autoencoder_loop_opt.h"
#include "utils.h"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <string>
#include <vector>
#include <cstdint>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)


void train_autoencoder(
    GPUAutoencoder* gpu_model,
    const std::vector<std::vector<float>>& train_images,
    const std::vector<std::vector<float>>& test_images,
    int batch_size,
    int epochs,
    float lr,
    int patience,
    int log_step_interval = 50)
{
    const size_t IMG_ELEM = IMG_C * IMG_H * IMG_W;

    std::vector<float> h_input(batch_size * IMG_ELEM);
    std::vector<float> h_output(batch_size * IMG_ELEM);

    // =============================
    // OPEN TWO LOG FILES
    // =============================
    std::ofstream train_log("train_log.csv", std::ios::out);
    std::ofstream eval_log("eval_log.csv", std::ios::out);

    
    if (!train_log.is_open()) std::cerr << "Failed to open train_log.csv\n";
    if (!eval_log.is_open()) std::cerr << "Failed to open eval_log.csv\n";

    train_log << "epoch,step,train_loss,step_time_ms\n";
    eval_log  << "epoch,eval_loss,epoch_time_ms\n";
    train_log.flush();
    eval_log.flush();
    // Early stopping
    int partition_counter = 0;
    float best_eval_loss = 1e9f;
    

    // CUDA event (reuse for epoch)
    cudaEvent_t epoch_start_ev, epoch_end_ev;
    CUDA_CHECK(cudaEventCreate(&epoch_start_ev));
    CUDA_CHECK(cudaEventCreate(&epoch_end_ev));

    for (int e = 0; e < epochs; ++e)
    {
        int train_step = 0;
        printf("\n=== Epoch %d ===\n", e + 1);

        // start epoch timer
        CUDA_CHECK(cudaEventRecord(epoch_start_ev, 0));

        // TRAIN LOOP
        float train_loss = 0.0f;
        size_t train_batches = 0;

        for (size_t i = 0; i + batch_size <= train_images.size(); i += batch_size)
        {
            // load batch
            for (int b = 0; b < batch_size; ++b) {
                memcpy(&h_input[b * IMG_ELEM],
                       train_images[i + b].data(),
                       IMG_ELEM * sizeof(float));
            }

            // Step cuda events
            cudaEvent_t step_start_ev, step_end_ev;
            CUDA_CHECK(cudaEventCreate(&step_start_ev));
            CUDA_CHECK(cudaEventCreate(&step_end_ev));

            CUDA_CHECK(cudaEventRecord(step_start_ev, 0));

            // forward + backward
            gpu_model->forward(h_input.data(), h_output.data(), batch_size);
            float loss = gpu_model->compute_loss(h_input.data(), batch_size);
            gpu_model->backward(h_input.data(), h_input.data(), batch_size);
            gpu_model->update_weights(lr);

            CUDA_CHECK(cudaEventRecord(step_end_ev, 0));
            CUDA_CHECK(cudaEventSynchronize(step_end_ev));

            float step_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&step_ms, step_start_ev, step_end_ev));

            // write train log
            if (train_log.is_open() && train_step % log_step_interval == 0) {
                train_log << (e + 1) << ","
                          << train_batches << ","
                          << std::fixed << std::setprecision(6) << loss << ","
                          << std::fixed << std::setprecision(3) << step_ms
                          << "\n";
                train_log.flush();
            }
            train_step++;

            CUDA_CHECK(cudaEventDestroy(step_start_ev));
            CUDA_CHECK(cudaEventDestroy(step_end_ev));

            train_loss += loss;
            train_batches++;
        }

        train_loss /= train_batches;
        printf("Train loss = %.6f\n", train_loss);

        // EVAL LOOP
        float eval_loss = 0.0f;
        size_t eval_batches = 0;

        for (size_t i = 0; i + batch_size <= test_images.size(); i += batch_size)
        {
            for (int b = 0; b < batch_size; ++b) {
                memcpy(&h_input[b * IMG_ELEM],
                       test_images[i + b].data(),
                       IMG_ELEM * sizeof(float));
            }

            gpu_model->forward(h_input.data(), h_output.data(), batch_size);
            float loss = gpu_model->compute_loss(h_input.data(), batch_size);

            eval_loss += loss;
            eval_batches++;
        }

        eval_loss /= eval_batches;
        printf("Eval loss = %.6f\n", eval_loss);

        // finish epoch timing
        CUDA_CHECK(cudaEventRecord(epoch_end_ev, 0));
        CUDA_CHECK(cudaEventSynchronize(epoch_end_ev));

        float epoch_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&epoch_ms, epoch_start_ev, epoch_end_ev));

        // WRITE EVAL LOG ONLY HERE
        if (eval_log.is_open()) {
            eval_log << (e + 1) << ","
                     << std::fixed << std::setprecision(6) << eval_loss << ","
                     << std::fixed << std::setprecision(3) << epoch_ms
                     << "\n";
            eval_log.flush();
        }


        // EARLY STOPPING
        if (eval_loss < best_eval_loss) {
            best_eval_loss = eval_loss;
            partition_counter = 0;
        } else {
            partition_counter++;
            printf("Eval not improved → counter = %d\n", partition_counter);
        }

        if (partition_counter >= patience) {
            printf("Early stopping triggered after %d eval stagnations.\n", patience);
            break;
        }
    }

    cudaEventDestroy(epoch_start_ev);
    cudaEventDestroy(epoch_end_ev);

    if (train_log.is_open()) train_log.close();
    if (eval_log.is_open()) eval_log.close();
}



// Extract features from in-memory vectors (keeps old behavior)
void extract_features_dataset(
    GPUAutoencoder* gpu_model,
    const std::vector<std::vector<float>>& train_images,
    const std::vector<std::vector<float>>& test_images,
    int batch_size,
    std::vector<float>& train_features_out,
    std::vector<float>& test_features_out)
{
    const int IMG_SIZE = IMG_C * IMG_H * IMG_W;
    const int FEAT_SIZE = 128 * 8 * 8;
    const size_t one_img_bytes = IMG_SIZE * sizeof(float);
    const size_t one_feat_bytes = FEAT_SIZE * sizeof(float);

    float* h_input = (float*)malloc(batch_size * one_img_bytes);
    float* h_features = (float*)malloc(batch_size * one_feat_bytes);
    if (!h_input || !h_features) { printf("Host malloc failed!\n"); return; }

    printf("\n========== EXTRACT TRAIN FEATURES ==========" "\n");
    train_features_out.resize(train_images.size() * FEAT_SIZE);
    size_t idx = 0;
    for (size_t i = 0; i + batch_size <= train_images.size(); i += batch_size)
    {
        for (int b = 0; b < batch_size; ++b) memcpy(&h_input[b * IMG_SIZE], train_images[i + b].data(), one_img_bytes);
        gpu_model->extract_features(h_input, h_features, batch_size);


        for (int b = 0; b < batch_size; ++b) memcpy(&train_features_out[(idx + b) * FEAT_SIZE], &h_features[b * FEAT_SIZE], one_feat_bytes);
        idx += batch_size; printf("Extracted %zu / %zu train images\r", idx, train_images.size());
    }
    printf("\nTrain feature extraction done.\n");

    printf("\n========== EXTRACT TEST FEATURES ==========" "\n");
    test_features_out.resize(test_images.size() * FEAT_SIZE);
    idx = 0;
    for (size_t i = 0; i + batch_size <= test_images.size(); i += batch_size)
    {
        for (int b = 0; b < batch_size; ++b) memcpy(&h_input[b * IMG_SIZE], test_images[i + b].data(), one_img_bytes);
        gpu_model->extract_features(h_input, h_features, batch_size);

        for (int b = 0; b < batch_size; ++b) memcpy(&test_features_out[(idx + b) * FEAT_SIZE], &h_features[b * FEAT_SIZE], one_feat_bytes);
        idx += batch_size; printf("Extracted %zu / %zu test images\r", idx, test_images.size());
    }
    printf("\nTest feature extraction done.\n");

    free(h_input); free(h_features);
}



int gpu_phase_main(int argc, char** argv)
{

    Config cfg;
    parse_args(argc, argv, cfg);
    int batch_size = cfg.batch_size;
    int epochs     = cfg.epochs;
    float lr       = cfg.lr;
    int patience   = cfg.patience;

    std::string weight_path = cfg.weight_path;
    std::string train_folder = cfg.train_folder;
    std::string test_folder  = cfg.test_folder;
    std::string out_folder   = cfg.output_folder;

    printf("batch_size: %d, epochs: %d\n",batch_size, epochs);

    // Create model based on optimization type
    GPUAutoencoder* gpu_model = nullptr;
    if (cfg.optimization_type == "loop-unroll") {
        printf("[INFO] Using Loop-Unrolled Optimized GPU Autoencoder\n");
        gpu_model = new GPUAutoencoderLoopOpt();
    } else {
        printf("[INFO] Using Baseline GPU Autoencoder\n");
        gpu_model = new GPUAutoencoder();
    }
    gpu_model->initialize();

    // Try to load pre-trained weights
    std::ifstream weight_check(weight_path);
    bool weights_loaded = false;

    if (weight_check.good()) {
        weight_check.close();
        gpu_model->load_weights(weight_path);
        weights_loaded = true;
        printf("[INFO] Loaded pre-trained weights from: %s\n", weight_path.c_str());
    } else {
        printf("[WARNING] No pre-trained weights found at: %s\n", weight_path.c_str());
        printf("[INFO] Using Kaiming-initialized weights (model will need training)\n");
    }

    // load dataset 
    std::vector<std::vector<float>> train_images;
    std::vector<int> train_labels;
    std::vector<std::string> train_files = load_bin_files_from_folder(train_folder);

    // Loại bỏ test_batch.bin ra khỏi train nếu bạn muốn tách riêng
    train_files.erase(
        std::remove_if(train_files.begin(), train_files.end(),
                    [](const std::string& s) { 
                        return s.find("test_batch") != std::string::npos; 
                    }),
        train_files.end()
    );

    if (!load_cifar10_dataset(train_files, train_images, train_labels, true)) {
        printf("Error loading training data\n");
        return 0;
    } else {
        printf("Loading training data: %zu images\n", train_labels.size());
    }


    // Example to load test data (10,000 images)
    std::vector<std::vector<float>> test_images;
    std::vector<int> test_labels;
    std::vector<std::string> test_files = load_bin_files_from_folder(test_folder);

    // chỉ lấy file test_batch.bin
    test_files.erase(
        std::remove_if(test_files.begin(), test_files.end(),
                    [](const std::string& s) {
                        return s.find("test_batch.bin") == std::string::npos;
                    }),
        test_files.end()
    );

    if (!load_cifar10_dataset(test_files, test_images, test_labels, false)) {
        printf("Error loading test data\n");
        return 0;
    } else {
        printf("Loading test data: %zu images\n", test_labels.size());
    }


    bool should_train = !weights_loaded;  // Train if weights weren't loaded
    if (should_train) {
        printf("\n[INFO] Starting training phase...\n");
        train_autoencoder(
            gpu_model,
            train_images,
            test_images,
            batch_size,
            epochs,
            lr,
            patience
        );

        // Create weight directory if it doesn't exist
        std::filesystem::path weight_file_path(weight_path);
        std::filesystem::create_directories(weight_file_path.parent_path());

        // Save trained weights
        gpu_model->save_weights(weight_path);
        printf("[INFO] Trained weights saved to: %s\n", weight_path.c_str());
    } else {
        printf("\n[INFO] Skipping training (using pre-trained weights)\n");
    }

    gpu_model->save_weights(weight_path);
    // After training, extract features from the datasets (preserve order)
    std::vector<float> train_feats;
    std::vector<float> test_feats;
    extract_features_dataset(gpu_model, train_images, test_images, batch_size, train_feats, test_feats);

    // Create output directory if it doesn't exist
    std::filesystem::create_directories(out_folder);

    // Save features and labels to binary files (row-major float32, labels int32)
    // Train features
    {
        std::string train_features = out_folder + "/train_features.bin";
        std::ofstream fout(train_features, std::ios::binary);
        fout.write(reinterpret_cast<const char*>(train_feats.data()), train_feats.size() * sizeof(float));
        fout.close();
    }
    // Train labels
    {
        std::string train_labels = out_folder + "/train_labels.bin";
        std::ofstream fout(train_labels, std::ios::binary);
        // write as int32
        for (int v : train_labels) {
            int32_t x = static_cast<int32_t>(v);
            fout.write(reinterpret_cast<const char*>(&x), sizeof(int32_t));
        }
        fout.close();
    }
    // Test features
    {
        std::string test_features = out_folder + "/test_features.bin";
        std::ofstream fout(test_features, std::ios::binary);
        fout.write(reinterpret_cast<const char*>(test_feats.data()), test_feats.size() * sizeof(float));
        fout.close();
    }
    // Test labels
    {
        std::string test_labels = out_folder + "/test_labels.bin";
        std::ofstream fout(test_labels, std::ios::binary);
        for (int v : test_labels) {
            int32_t x = static_cast<int32_t>(v);
            fout.write(reinterpret_cast<const char*>(&x), sizeof(int32_t));
        }
        fout.close();
    }
    printf("\n[SUCCESS] GPU phase completed successfully!\n");
    printf("[INFO] Features saved:\n");
    printf("  - train_features.bin (%zu samples)\n", train_images.size());
    printf("  - test_features.bin (%zu samples)\n", test_images.size());
    printf("  - train_labels.bin\n");
    printf("  - test_labels.bin\n");

    // Cleanup
    delete gpu_model;

    return 0;
}