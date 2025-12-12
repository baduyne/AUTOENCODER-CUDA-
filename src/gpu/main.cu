#include "gpu_autoencoder.h"
#include "utils.h"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)


void train_autoencoder(
    GPUAutoencoder& gpu_model,
    const std::vector<std::vector<float>>& train_images,
    const std::vector<std::vector<float>>& test_images,
    int batch_size,
    int epochs,
    float lr,
    int patience,
    int log_step_interval = 30)
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
            gpu_model.forward(h_input.data(), h_output.data(), batch_size);
            float loss = gpu_model.compute_loss(h_input.data(), batch_size);
            gpu_model.backward(h_input.data(), h_input.data(), batch_size);
            gpu_model.update_weights(lr);

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

            gpu_model.forward(h_input.data(), h_output.data(), batch_size);
            float loss = gpu_model.compute_loss(h_input.data(), batch_size);

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






// // Extract features from in-memory vectors (keeps old behavior)
// void extract_features_dataset(
//     GPUAutoencoder& gpu_model,
//     const std::vector<std::vector<float>>& train_images,
//     const std::vector<std::vector<float>>& test_images,
//     int batch_size,
//     std::vector<float>& train_features_out,
//     std::vector<float>& test_features_out)
// {
//     const int IMG_SIZE = IMG_C * IMG_H * IMG_W;
//     const int FEAT_SIZE = 128 * 8 * 8;
//     const size_t one_img_bytes = IMG_SIZE * sizeof(float);
//     const size_t one_feat_bytes = FEAT_SIZE * sizeof(float);

//     float* h_input = (float*)malloc(batch_size * one_img_bytes);
//     float* h_features = (float*)malloc(batch_size * one_feat_bytes);
//     if (!h_input || !h_features) { printf("Host malloc failed!\n"); return; }

//     printf("\n========== EXTRACT TRAIN FEATURES ==========" "\n");
//     train_features_out.resize(train_images.size() * FEAT_SIZE);
//     size_t idx = 0;
//     for (size_t i = 0; i + batch_size <= train_images.size(); i += batch_size) {
//         for (int b = 0; b < batch_size; ++b) memcpy(&h_input[b * IMG_SIZE], train_images[i + b].data(), one_img_bytes);
//         gpu_model.extract_features(h_input, h_features, batch_size);
//         for (int b = 0; b < batch_size; ++b) memcpy(&train_features_out[(idx + b) * FEAT_SIZE], &h_features[b * FEAT_SIZE], one_feat_bytes);
//         idx += batch_size; printf("Extracted %zu / %zu train images\r", idx, train_images.size());
//     }
//     printf("\nTrain feature extraction done.\n");

//     printf("\n========== EXTRACT TEST FEATURES ==========" "\n");
//     test_features_out.resize(test_images.size() * FEAT_SIZE);
//     idx = 0;
//     for (size_t i = 0; i + batch_size <= test_images.size(); i += batch_size) {
//         for (int b = 0; b < batch_size; ++b) memcpy(&h_input[b * IMG_SIZE], test_images[i + b].data(), one_img_bytes);
//         gpu_model.extract_features(h_input, h_features, batch_size);
//         for (int b = 0; b < batch_size; ++b) memcpy(&test_features_out[(idx + b) * FEAT_SIZE], &h_features[b * FEAT_SIZE], one_feat_bytes);
//         idx += batch_size; printf("Extracted %zu / %zu test images\r", idx, test_images.size());
//     }
//     printf("\nTest feature extraction done.\n");

//     free(h_input); free(h_features);
// }


// // Extract features from CIFAR bins in directory (streaming) and append into provided vectors
// bool extract_features_from_dir(GPUAutoencoder& gpu_model,
//                                const std::string& cifar_dir,
//                                const std::string& which, // "data_batch" or "test_batch"
//                                std::vector<float>& features_out,
//                                std::vector<int>& labels_out,
//                                int batch_size)
// {
//     namespace fs = std::filesystem;
//     if (!fs::exists(cifar_dir) || !fs::is_directory(cifar_dir)) { std::cerr << "Bad dir: " << cifar_dir << "\n"; return false; }

//     std::vector<std::string> files;
//     for (auto &ent : fs::directory_iterator(cifar_dir)) {
//         if (!ent.is_regular_file()) continue;
//         auto name = ent.path().filename().string();
//         if (which == "test_batch") {
//             if (name.find("test_batch") != std::string::npos) files.push_back(ent.path().string());
//         } else {
//             if (name.find("data_batch") != std::string::npos) files.push_back(ent.path().string());
//         }
//     }
//     std::sort(files.begin(), files.end());
//     if (files.empty()) { std::cerr << "No files for " << which << " in " << cifar_dir << "\n"; return false; }

//     const int FEAT_SIZE = 128 * 8 * 8;
//     const size_t IMG_ELEM = IMG_C * IMG_H * IMG_W;

//     std::vector<float> h_input(batch_size * IMG_ELEM);
//     std::vector<float> h_features(batch_size * FEAT_SIZE);

//     for (const auto &fpath : files) {
//         std::ifstream fin(fpath, std::ios::binary);
//         if (!fin.is_open()) { fprintf(stderr, "Can't open %s\n", fpath.c_str()); continue; }
//         std::vector<uint8_t> imgbuf(IMG_ELEM);
//         uint8_t lbl; size_t in_batch = 0;
//         while (true) {
//             fin.read((char*)&lbl, 1); if (!fin) break;
//             fin.read((char*)imgbuf.data(), IMG_ELEM); if (!fin) break;
//             for (size_t j = 0; j < IMG_ELEM; ++j) h_input[in_batch * IMG_ELEM + j] = static_cast<float>(imgbuf[j]) / 255.0f;
//             in_batch++;
//             if (in_batch == (size_t)batch_size) {
//                 gpu_model.extract_features(h_input.data(), h_features.data(), batch_size);
//                 for (int b = 0; b < batch_size; ++b) {
//                     for (int f = 0; f < FEAT_SIZE; ++f) features_out.push_back(h_features[b * FEAT_SIZE + f]);
//                     labels_out.push_back((int)lbl); // label from last read - acceptable for batching if consistent per sample
//                 }
//                 in_batch = 0;
//             }
//         }
//         fin.close();
//     }
//     return true;
// }

// // Write LIBSVM file
// bool save_libsvm_file(const std::string& filepath, const std::vector<float>& features, const std::vector<int>& labels, int feat_dim)
// {
//     if (labels.empty() || features.empty()) return false;
//     size_t n = labels.size(); if (features.size() != n * (size_t)feat_dim) return false;
//     std::ofstream f(filepath);
//     if (!f.is_open()) return false;
//     for (size_t i = 0; i < n; ++i) {
//         f << labels[i];
//         const float* row = &features[i * feat_dim];
//         for (int j = 0; j < feat_dim; ++j) {
//             float v = row[j];
//             if (v != 0.0f) f << " " << (j + 1) << ":" << std::fixed << std::setprecision(6) << v;
//         }
//         f << "\n";
//     }
//     return true;
// }

// void print_libsvm_commands(const std::string& train_file, const std::string& model_file, const std::string& test_file, const std::string& out_file) {
//     std::cout << "svm-train -s 0 -t 2 " << train_file << " " << model_file << "\n";
//     std::cout << "svm-predict " << test_file << " " << model_file << " " << out_file << "\n";
// }

// // int main() {
// //     int batch_size = 32;
// //     int epochs = 20;
// //     float lr = 0.001f;
// //     int patience = 2;

// //     GPUAutoencoder gpu_model;
// //     gpu_model.initialize();
// //     gpu_model.load_weights("./weights/model.bin");

// //     std::string cifar_dir = "../../data/cifar-10-batches-bin";

// //     // Train (streaming per-batch)
// //     train_autoencoder(gpu_model, cifar_dir, batch_size, epochs, lr, patience);

// //     // Optionally extract features and produce LIBSVM files (commented here; enable as needed)
// //     // std::vector<float> train_feats; std::vector<int> train_labels;
// //     // extract_features_from_dir(gpu_model, cifar_dir, "data_batch", train_feats, train_labels, batch_size);
// //     // save_libsvm_file("train.svm", train_feats, train_labels, 128*8*8);

// //     printf("Done.\n");
// //     return 0;
// // }

int main()
{

    int batch_size = 64;
    int epochs = 20;
    float lr = 0.001f;
    int patience = 2;

    GPUAutoencoder gpu_model;
    gpu_model.initialize();
    gpu_model.load_weights("./weights/model.bin");

    // load dataset 
    std::vector<std::vector<float>> train_images;
    std::vector<int> train_labels;
    std::vector<std::string> train_files = {
        "../../data/cifar-10-batches-bin/data_batch_1.bin",
        "../../data/cifar-10-batches-bin/data_batch_2.bin",
        "../../data/cifar-10-batches-bin/data_batch_3.bin",
        "../../data/cifar-10-batches-bin/data_batch_4.bin",
        "../../data/cifar-10-batches-bin/data_batch_5.bin"
    };
    if (!load_cifar10_dataset(train_files, train_images, train_labels, true)) 
    {  
       printf("Error loading training data\n");
       return 0;
    }
    else{
        printf("Loading training data: %zu images\n", train_labels.size());
    }

    // Example to load test data (10,000 images)
    std::vector<std::vector<float>> test_images;
    std::vector<int> test_labels;
    std::vector<std::string> test_files = {"../../data/cifar-10-batches-bin/test_batch.bin"};
    if (!load_cifar10_dataset(test_files, test_images, test_labels, false)) 
    {  
        printf("Error loading test data\n");
        return 0;
    }
    else{
        printf("Loading test data: %zu images\n", test_labels.size());
    }

    train_autoencoder(
    gpu_model,
    train_images,
    test_images,
    batch_size,
    epochs,
    lr,
    patience
    );

    // Save weights after finishing
    gpu_model.save_weights("./weights/model.bin");  
    // training 
}