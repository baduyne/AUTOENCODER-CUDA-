#include"gpu_autoencoder.h"
#include"utils.h"


// --- HÀM MAIN ---
int main() {

    int batch_size = 32;
    int epochs = 2;
    float lr = 0.001f;

    GPUAutoencoder gpu_model;
    gpu_model.initialize();
    
    gpu_model.load_weights("./weights/model.bin");
    // ================================
    // 2. Load CIFAR data
    // ================================
    std::vector<std::vector<float>> train_images;
    std::vector<int> train_labels;

    printf("Loading training data...\n");
    if (!load_cifar10_images("../../data/cifar-100-binary/cifar-100-binary/train.bin",
                              train_images,
                              train_labels,
                              1000))
    {
        printf("Load training failed!\n");
        return 1;
    }

    std::vector<std::vector<float>> test_images;
    std::vector<int> test_labels;

    printf("Loading test data...\n");
    if (!load_cifar10_images("../../data/cifar-100-binary/cifar-100-binary/test.bin",
                              test_images,
                              test_labels,
                              100))
    {
        printf("Load test failed!\n");
        return 1;
    }

    printf("Loaded %zu train images, %zu test images.\n",
           train_images.size(), test_images.size());

    // ================================
    // 3. Host batch memory
    // ================================
    size_t one_img = IMG_C * IMG_H * IMG_W;
    size_t input_sz = batch_size * one_img;

    float *h_input  = (float*)malloc(input_sz * sizeof(float));
    float *h_output = (float*)malloc(input_sz * sizeof(float));

    if (!h_input || !h_output) {
        printf("Host malloc failed!\n");
        return 1;
    }

    // ================================
    // 4. Train Loop + Early Stop
    // ================================
    int partition = 2;                   // patience = 2
    int partition_counter = 0;           // số lần eval không cải thiện
    float best_eval_loss = 1e9f;

    for (int e = 0; e < epochs; ++e)
    {
        printf("\n=== Epoch %d ===\n", e + 1);

        // ===================================================
        // TRAIN
        // ===================================================
        float train_loss = 0.0f;
        size_t train_batches = 0;

        for (size_t i = 0; i + batch_size <= train_images.size(); i += batch_size)
        {
            // Copy batch
            for (int b = 0; b < batch_size; b++) {
                memcpy(&h_input[b * one_img],
                       train_images[i + b].data(),
                       one_img * sizeof(float));
            }

            gpu_model.forward(h_input, h_output, batch_size);

            float loss = gpu_model.compute_loss(h_input, batch_size);

            gpu_model.backward(h_input, h_input, batch_size);
            gpu_model.update_weights(lr);

            train_loss += loss;
            train_batches++;
        }

        train_loss /= train_batches;
        printf("Train loss = %.6f\n", train_loss);


        // ===================================================
        // EVALUATION
        // ===================================================
        float eval_loss = 0.0f;
        size_t eval_batches = 0;

        for (size_t i = 0; i + batch_size <= test_images.size(); i += batch_size)
        {
            for (int b = 0; b < batch_size; b++) {
                memcpy(&h_input[b * one_img],
                       test_images[i + b].data(),
                       one_img * sizeof(float));
            }

            gpu_model.forward(h_input, h_output, batch_size);

            float loss = gpu_model.compute_loss(h_input, batch_size);

            eval_loss += loss;
            eval_batches++;
        }

        eval_loss /= eval_batches;
        printf("Eval loss = %.6f\n", eval_loss);


        // ===================================================
        // EARLY STOPPING (PATIENT PARTITION)
        // ===================================================
        if (eval_loss < best_eval_loss) {
            best_eval_loss = eval_loss;
            partition_counter = 0; 
        } else {
            partition_counter++;
            printf("Eval loss not improved → counter = %d\n", partition_counter);
        }

        if (partition_counter >= partition) {
            printf("Early stopping: eval loss did not improve for %d evaluations.\n",
                   partition);
            break;
        }
    }

    // ================================
    // Cleanup
    // ================================
    free(h_input);
    free(h_output);
    gpu_model.save_weights("./weights/model.bin");
    printf("\nTraining Finished.\n");
    return 0;
}
