import tensorflow as tf
import time

# Check if GPU is available
gpus = tf.config.list_physical_devices('GPU')
if not gpus:
    print("GPU is not available. Please configure your GPU.")
else:
    print(f"Using GPU: {tf.test.gpu_device_name()}")

# Define matrix size (large size for heavy computation)
matrix_size = 5000  # Reduce size slightly, as complex numbers require more memory

# Generate large random complex matrices on the GPU
A_real = tf.random.uniform((matrix_size, matrix_size), dtype=tf.float32)
A_imag = tf.random.uniform((matrix_size, matrix_size), dtype=tf.float32)
B_real = tf.random.uniform((matrix_size, matrix_size), dtype=tf.float32)
B_imag = tf.random.uniform((matrix_size, matrix_size), dtype=tf.float32)

# Combine real and imaginary parts into complex matrices
A = tf.complex(A_real, A_imag)
B = tf.complex(B_real, B_imag)

# Perform complex matrix multiplication in a loop for heavy computation
num_iterations = 2500  # Adjust the number of iterations for longer computation
total_time = 0

print(f"Starting complex matrix multiplication for {num_iterations} iterations...")
for i in range(num_iterations):
    start_time = time.time()
    C = tf.matmul(A, B)  # Complex matrix multiplication
    end_time = time.time()

    iteration_time = end_time - start_time
    total_time += iteration_time
    print(f"Iteration {i + 1}: Time taken = {iteration_time:.2f} seconds")

# Display total and average time
print(f"\nComplex matrix multiplication completed for {num_iterations} iterations.")
print(f"Total time taken: {total_time:.2f} seconds")
print(f"Average time per iteration: {total_time / num_iterations:.2f} seconds")

