import torch

def check_cuda():
    if torch.cuda.is_available():
        print("CUDA is available!")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

        # Perform a simple computation on the CUDA device
        device = torch.device("cuda")
        print("Performing a simple matrix multiplication on the GPU...")

        # Create two random matrices
        matrix_a = torch.randn(1000, 1000, device=device)
        matrix_b = torch.randn(1000, 1000, device=device)

        # Perform matrix multiplication
        result = torch.matmul(matrix_a, matrix_b)

        print("Computation completed.")
        print("Result (first element):", result[0, 0].item())
    else:
        print("CUDA is not available.")

if __name__ == "__main__":
    print("Hellor from Cirrus")
    check_cuda()