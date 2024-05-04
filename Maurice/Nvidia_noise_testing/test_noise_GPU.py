import cudaq

def run_simulation(target, noise_model=False, shots_count=1500):
    # Set the target for the simulator
    cudaq.set_target(target)

    # Define the noise model if any
    noise = cudaq.NoiseModel()
    bit_flip = cudaq.BitFlipChannel(.5)  # 50% probability of a bit flip
    noise.add_channel('x', [0], bit_flip)

    # Construct a quantum circuit
    kernel = cudaq.make_kernel()
    qubit = kernel.qalloc(1)

    # Measure the qubit in the zbasis
    kernel.x(qubit)
    kernel.mz(qubit)

    # Perform the simulation
    noise_model = noise if noise_model == True else None
    result = cudaq.sample(kernel, noise_model=noise_model, shots_count=shots_count)
    result.dump()
    return result

# Test scenario
def test_simulation():
    # Run with CPU target (noisy)
    print("Noisy simulation (CPU):")
    noisy_result_cpu = run_simulation('density-matrix-cpu', noise_model=True)

    # Run with NVIDIA target (noisy)
    print("Noisy simulation (NVIDIA):")
    noisy_result_gpu = run_simulation('nvidia', noise_model=True)

    # Run with CPU target (noiseless)
    print("Noiseless simulation (CPU):")
    noiseless_result_cpu = run_simulation('density-matrix-cpu', noise_model=False)


    print("\nComparing results:")
    print(f"CPU noisy: {noisy_result_cpu}")
    print(f"GPU noisy: {noisy_result_gpu}")
    print(f"CPU noiseless: {noiseless_result_cpu}")


if __name__ == "__main__":
    test_simulation()
