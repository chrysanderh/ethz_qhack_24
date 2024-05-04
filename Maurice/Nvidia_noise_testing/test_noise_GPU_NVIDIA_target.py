import cudaq

def run_simulation(target, noise_model=None, shots_count=100):
    # Set the target for the simulator
    cudaq.set_target(target)
    print(f"Running simulation on: {cudaq.get_target()}")

    # Define the noise model if any
    noise = cudaq.NoiseModel()
    if noise_model:
        bit_flip = cudaq.BitFlipChannel(.5)  # 50% probability of a bit flip
        noise.add_channel('x', [0], bit_flip)

    # Construct a quantum circuit
    kernel = cudaq.make_kernel()
    qubit = kernel.qalloc()

    # Measure the qubit in the zbasis
    kernel.mz(qubit)

    # Perform the simulation
    result = cudaq.sample(kernel, noise_model=noise if noise_model else None, shots_count=shots_count)
    result.dump()
    print(result)
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


    print("Comparing results:")
    print(f"CPU noisy: {noisy_result_cpu}")
    print(f"GPU noisy: {noisy_result_gpu}")
    print(f"CPU noiseless: {noiseless_result_cpu}")

    # Check if the results are the same
    print(f"noisy_result_cpu == noisy_result_gpu: {noisy_result_cpu == noisy_result_gpu}")
    print(f"noisy_result_cpu == noiseless_result_cpu: {noisy_result_cpu == noiseless_result_cpu}")

test_simulation()
