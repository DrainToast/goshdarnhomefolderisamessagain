from llama_cpp import Llama
import os
import sys

def load_reranker_model(model_path):
    """Safely load the Qwen3 reranker model with validation"""

    # 1. First verify the file exists and is readable
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print(f"   Current directory: {os.getcwd()}")
        return None

    file_size_gb = os.path.getsize(model_path) / (1024**3)
    print(f"✅ File found: {model_path}")
    print(f"   Size: {file_size_gb:.2f} GB")

    # 2. Try to load with various compatibility settings
    load_params = {
        'model_path': model_path,
        'n_ctx': 2048,  # Start with smaller context for testing
        'verbose': True,
        'n_gpu_layers': 0  # Start with CPU only
    }

    # Common architectures for Qwen models - try each
    architectures_to_try = [
        {'model_type': None},  # Default, let library auto-detect
        {'model_type': 'qwen'},  # Try explicit Qwen type
        {'model_type': 'llama'},  # Some Qwen GGUF files use Llama architecture
    ]

    for arch in architectures_to_try:
        try:
            print(f"\n🔧 Attempting to load with params: {arch}")
            current_params = {**load_params, **arch}

            # Try with different n_gpu_layers
            for gpu_layers in [0, 1, 10, -1]:  # -1 means "all layers"
                try:
                    current_params['n_gpu_layers'] = gpu_layers
                    llm = Llama(**current_params)

                    # Test if model responds
                    test_output = llm("Hello", max_tokens=5, echo=False)
                    print(f"✅ Success! Loaded with {arch} and {gpu_layers} GPU layers")
                    print(f"   Test response: {test_output}")

                    return llm
                except Exception as e:
                    if gpu_layers != 0:
                        print(f"   Failed with {gpu_layers} GPU layers: {str(e)[:100]}...")
                        continue

        except Exception as e:
            print(f"❌ Failed with {arch}: {str(e)[:100]}...")
            continue

    print("\n🚨 All loading attempts failed!")
    return None

# Usage in your code
if __name__ == "__main__":
    # Update this path to your actual model location
    MODEL_PATH = "/home/aj/models/Qwen3-VL-Reranker-2B.i1-Q4_K_S.gguf"

    # Check file integrity first
    print("🔍 Checking model file...")
    with open(MODEL_PATH, 'rb') as f:
        header = f.read(8)
        print(f"   File header (hex): {header.hex()}")
        # GGUF files should start with specific magic bytes
        if header[4:8] == b'GGUF':
            print("   ✓ Valid GGUF file header detected")
        else:
            print("   ⚠️ File may not be a valid GGUF file")

    # Try to load
    llm = load_reranker_model(MODEL_PATH)

    if llm:
        print("\n🎉 Model loaded successfully!")
        # Now use it for reranking as shown in previous examples
    else:
        print("\n💡 Next steps:")
        print("1. Verify the file was downloaded completely")
        print("2. Try redownloading from a trusted source")
        print("3. Check file permissions: ls -la", MODEL_PATH)
