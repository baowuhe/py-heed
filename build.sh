#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "Building py-heed..."
uv build

mkdir -p bin

# Clean bin directory
rm -rf bin/*

# Extract wheel to bin
wheel_file=$(ls dist/*.whl 2>/dev/null | head -1)
if [ -n "$wheel_file" ]; then
    echo "Extracting wheel to bin/..."
    cd bin
    unzip -o "../$wheel_file"
    cd ..
fi

# Create standalone py-heed wrapper script
cat > bin/py-heed << 'WRAPPER'
#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Add CUDA 13 libraries to LD_LIBRARY_PATH if available
if [ -d "/usr/local/cuda/lib64" ]; then
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
fi

# Use uv run to handle dependencies
exec uv run --directory "$PROJECT_DIR" python -m heed "$@"
WRAPPER

chmod +x bin/py-heed

echo "Done! Output in bin/"
echo "Run: ./bin/py-heed --help"
