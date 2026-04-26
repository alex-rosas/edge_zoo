import subprocess
import sys
import os

def run(cmd):
    subprocess.run(cmd, shell=True, check=True)

# Install dependencies
run("pip install -q onnx onnxruntime pyyaml pandas")

# Add project root to path
sys.path.insert(0, '/content/edgezoo')

print("✓ Dependencies installed")
print("✓ Python path configured")
print("✓ Ready to run EdgeZoo")