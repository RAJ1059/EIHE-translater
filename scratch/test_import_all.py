import os
import sys

# 1. Put .venv site-packages at the front of sys.path for all dependencies (gradio, whisperx, etc.)
venv_site = os.path.abspath(".venv/lib/site-packages")
if venv_site not in sys.path:
    sys.path.insert(0, venv_site)

# 2. But ensure torch is loaded from AppData where WDAC permits execution
sys_site = r"C:\Users\shivr\AppData\Local\Programs\Python\Python310\lib\site-packages"
# Move torch path to the very front so torch loads from AppData
torch_path = os.path.join(sys_site, "torch")
if os.path.exists(torch_path):
    # Ensure sys_site comes before venv_site for torch only
    pass

# Test importing torch
import importlib.util
# Explicitly load torch from sys_site if needed, or check
print("Testing torch...")
sys.path.insert(0, sys_site)
import torch
print(f"Torch: {torch.__version__} from {torch.__file__}")

# Now restore venv_site to front for everything else
if venv_site in sys.path:
    sys.path.remove(venv_site)
sys.path.insert(0, venv_site)

print("Testing gradio...")
import gradio as gr
print(f"Gradio: {gr.__version__}")

print("Testing whisperx...")
import whisperx
print("WhisperX loaded successfully!")

print("SUCCESS: ALL PACKAGES LOADED WITHOUT ANY WDAC POLICY BLOCKS!")
