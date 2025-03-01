#####################################################################
# Getting onnx from pth model for hrnet requires a different setup  #
# python 3.6                                                        #
# torch 1.10.1                                                      #
# torchvision 0.11.2                                                #
#####################################################################

import numpy as np
from matplotlib import pyplot as plt
import sys
import torch
import torch.utils.data
import torchvision.transforms as transforms
# from config import cfg
import os
import os.path as osp
import urllib.request
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("Starting HRNet ONNX export script.")

lib_path = osp.join(os.getcwd(), 'HRNet-Human-Pose-Estimation/lib')
logger.debug(f"lib_path: {lib_path}")
sys.path.insert(0, lib_path)
logger.info(f"Added {lib_path} to sys.path.")


if not os.path.exists("model_binaries"):
    os.makedirs("model_binaries")
    logger.info("Created model_binaries directory.")
else:
    logger.info("model_binaries directory already exists.")

##Getting .pth file
OPTIMIZED_CHECKPOINT_URL = (
    # "https://github.com/quic/aimet-model-zoo/releases/download/hrnet-posenet/hrnet_posenet_FP32.pth"
    "https://github.com/quic/aimet-model-zoo/releases/download/hrnet-posenet/"
)

pth_file_path = "./model_binaries/hrnet_posenet_FP32.pth"

if not os.path.exists(pth_file_path):
    logger.info(f"Downloading .pth file from {OPTIMIZED_CHECKPOINT_URL} to {pth_file_path}")
    try:
        urllib.request.urlretrieve(
            f"{OPTIMIZED_CHECKPOINT_URL}/hrnet_posenet_FP32.pth",
            pth_file_path,
        )
        logger.info("Download complete.")
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        sys.exit(1)  # Exit if download fails
else:
    logger.info(".pth file already exists.")


input_shape = (1, 3, 256, 192)
logger.debug(f"Input shape: {input_shape}")
dummy_input = torch.randn(input_shape)
logger.info("Created dummy input.")

try:
    logger.info(f"Loading model from: {pth_file_path}")
    model = torch.load(pth_file_path)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {e}.  Check the file path and ensure it's a valid PyTorch model.")
    sys.exit(1)

logger.info("Moving model to CPU.")
model.to('cpu')
model.eval()  #  Important: Set the model to evaluation mode before exporting.
logger.info("Model moved to CPU and set to evaluation mode.")

onnx_model_name = "model_binaries/AIMET_HRNET_posnet.onnx"
logger.info(f"Output ONNX model name: {onnx_model_name}")

opset = 11
logger.debug(f"ONNX opset version: {opset}")

logger.info("Starting ONNX export...")
try:
    torch.onnx.export(
        model.cpu(),
        dummy_input,
        onnx_model_name,
        verbose=True,
        do_constant_folding=True,
        export_params=True,
        input_names=['input'],
        output_names=['output'],
        opset_version=opset
    )
    logger.info(f"ONNX export successful. Model saved to {onnx_model_name}")
except Exception as e:
    logger.error(f"Error during ONNX export: {e}")
    logger.error("Possible causes:")
    logger.error("- Incorrect PyTorch/Torchvision versions.  This script is designed for PyTorch 1.10.1 and Torchvision 0.11.2.")
    logger.error("- Issues with the model itself.  Ensure the loaded model is valid and can be evaluated.")
    logger.error("- Problems with dependencies or the HRNet-Human-Pose-Estimation library.")
    sys.exit(1)

logger.info("Script completed.")