#  Copyright (C) 2021 Texas Instruments Incorporated - http://www.ti.com/
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions
#  are met:
#
#    Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
#    Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the
#    distribution.
#
#    Neither the name of Texas Instruments Incorporated nor the names of
#    its contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import cv2
from time import time

from edgeai_dl_inferer import onnxrt

# GStreamer pipeline for camera capture
capture_pipeline = "gst-launch-1.0 v4l2src device=/dev/video4 ! image/jpeg,width=1920,height=1080,framerate=30/1 ! jpegdec ! videoconvert n-threads=4 ! video/x-raw,format=BGR ! appsink"
cap = cv2.VideoCapture(capture_pipeline, cv2.CAP_GSTREAMER)

# GStreamer pipeline for video output
output_pipeline = "appsrc ! videoconvert n-threads=4 ! waylandsink sync=false"
out = cv2.VideoWriter(output_pipeline, cv2.CAP_GSTREAMER, 0, 30, (1920, 1080), True)

if not cap.isOpened() or not out.isOpened():
    print("Failed to open capture or writer")
    exit(1)

def preprocessImage(image, batchSize=1, floatInput=False):
    batch = []

    for _ in range(batchSize):
        img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if(floatInput):
            img = img.astype(np.float32) / 255.0  # Scale to [0,1]

        img = np.transpose(img, (2, 0, 1))

        batch.append(img)
    

    return np.array(batch)

def postprocessClass(result, n=5):
    result = result[0][0]
    topIndices = np.argsort(result)[-n:]
    topValues = []
    for idx in topIndices:
        topValues.append((idx, result[idx]))
    return topValues

# modelPath = "/opt/custom/models/cl-0000_tflitert_imagenet1k_mlperf_mobilenet_v1_1.0_224_tflite/model/mobilenet_v1_1.0_224.tflite"
# artifactsPath = "/opt/custom/models/cl-0000_tflitert_imagenet1k_mlperf_mobilenet_v1_1.0_224_tflite/artifacts"

# modelPath = "/opt/custom/models/cl-0000_tflitert_imagenet1k_mlperf_mobilenet_v1_1.0_224_low_latency_4_tflite/model/mobilenet_v1_1.0_224.tflite"
# artifactsPath = "/opt/custom/models/cl-0000_tflitert_imagenet1k_mlperf_mobilenet_v1_1.0_224_low_latency_4_tflite/artifacts"

conf_batch_size = 2
core_count = 4

modelPath = f"/opt/custom/models/mobilenetv2/opset/mobilenetv2_224x224_{conf_batch_size}_optimized.onnx"
artifactsPath = f"/opt/custom/models/mobilenetv2/opset/ll/mobilenetv2ll{core_count}_224x224_{conf_batch_size}"

t0 = time()
interpreter = onnxrt(artifactsPath, modelPath, enable_tidl=True, core_number = core_count, inference_mode = 2)
t1 = time()

tdiff = 1000.0*(t1-t0)

print(f"Interpreter Initialization time: {tdiff:.8} ms")

batchSize = interpreter.batch_size
runs = int(256/batchSize)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    preprocessed = preprocessImage(frame, batchSize)
    result = interpreter(preprocessed)
    t0 = time()
    for _ in range(0, runs):
        result = interpreter(preprocessed)
    t1 = time()
    
    tdiff = (1000.0*(t1-t0))/runs
    fps = (batchSize*1000.0)/tdiff
    print(f"Inference time: {tdiff:.8} ms ({fps:.8} fps) - Batch Size: {batchSize}")
    
    # print(postprocessClass(result))

    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

exit(0)