import cv2
from ultralytics import YOLO

# Carrega o modelo YOLO (ajuste o caminho pro seu modelo)
#model = YOLO("best.onnx", task='segment')  # ex: "best.pt"
model = YOLO("yolov8s-seg.onnx", task='segment')

# Carrega a imagem
img_path = "exemplo_corrosao.jpg"
img = cv2.imread(img_path)

# Faz a predição (modo segmentação)
results = model(img)

# Gera imagem com predições desenhadas
output_img = results[0].plot()

# Salva com o nome desejado
cv2.imwrite("segmentacao_onnx_com_yolo.jpg", output_img)
