from app_dl_inferer_test import main as infer_main
import sys

def run_inference_py(model_dir, image_path, alpha=0.4):
    sys.argv = [
        "app_dl_inferer_test.py",
        "-d", model_dir,
        "-i", image_path,
        "-m", "TIDL",
        "-a", str(alpha)
    ]
    infer_main()

# 👇 Aqui é onde você chama a função com seus paths
run_inference_py(
    model_dir="/home/usuario/modelos/segmentador",
    image_path="/home/usuario/imagens/teste.jpg",
    alpha=0.4
)

#no terminal: python3 run_seg.py

#tudo pelo terminal:
# python3 app_dl_inferer_test.py \
#   -d /home/usuario/meu_modelo \ --> é a pasta toda, descomprimida
#   -i /home/usuario/imagem.jpg \ -->imagem
#   -m TIDL \ --> mode
#   -a 0.4
