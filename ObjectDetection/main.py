import cv2
from matplotlib import pyplot as plt

# Abrindo a imagem
img = cv2.imread("imageOlhos.png")

# OpenCV abre a imagem como BRG
# mas como a queremos em RGB iremos
# também precisar de uma versão em grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Usando minSize para não confundir
# pontos muito pequenos com o objeto
# procurado na imagem
stop_data = cv2.CascadeClassifier('olhos_data.xml')

found = stop_data.detectMultiScale(img_gray,
                                   minSize=(20, 20))

# Não se faz nada se
# não houver nenhum objeto
amount_found = len(found)

if amount_found != 0:

    # Pode haver mais de um
    # sinal na imagem
    for (x, y, width, height) in found:
        # Desenhamos um retangulo verde ao redor de
        # todo o objeto reconhecido
        cv2.rectangle(img_rgb, (x, y),
                      (x + height, y + width),
                      (0, 255, 0), 5)

# Cria o ambiente da
# imagem e mostra
plt.subplot(1, 1, 1)
plt.imshow(img_rgb)
plt.show()