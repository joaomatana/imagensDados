import cv2
import matplotlib.pyplot as plt

def show(image):
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    plt.show()

def show_contours(image, contours):
    image = image.copy()
    for contour in contours:
        cv2.drawContours(image, contour, -1, (0, 255, 0), 3)
    plt.figure()
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])

    plt.show()

# Leitura da imagem (alterar entre img2.jpg, img3.jpg e img4.jpg)
img: object = cv2.imread('imagens/img1.jpg')
show(img)

# Converte para escala de cinza
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
show(gray)

# Aplica a limiarização
ret, thresh = cv2.threshold(gray, 220, 1, cv2.THRESH_BINARY)
show(thresh)

# Encontra os contornos
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
show_contours(thresh, contours)

# Calcula e imprime a soma das áreas dos contornos
sum_contour = 0
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 100:
        sum_contour += 1

print("A soma das áreas das faces dos dados é:", sum_contour)

show_contours(thresh, contours)