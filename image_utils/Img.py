import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv

ROJO = (0, 0, 255)
VERDE = (0, 255, 0)
AZUL = (255, 0, 0)
TRANSPARENTE = (0, 0, 0, 0)


def cargar_imagen(route: str = 'img/kong.jpg', window_name: str = 'prueba1', mostrar=False) -> np.ndarray:
    src = cv.imread(route)
    if mostrar:
        cv.namedWindow(window_name, cv.WINDOW_AUTOSIZE)
        cv.imshow(window_name, src)
        cv.waitKey(0)
    return src


def print_image_info(image: np.ndarray):
    print(image.shape)
    print(image.size)
    print(image.dtype)
    print(image.ndim)


def show_3_gray(one, two, three):
    figura, (eje1, eje2, eje3) = plt.subplots(1, 3)
    eje1.imshow(one, cmap="gray", vmin=0, vmax=255)
    eje2.imshow(two, cmap="gray")
    eje3.imshow(three, cmap="gray")
    plt.show()


def separar_rgb(image, mostrar_var=False):
    red, green, blue = cv.split(image)
    # separar colores
    if mostrar_var:
        show_3_gray(red, green, blue)
    return red, green, blue


def separar_hsv(image, mostrar_var=False):
    src = cv.cvtColor(image, cv.COLOR_RGB2HSV)
    h = src[:, :, 0]
    s = src[:, :, 1]
    v = src[:, :, 2]
    if mostrar_var:
        show_3_gray(h, s, v)
    return h, s, v


def separar_luv(image, mostrar_var=False):
    src = cv.cvtColor(image, cv.COLOR_BGR2Luv)
    h = src[:, :, 0]
    s = src[:, :, 1]
    v = src[:, :, 2]
    if mostrar_var:
        show_3_gray(h, s, v)
    return h, s, v
    src


def unir_colores(red, green, blue):
    return cv.merge([blue, green, red])


def escala_grises(img, mostrar_var=False):
    grises = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    if mostrar_var:
        mostrar(grises, gris=True)
    return grises


def ver_todos_los_grises(img):
    r, g, b = separar_rgb(img, mostrar_var=True)
    h, s, v = separar_hsv(img, mostrar_var=True)
    l, u, v = separar_luv(img, mostrar_var=True)


def restar_imagenes(img, img2, mostrar=False):
    """
    Resta con saturación
    :param img:
    :param img2:
    :return:
    """
    resta = cv.subtract(img, img2)
    if mostrar:
        mostrar(resta, gris=True)
    return resta


def sumar_imagenes(img, img2, mostrar=False):
    suma = cv.add(img, img2)
    if mostrar:
        mostrar(suma, gris=True)
    return suma


def multiplicar_imagenes(img, img2):
    multiplicacion = cv.multiply(img, img2)
    mostrar(multiplicacion, gris=True)
    return multiplicacion


def dividir_imanenes(img, img2):
    division = cv.divide(img, img2)
    mostrar(division, gris=True)
    return division


def ver_cambios_entre(img1, img2):
    caja1 = escala_grises(
        img1
    )
    caja2 = escala_grises(
        img2
    )
    cambios = cv.bitwise_xor(caja1, caja2)
    show_3_gray(caja1, caja2, cambios)


def mostrar(img, gris=False):
    if gris:
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(img)
    plt.show()


def mostrar_rgb(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()


def binarizar(img, media):
    img[img < media] = 0
    img[img >= media] = 255
    return img


def binarizar_inv(img, media):
    img[img < media] = 255
    img[img >= media] = 0
    return img


def normalizar_entre_01(img):
    img_array = np.float32(img)
    dst = np.zeros(img_array.shape, dtype=np.float32)
    normalizado = cv.normalize(img_array, dst=dst, alpha=1, beta=0, norm_type=cv.NORM_INF)
    # si se quisiera mostrar multiplicar por 255 para que se vea bien
    # plt.imshow(nrm*255, cmap="gray", vmin = 0, vmax=255)
    return normalizado


def mostar_histograma(image: np.ndarray):
    figura, ejes = plt.subplots(1, 3, figsize=(10, 4))
    figura.suptitle("Histogramas RBG")

    colores = ["red", "green", "blue"]
    for i, color in enumerate(colores):
        histograma = cv.calcHist([image], [i], None, [256], [0, 256])
        ejes[i].set_title(f"Histograma de {color}")
        ejes[i].plot(histograma, color=color)
    plt.show()


def histograma_equalized(image: np.ndarray):
    figura, ejes = plt.subplots(1, 3, figsize=(10, 4))
    figura.suptitle("Histogramas RBG ecualizados")

    equalizadas = []
    colores = ["red", "green", "blue"]
    for i, color in enumerate(colores):
        equalization = cv.equalizeHist(image[:, :, i])
        equalizadas.append(equalization)
        histograma = cv.calcHist([equalization], [0], None, [256], [0, 256])
        ejes[i].set_title(f"Histograma de {color}")
        ejes[i].plot(histograma, color=color)
    plt.show()
    return equalizadas[0], equalizadas[1], equalizadas[2]


def suavizar(image):
    kernel = (1 / 4) * np.array([[0, 1, 0],
                                 [1, 0, 1],
                                 [0, 1, 0]])
    todos_los_canales = -1
    return cv.filter2D(src=image, ddepth=todos_los_canales, kernel=kernel)


def realzar(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    todos_los_canales = -1
    return cv.filter2D(src=image, ddepth=todos_los_canales, kernel=kernel)


def mostrar_varios_rgb(*imagenes: np.ndarray):
    largo = len(imagenes)
    alto = 1
    """if largo > 3:
        alto = largo % 3
        largo = 3
"""
    figura, ejes = plt.subplots(alto, largo, figsize=(10, 4))
    figura.suptitle("Imagenes")
    if ejes.ndim == 1:
        for i, imagen in enumerate(imagenes):
            ejes[i].imshow(cv.cvtColor(imagen, cv.COLOR_BGR2RGB))
    else:
        for i, imagen in enumerate(imagenes):
            ejes[i // largo, i % largo].imshow(cv.cvtColor(imagen, cv.COLOR_BGR2RGB))
    plt.show()


def escribir(image: np.ndarray, pos: tuple, size: int, text: str):
    cv.putText(
        image,
        text,
        pos,
        cv.FONT_HERSHEY_SIMPLEX,
        0.15 * size,
        (255, 255, 255),
        2,
        cv.LINE_AA
    )


def guardar_imagen(image, name):
    cv.imwrite(name, image)
