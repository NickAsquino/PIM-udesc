import cv2
import numpy as np
import matplotlib.pyplot as plt

def ler_imagens():
    coin = cv2.imread('imagens/moedas.png', cv2.IMREAD_GRAYSCALE)
    chess = cv2.imread('imagens/chessboard_inv.png', cv2.IMREAD_GRAYSCALE)
    img = cv2.imread('imagens/img.jpg', cv2.IMREAD_GRAYSCALE)
    lua = cv2.imread('imagens/Lua1_gray.jpg', cv2.IMREAD_GRAYSCALE)

    return coin, chess, img, lua

def suavizar(imagens, nomes):
    imagensSuaves = [cv2.GaussianBlur(im, (3, 3), 0) for im in imagens]
    
    for imagem, nome in zip(imagensSuaves, nomes):
        plt.imshow(imagem, cmap='gray')
        plt.title(f'Imagem Suavizada - {nome}')
        plt.axis('off')
        plt.show()
    
    return imagensSuaves

def exibir_bordas_com_k(magnitude, direcao, nome, mascara):
    valores_K = [1.0, 1.2, 1.5]

    plt.figure(figsize=(12, 4))
    
    for i, K in enumerate(valores_K):
        bordas = supressao_n_maximos(magnitude, direcao, K)

        plt.subplot(1, len(valores_K), i + 1)
        plt.imshow(bordas, cmap='gray')
        plt.title(f'{mascara} - K={K}')
        plt.axis('off')

    plt.suptitle(f'Supressão de Não-Máximos - {nome} ({mascara})', fontsize=14)
    plt.tight_layout()
    plt.show()


def aplicarPrewitt(imagem):
    prewitt_x = np.array([[-1, 0, 1],
                          [-1, 0, 1],
                          [-1, 0, 1]])

    prewitt_y = np.array([[-1, -1, -1],
                          [ 0,  0,  0],
                          [ 1,  1,  1]])

    Gx = cv2.filter2D(imagem, -1, prewitt_x)
    Gy = cv2.filter2D(imagem, -1, prewitt_y)

    magnitude = np.sqrt(Gx**2 + Gy**2).astype(np.uint8)
    """
    plt.imshow(mag, cmap='gray')
    plt.title(f'Magnitude - Prewitt ({nome})')
    plt.axis('off')
    plt.show()
    """
    return Gx, Gy, magnitude

def calcularDirecao(Gx, Gy):
    direcao = np.arctan2(Gy, Gx + 1e-8)
    direcao = np.degrees(direcao)
    direcao[direcao < 0] += 180
    return direcao

def supressao_n_maximos(magnitude, direcao, K=1.0):
    altura, largura = magnitude.shape
    resultado = np.zeros((altura, largura), dtype=np.uint8)

    for i in range(1, altura - 1):
        for j in range(1, largura - 1):
            angulo = direcao[i, j]

            vizinho1 = 0
            vizinho2 = 0

            # Direção 0°
            if (0 <= angulo < 22.5) or (157.5 <= angulo <= 180):
                vizinho1 = magnitude[i, j - 1]
                vizinho2 = magnitude[i, j + 1]
            # Direção 45°
            elif (22.5 <= angulo < 67.5):
                vizinho1 = magnitude[i - 1, j + 1]
                vizinho2 = magnitude[i + 1, j - 1]
            # Direção 90°
            elif (67.5 <= angulo < 112.5):
                vizinho1 = magnitude[i - 1, j]
                vizinho2 = magnitude[i + 1, j]
            # Direção 135°
            elif (112.5 <= angulo < 157.5):
                vizinho1 = magnitude[i - 1, j - 1]
                vizinho2 = magnitude[i + 1, j + 1]

            if magnitude[i, j] >= K * vizinho1 and magnitude[i, j] >= K * vizinho2:
                resultado[i, j] = magnitude[i, j]

    return resultado

def processar_prewitt_completo(imagens, nomes):
    for imagem, nome in zip(imagens, nomes):
        Gx, Gy, magnitude = aplicarPrewitt(imagem)
        direcao = calcularDirecao(Gx, Gy)
        #bordas = supressao_n_maximos(magnitude, direcao)
        
        exibir_bordas_com_k(magnitude, direcao, nome, mascara='Prewitt')

        """plt.imshow(bordas, cmap='gray')
        plt.title(f'Bordas - {nome}')
        plt.axis('off')
        plt.show()
        """
    return Gx, Gy, magnitude

def aplicarScharr(imagem, nome='Imagem'):
    scharr_x = np.array([[-3, 0, 3],
                         [-10, 0, 10],
                         [-3, 0, 3]])

    scharr_y = np.array([[-3, -10, -3],
                         [ 0,   0,  0],
                         [ 3,  10,  3]])

    Gx = cv2.filter2D(imagem, -1, scharr_x)
    Gy = cv2.filter2D(imagem, -1, scharr_y)

    magnitude = np.sqrt(Gx**2 + Gy**2).astype(np.uint8)
    direcao = calcularDirecao(Gx, Gy)

    exibir_bordas_com_k(magnitude, direcao, nome, 'Scharr')

    """
    plt.imshow(magnitude, cmap='gray')
    plt.title(f'Magnitude - Scharr ({nome})')
    plt.axis('off')
    plt.show()
    """

    return Gx, Gy, magnitude

def comparar_prewitt_scharr(imagens, nomes):
    for imagem, nome in zip(imagens, nomes):
        PGx, PGy, mag_prewitt = aplicarPrewitt(imagem)
        SGx, SGy, mag_scharr = aplicarScharr(imagem, nome)

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(mag_prewitt, cmap='gray')
        plt.title(f'Prewitt - {nome}')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(mag_scharr, cmap='gray')
        plt.title(f'Scharr - {nome}')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

# Main
if __name__ == '__main__':
    coin, chess, img, lua = ler_imagens()
    imagens = [coin, chess, img, lua]
    nomes = ['Moedas', 'Chessboard', 'Img', 'Lua']

    imagens_suaves = suavizar(imagens, nomes)    

    processar_prewitt_completo(imagens_suaves, nomes)
    
    comparar_prewitt_scharr(imagens_suaves, nomes)