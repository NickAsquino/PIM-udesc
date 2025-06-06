import pickle
import numpy as np
import os
import matplotlib.pyplot as plt

def volumeDFS(volume, valor_alvo):
    M, N, P = volume.shape
    rotulado = np.zeros_like(volume, dtype=np.int32)
    label = 1 
    
    
    #Lista de vizinhos 6-conectados em 3D:
    vizinhos = [
        ( 0,  0,  1),
        ( 0,  0, -1),
        ( 0,  1,  0),
        ( 0, -1,  0),
        ( 1,  0,  0),
        (-1,  0,  0)
    ]


    """
    #Lista de vizinhos 26-conectados em 3D:
    vizinhos = [
    (dz, dy, dx)
    for dz in [-1, 0, 1]
    for dy in [-1, 0, 1]
    for dx in [-1, 0, 1]
    if not (dz == dy == dx == 0)
    ]
    """

    def dfs(z, y, x):
        pilha = [(z, y, x)]
        while pilha:
            cz, cy, cx = pilha.pop()
            if (0 <= cz < M and 0 <= cy < N and 0 <= cx < P and
                volume[cz, cy, cx] == valor_alvo and
                rotulado[cz, cy, cx] == 0):
                rotulado[cz, cy, cx] = label
                for dz, dy, dx in vizinhos:
                    pilha.append((cz + dz, cy + dy, cx + dx))

    for z in range(M):
        for y in range(N):
            for x in range(P):
                if volume[z, y, x] == valor_alvo and rotulado[z, y, x] == 0:
                    dfs(z, y, x)
                    label += 1

    return rotulado, label - 1

def maiorAgrupamento(volume, valor_alvo):
    rotulado, total = volumeDFS(volume, valor_alvo)

    if total == 0:
        print(f"Nenhum agrupamento encontrado para o valor {valor_alvo}.")
        return np.zeros_like(volume)

    tamanhos = [(rotulado == l).sum() for l in range(1, total + 1)]

    maior_label = np.argmax(tamanhos) + 1  # +1 porque labels começam em 1
    maior_tamanho = tamanhos[maior_label - 1]
    print(f"Maior agrupamento de valor {valor_alvo} tem {maior_tamanho} voxels.")

    volume_maior = np.zeros_like(volume)
    volume_maior[rotulado == maior_label] = valor_alvo

    return volume_maior

with open('volume_TAC', 'rb') as f:
    volume = pickle.load(f)

def salvar(volume, pasta_destino, titulo=""):
    os.makedirs(pasta_destino, exist_ok=True)
    num_salvas = 0

    for i in range(volume.shape[0]):
        fatia = volume[i]
        if np.max(fatia) > 0:
            nome_arquivo = os.path.join(pasta_destino, f"{titulo}_slice_{i:03d}.png")
            plt.imsave(nome_arquivo, fatia, cmap='gray', vmin=0, vmax=255)
            num_salvas += 1

    print(f"{num_salvas} imagens salvas em '{pasta_destino}'.")

print(volume.shape)

unique_values, counts = np.unique(volume, return_counts=True)
for val, count in zip(unique_values, counts):
    print(f"Valor: {val}, quantidade: {count}")
print()
# proliferativas = volume == 255
# quiescentes   = volume == 200
# necróticas    = volume == 140

rot_255, tot_255 = volumeDFS(volume, 255)
tamanhos_255 = [(rot_255 == l).sum() for l in range(1, tot_255 + 1)]
np.savetxt("tamanhos_proliferativas.txt", tamanhos_255, fmt="%d")

rot_200, tot_200 = volumeDFS(volume, 200)
tamanhos_200 = [(rot_200 == l).sum() for l in range(1, tot_200 + 1)]
np.savetxt("tamanhos_quiescentes.txt", tamanhos_200, fmt="%d")

rot_140, tot_140 = volumeDFS(volume, 140)
tamanhos_140 = [(rot_140 == l).sum() for l in range(1, tot_140 + 1)]
np.savetxt("tamanhos_necroticas.txt", tamanhos_140, fmt="%d")

proliferativas_segmentadas = maiorAgrupamento(volume, 255)
quiescentes_segmentadas   = maiorAgrupamento(volume, 200)
necróticas_segmentadas   = maiorAgrupamento(volume, 140)
print()

salvar(volume, "slices_original", "original")
salvar(proliferativas_segmentadas, "slices_proliferativas", "prolif")
salvar(quiescentes_segmentadas, "slices_quiescentes", "quies")
salvar(necróticas_segmentadas, "slices_necroticas", "necro")
print()