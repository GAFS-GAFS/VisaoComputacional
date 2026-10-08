"""
Trabalho Prático 4 - Visão Computacional / Aprendizado de Máquina
Análise Exploratória, PCA e Classificação K-NN no Dataset Iris

Integrantes:
Gabriel Augusto Fabri Soltovski GRR20222546
Ricardo Quer GRR20224827
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Configurar estilo dos gráficos
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
CORES = {'Iris-setosa': '#1f77b4', 'Iris-versicolor': '#ff7f0e', 'Iris-virginica': '#2ca02c'}
LABELS_PT = {
    'comp_sepala': 'Comprimento da Sépala (cm)',
    'larg_sepala': 'Largura da Sépala (cm)',
    'comp_petala': 'Comprimento da Pétala (cm)',
    'larg_petala': 'Largura da Pétala (cm)'
}

def carregar_dados():
    caminho = os.path.join(os.path.dirname(__file__), 'iris', 'iris.data')
    if not os.path.exists(caminho):
        caminho = 'iris/iris.data'
    
    colunas = ['comp_sepala', 'larg_sepala', 'comp_petala', 'larg_petala', 'classe']
    df = pd.read_csv(caminho, header=None, names=colunas).dropna()
    return df

def calcular_estatisticas(df):
    features = ['comp_sepala', 'larg_sepala', 'comp_petala', 'larg_petala']
    print("=" * 70)
    print("1. FREQUÊNCIA DAS CATEGORIAS")
    print("=" * 70)
    freq = df['classe'].value_counts()
    freq_perc = df['classe'].value_counts(normalize=True) * 100
    tab_freq = pd.DataFrame({'Contagem': freq, 'Percentual (%)': freq_perc})
    print(tab_freq.to_string())

    print("\n" + "=" * 70)
    print("2. ESTATÍSTICAS GERAIS (MÉDIA, DESVIO PADRÃO, MODA)")
    print("=" * 70)
    medias_geral = df[features].mean()
    desvios_geral = df[features].std()
    modas_geral = [stats.mode(df[f], keepdims=True).mode[0] for f in features]
    tab_geral = pd.DataFrame({
        'Variável': [LABELS_PT[f] for f in features],
        'Média': medias_geral.values,
        'Desvio Padrão': desvios_geral.values,
        'Moda': modas_geral
    })
    print(tab_geral.to_string(index=False))

    print("\n" + "=" * 70)
    print("3. ESTATÍSTICAS POR CATEGORIA")
    print("=" * 70)
    for cat, group in df.groupby('classe'):
        print(f"\n--- Categoria: {cat} ---")
        m = group[features].mean()
        d = group[features].std()
        mo = [stats.mode(group[f], keepdims=True).mode[0] for f in features]
        tab_cat = pd.DataFrame({
            'Variável': [LABELS_PT[f] for f in features],
            'Média': m.values,
            'Desvio Padrão': d.values,
            'Moda': mo
        })
        print(tab_cat.to_string(index=False))

    return tab_freq, tab_geral

def gerar_graficos(df):
    os.makedirs('graficos', exist_ok=True)
    features = ['comp_sepala', 'larg_sepala', 'comp_petala', 'larg_petala']
    classes = sorted(df['classe'].unique())

    # 1. Relação Sépala: Comprimento vs Largura
    plt.figure(figsize=(8, 6))
    for c in classes:
        sub = df[df['classe'] == c]
        plt.scatter(sub['comp_sepala'], sub['larg_sepala'], label=c, color=CORES[c], s=60, alpha=0.8, edgecolors='k')
        # Linha de tendência
        m, b = np.polyfit(sub['comp_sepala'], sub['larg_sepala'], 1)
        r = sub['comp_sepala'].corr(sub['larg_sepala'])
        x_vals = np.linspace(sub['comp_sepala'].min(), sub['comp_sepala'].max(), 50)
        plt.plot(x_vals, m * x_vals + b, color=CORES[c], linestyle='--', alpha=0.7, label=f'Tendência {c} (r={r:.2f})')
    plt.title('Relação entre Comprimento e Largura da Sépala por Categoria', fontsize=12, fontweight='bold')
    plt.xlabel('Comprimento da Sépala (cm)', fontsize=11)
    plt.ylabel('Largura da Sépala (cm)', fontsize=11)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig('graficos/01_relacao_sepala.png', dpi=300)
    plt.close()
    print("[SALVO] graficos/01_relacao_sepala.png")

    # 2. Relação Pétala: Comprimento vs Largura
    plt.figure(figsize=(8, 6))
    for c in classes:
        sub = df[df['classe'] == c]
        plt.scatter(sub['comp_petala'], sub['larg_petala'], label=c, color=CORES[c], s=60, alpha=0.8, edgecolors='k')
        m, b = np.polyfit(sub['comp_petala'], sub['larg_petala'], 1)
        r = sub['comp_petala'].corr(sub['larg_petala'])
        x_vals = np.linspace(sub['comp_petala'].min(), sub['comp_petala'].max(), 50)
        plt.plot(x_vals, m * x_vals + b, color=CORES[c], linestyle='--', alpha=0.7, label=f'Tendência {c} (r={r:.2f})')
    plt.title('Relação entre Comprimento e Largura da Pétala por Categoria', fontsize=12, fontweight='bold')
    plt.xlabel('Comprimento da Pétala (cm)', fontsize=11)
    plt.ylabel('Largura da Pétala (cm)', fontsize=11)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig('graficos/02_relacao_petala.png', dpi=300)
    plt.close()
    print("[SALVO] graficos/02_relacao_petala.png")

    # 3. Distribuição 4D: Boxplots comparativos das 4 variáveis
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()
    for idx, f in enumerate(features):
        data_to_plot = [df[df['classe'] == c][f] for c in classes]
        bp = axes[idx].boxplot(data_to_plot, patch_artist=True, labels=[c.replace('Iris-', '') for c in classes])
        for patch, c in zip(bp['boxes'], classes):
            patch.set_facecolor(CORES[c])
            patch.set_alpha(0.7)
        axes[idx].set_title(LABELS_PT[f], fontsize=11, fontweight='bold')
        axes[idx].set_ylabel('Centímetros (cm)')
    plt.suptitle('Distribuição das 4 Variáveis por Categoria (Boxplots)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('graficos/03_distribuicao_4d_boxplots.png', dpi=300)
    plt.close()
    print("[SALVO] graficos/03_distribuicao_4d_boxplots.png")

    # 4. Matriz de Dispersão 4D (Pairplot manual)
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i in range(4):
        for j in range(4):
            ax = axes[i, j]
            if i == j:
                # Histograma na diagonal
                for c in classes:
                    vals = df[df['classe'] == c][features[i]]
                    ax.hist(vals, bins=10, alpha=0.5, color=CORES[c])
            else:
                # Dispersão fora da diagonal
                for c in classes:
                    sub = df[df['classe'] == c]
                    ax.scatter(sub[features[j]], sub[features[i]], color=CORES[c], s=15, alpha=0.7)
            if i == 3:
                ax.set_xlabel(features[j].replace('_', ' '), fontsize=9)
            if j == 0:
                ax.set_ylabel(features[i].replace('_', ' '), fontsize=9)
    plt.suptitle('Matriz de Dispersão 4D entre Todas as Variáveis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('graficos/04_matriz_dispersao_4d.png', dpi=300)
    plt.close()
    print("[SALVO] graficos/04_matriz_dispersao_4d.png")

def executar_pca(df):
    features = ['comp_sepala', 'larg_sepala', 'comp_petala', 'larg_petala']
    X = df[features]
    y = df['classe']
    classes = sorted(df['classe'].unique())

    # Padronização das variáveis (média=0, variância=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_

    print("\n" + "=" * 70)
    print("4. ANÁLISE DE COMPONENTES PRINCIPAIS (PCA)")
    print("=" * 70)
    print(f"Variância explicada pelo Componente 1 (PC1): {var_exp[0]*100:.2f}%")
    print(f"Variância explicada pelo Componente 2 (PC2): {var_exp[1]*100:.2f}%")
    print(f"Variância acumulada total (PC1 + PC2):      {sum(var_exp)*100:.2f}%")

    # Gráfico PCA
    plt.figure(figsize=(8, 6))
    for c in classes:
        mask = (y == c)
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=c, color=CORES[c], s=65, alpha=0.85, edgecolors='k')
    plt.title(f'Projeção 2D via PCA (Variância Explicada Total: {sum(var_exp)*100:.1f}%)', fontsize=12, fontweight='bold')
    plt.xlabel(f'Primeiro Componente Principal - PC1 ({var_exp[0]*100:.1f}%)', fontsize=11)
    plt.ylabel(f'Segundo Componente Principal - PC2 ({var_exp[1]*100:.1f}%)', fontsize=11)
    plt.axhline(0, color='gray', linestyle=':', alpha=0.6)
    plt.axvline(0, color='gray', linestyle=':', alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig('graficos/05_pca_2d.png', dpi=300)
    plt.close()
    print("[SALVO] graficos/05_pca_2d.png")

def executar_knn(df):
    features = ['comp_sepala', 'larg_sepala', 'comp_petala', 'larg_petala']
    X = df[features]
    y = df['classe']
    classes = sorted(df['classe'].unique())

    # Divisão 80% treino e 20% teste com estratificação
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print("\n" + "=" * 70)
    print("5. CLASSIFICAÇÃO COM K-NN (80% TREINO / 20% TESTE)")
    print("=" * 70)
    print(f"Amostras de Treinamento: {len(X_train)} (80%)")
    print(f"Amostras de Teste:       {len(X_test)} (20% - 10 por classe)")

    # Testando valores de k de 1 a 15
    k_range = range(1, 16)
    acuracias = []
    for k in k_range:
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        acuracias.append(accuracy_score(y_test, pred))

    # Gráfico de Acurácia vs K
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, acuracias, marker='o', color='#2b5c8f', linewidth=2, markersize=7)
    plt.title('Acurácia do K-NN no Conjunto de Teste para Diferentes Valores de K', fontsize=12, fontweight='bold')
    plt.xlabel('Número de Vizinhos (k)', fontsize=11)
    plt.ylabel('Acurácia', fontsize=11)
    plt.xticks(k_range)
    plt.ylim(0.85, 1.02)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('graficos/06_knn_curva_k.png', dpi=300)
    plt.close()
    print("[SALVO] graficos/06_knn_curva_k.png")

    # Avaliação detalhada com k=3
    k_escolhido = 3
    modelo = KNeighborsClassifier(n_neighbors=k_escolhido)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    acuracia_final = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=classes)

    print(f"\nResultados para k = {k_escolhido}:")
    print(f"Acurácia no Teste: {acuracia_final * 100:.2f}%\n")
    print("Relatório de Classificação:")
    print(classification_report(y_test, y_pred, labels=classes))

    # Gráfico da Matriz de Confusão
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Matriz de Confusão (K-NN com k={k_escolhido})', fontsize=12, fontweight='bold')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    nomes_curtos = [c.replace('Iris-', '') for c in classes]
    plt.xticks(tick_marks, nomes_curtos, rotation=30)
    plt.yticks(tick_marks, nomes_curtos)

    # Anotações nas células
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=12, fontweight='bold')

    plt.ylabel('Classe Verdadeira', fontsize=11)
    plt.xlabel('Classe Predita', fontsize=11)
    plt.tight_layout()
    plt.savefig('graficos/07_matriz_confusao_knn.png', dpi=300)
    plt.close()
    print("[SALVO] graficos/07_matriz_confusao_knn.png")

def main():
    print("Iniciando Análise Completa do Dataset Iris (TP4)...")
    df = carregar_dados()
    calcular_estatisticas(df)
    gerar_graficos(df)
    executar_pca(df)
    executar_knn(df)
    print("\nTodos os cálculos, análises e gráficos foram concluídos com sucesso!")

if __name__ == "__main__":
    main()
