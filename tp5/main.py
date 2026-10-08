"""
Trabalho Prático 5 - Visão Computacional / Aprendizado de Máquina
Depuração, Experimentos e Classificação Multiclasse com MLP from Scratch no Dataset Iris

Integrantes:
Gabriel Augusto Fabri Soltovski GRR20222546
Ricardo Quer GRR20224827

Repositório Git: https://github.com/GAFS-GAFS/Vis-oComputacional
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Configuração de estilo visual dos gráficos
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
os.makedirs('graficos', exist_ok=True)

# ---------------------------------------------------------------------------
# 1. IMPLEMENTAÇÃO DO MLP ORIGINAL (Com os comportamentos do Colab)
# ---------------------------------------------------------------------------
class OriginalMLP:
    """Implementação baseada no notebook do Colab para fins de comparação e depuração."""
    def __init__(self, input_dim=4, hidden_dim=5, output_dim=3, lr=0.005, epochs=1500, fix_predict=False):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lr = lr
        self.epochs = epochs
        self.fix_predict = fix_predict
        
        np.random.seed(42)
        # Inicialização com valores entre -1 e +1
        self.W_hidden = np.random.uniform(-1, 1, (input_dim, hidden_dim))
        self.W_output = np.random.uniform(-1, 1, (hidden_dim, output_dim))
        self.b_hidden = np.full(hidden_dim, -1.0)
        self.b_output = np.full(output_dim, -1.0)
        
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
        
    def sigmoid_deriv(self, x):
        return x * (1.0 - x)
        
    def fit(self, X, y):
        n = len(X)
        self.history_loss = []
        self.w_hidden_sample = []
        self.w_output_sample = []
        
        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            for idx, x_i in enumerate(X):
                # Forward
                net_h = np.dot(x_i, self.W_hidden) + self.b_hidden
                out_h = self.sigmoid(net_h)
                net_o = np.dot(out_h, self.W_output) + self.b_output
                out_o = self.sigmoid(net_o)
                
                # Target One-hot
                target = np.zeros(self.output_dim)
                target[int(y[idx])] = 1.0
                
                # Erro quadrático da amostra
                epoch_loss += np.sum((target - out_o) ** 2)
                
                # Backpropagation (do notebook original)
                error_o = target - out_o
                delta_o = -1.0 * error_o * self.sigmoid_deriv(out_o)
                
                # Atualização camada de saída (antes da oculta no código original)
                for i in range(self.hidden_dim):
                    for j in range(self.output_dim):
                        self.W_output[i, j] -= self.lr * (delta_o[j] * out_h[i])
                        if i == 0:
                            self.b_output[j] -= self.lr * delta_o[j]
                            
                # Gradiente camada oculta
                delta_h = np.dot(self.W_output, delta_o) * self.sigmoid_deriv(out_h)
                for i in range(self.input_dim):
                    for j in range(self.hidden_dim):
                        self.W_hidden[i, j] -= self.lr * (delta_h[j] * x_i[i])
                        if i == 0:
                            self.b_hidden[j] -= self.lr * delta_h[j]
                            
            mse = epoch_loss / n
            self.history_loss.append(mse)
            self.w_hidden_sample.append(self.W_hidden[0, 0])
            self.w_output_sample.append(self.W_output[0, 0])
            
        return self

    def predict(self, X):
        preds = []
        for x_i in X:
            if not self.fix_predict:
                # Bug do código original: cálculo estritamente linear sem ativação!
                f_h = np.dot(x_i, self.W_hidden) + self.b_hidden
                f_o = np.dot(f_h, self.W_output) + self.b_output
            else:
                # Correção: aplicação da sigmoide nas duas camadas
                f_h = self.sigmoid(np.dot(x_i, self.W_hidden) + self.b_hidden)
                f_o = self.sigmoid(np.dot(f_h, self.W_output) + self.b_output)
            preds.append(np.argmax(f_o))
        return np.array(preds)


# ---------------------------------------------------------------------------
# 2. IMPLEMENTAÇÃO DO MLP DEPURAÇÃO OTIMIZADA (Vetorizada, Robusta)
# ---------------------------------------------------------------------------
class OptimizedMLP:
    """Implementação corrigida, vetorizada e com monitoramento de treino e validação."""
    def __init__(self, input_dim=4, hidden_dim=8, output_dim=3, lr=0.08, epochs=600):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lr = lr
        self.epochs = epochs
        
        # Inicialização He/Xavier para evitar saturação
        np.random.seed(42)
        limit1 = np.sqrt(6.0 / (input_dim + hidden_dim))
        self.W1 = np.random.uniform(-limit1, limit1, (input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        
        limit2 = np.sqrt(6.0 / (hidden_dim + output_dim))
        self.W2 = np.random.uniform(-limit2, limit2, (hidden_dim, output_dim))
        self.b2 = np.zeros(output_dim)
        
    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -25, 25)))
        
    def sigmoid_deriv(self, a):
        return a * (1.0 - a)
        
    def one_hot(self, y):
        oh = np.zeros((len(y), self.output_dim))
        for i, val in enumerate(y):
            oh[i, int(val)] = 1.0
        return oh
        
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        y_train_oh = self.one_hot(y_train)
        if y_val is not None:
            y_val_oh = self.one_hot(y_val)
            
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.w1_traj = []
        self.w2_traj = []
        
        n_samples = len(X_train)
        
        for epoch in range(1, self.epochs + 1):
            # Gradiente Descendente Estocástico (amostra a amostra com pesos consistentes)
            indices = np.random.permutation(n_samples)
            for idx in indices:
                xi = X_train[idx:idx+1]      # (1, input_dim)
                yi = y_train_oh[idx:idx+1]   # (1, output_dim)
                
                # 1. Forward
                z1 = np.dot(xi, self.W1) + self.b1
                a1 = self.sigmoid(z1)
                z2 = np.dot(a1, self.W2) + self.b2
                a2 = self.sigmoid(z2)
                
                # 2. Backward (Cálculo dos deltas ANTES de atualizar qualquer peso)
                delta2 = (a2 - yi) * self.sigmoid_deriv(a2)
                delta1 = np.dot(delta2, self.W2.T) * self.sigmoid_deriv(a1)
                
                # 3. Atualização simultânea
                self.W2 -= self.lr * np.dot(a1.T, delta2)
                self.b2 -= self.lr * delta2[0]
                self.W1 -= self.lr * np.dot(xi.T, delta1)
                self.b1 -= self.lr * delta1[0]
                
            # Métricas da época completa no treino
            z1_all = np.dot(X_train, self.W1) + self.b1
            a1_all = self.sigmoid(z1_all)
            z2_all = np.dot(a1_all, self.W2) + self.b2
            a2_all = self.sigmoid(z2_all)
            train_mse = np.mean((a2_all - y_train_oh) ** 2)
            train_acc = accuracy_score(y_train, np.argmax(a2_all, axis=1))
            
            self.train_losses.append(train_mse)
            self.train_accs.append(train_acc)
            self.w1_traj.append(self.W1[0, 0])
            self.w2_traj.append(self.W2[0, 0])
            
            # Métricas de validação/teste para análise de overfitting
            if X_val is not None and y_val is not None:
                val_preds_prob = self.predict_proba(X_val)
                val_mse = np.mean((val_preds_prob - y_val_oh) ** 2)
                val_acc = accuracy_score(y_val, np.argmax(val_preds_prob, axis=1))
                self.val_losses.append(val_mse)
                self.val_accs.append(val_acc)
                
        return self

    def predict_proba(self, X):
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self.sigmoid(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        return self.sigmoid(z2)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


# ---------------------------------------------------------------------------
# 3. ROTINAS DE EXPERIMENTOS E GERAÇÃO DE GRÁFICOS
# ---------------------------------------------------------------------------
def executar_experimentos():
    print("=" * 75)
    print("TP5: EXPERIMENTOS E DEPURAÇÃO DO MLP FROM SCRATCH (DATASET IRIS)")
    print("=" * 75)
    
    # Carregamento do dataset Iris
    iris = load_iris()
    X = iris.data
    y = iris.target
    class_names = iris.target_names
    
    # Divisão 80% treino / 20% teste com estratificação
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Padronização (StandardScaler)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Total de amostras: {len(X)} | Treino: {len(X_train)} (80%) | Teste: {len(X_test)} (20%)")
    
    # -----------------------------------------------------------------------
    # EXPERIMENTO 1: Diagnóstico dos Bugs da Implementação Original
    # -----------------------------------------------------------------------
    print("\n[EXPERIMENTO 1] Comparativo de Depuração dos Modelos:")
    
    # 1.1 Original puro (dados não normalizados, predict linear sem ativação)
    mlp_orig = OriginalMLP(lr=0.005, epochs=1000, fix_predict=False)
    mlp_orig.fit(X_train, y_train)
    pred_orig = mlp_orig.predict(X_test)
    acc_orig = accuracy_score(y_test, pred_orig) * 100
    
    # 1.2 Original com correção da ativação no predict
    mlp_fix_pred = OriginalMLP(lr=0.005, epochs=1000, fix_predict=True)
    mlp_fix_pred.fit(X_train, y_train)
    pred_fix = mlp_fix_pred.predict(X_test)
    acc_fix = accuracy_score(y_test, pred_fix) * 100
    
    # 1.3 Modelo com Normalização + Correções (OptimizedMLP)
    mlp_opt = OptimizedMLP(hidden_dim=8, lr=0.08, epochs=600)
    mlp_opt.fit(X_train_scaled, y_train, X_test_scaled, y_test)
    pred_opt = mlp_opt.predict(X_test_scaled)
    acc_opt = accuracy_score(y_test, pred_opt) * 100
    
    print(f"  1. Código Original (com bug no predict e dados brutos):   {acc_orig:.2f}% de acurácia")
    print(f"  2. Código Corrigido no Predict (dados brutos):           {acc_fix:.2f}% de acurácia")
    print(f"  3. Modelo Otimizado (Normalizado + Backprop Corrigido):  {acc_opt:.2f}% de acurácia")

    # -----------------------------------------------------------------------
    # GRÁFICO 1: Curvas de Convergência do MSE (Original vs Otimizado)
    # -----------------------------------------------------------------------
    plt.figure(figsize=(9, 5))
    plt.plot(range(1, 1001), mlp_orig.history_loss[:1000], label='Original (Dados Brutos, lr=0.005)', color='#d62728', linestyle='--')
    plt.plot(range(1, 601), mlp_opt.train_losses, label='Otimizado (Normalizado, lr=0.08)', color='#1f77b4', linewidth=2)
    plt.title('Comparação de Convergência do Erro Quadrático Médio (MSE)', fontsize=12, fontweight='bold')
    plt.xlabel('Época', fontsize=11)
    plt.ylabel('Erro Quadrático Médio (MSE)', fontsize=11)
    plt.yscale('log')
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig('graficos/01_convergencia_mse.png', dpi=300)
    plt.close()
    print("[SALVO] graficos/01_convergencia_mse.png")

    # -----------------------------------------------------------------------
    # GRÁFICO 2: Análise de Overfitting (Perda Treino vs. Perda Teste)
    # -----------------------------------------------------------------------
    # Treinando por 1500 épocas para evidenciar comportamento de longo prazo
    mlp_long = OptimizedMLP(hidden_dim=8, lr=0.08, epochs=1200)
    mlp_long.fit(X_train_scaled, y_train, X_test_scaled, y_test)
    
    plt.figure(figsize=(9, 5))
    epochs_axis = range(1, 1201)
    plt.plot(epochs_axis, mlp_long.train_losses, label='Perda no Treino (Train Loss)', color='#2ca02c', linewidth=2)
    plt.plot(epochs_axis, mlp_long.val_losses, label='Perda no Teste (Test Loss)', color='#ff7f0e', linewidth=2, linestyle='--')
    
    # Encontrar melhor época de teste (ponto ideal para Early Stopping)
    best_epoch = np.argmin(mlp_long.val_losses) + 1
    best_val_loss = np.min(mlp_long.val_losses)
    plt.axvline(best_epoch, color='gray', linestyle=':', label=f'Ponto Ótimo Early Stopping (Época {best_epoch})')
    plt.scatter(best_epoch, best_val_loss, color='red', s=80, zorder=5)
    
    plt.title('Análise de Overfitting: Perda de Treinamento vs. Teste ao Longo das Épocas', fontsize=12, fontweight='bold')
    plt.xlabel('Número de Épocas', fontsize=11)
    plt.ylabel('MSE', fontsize=11)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig('graficos/02_overfitting_treino_teste.png', dpi=300)
    plt.close()
    print("[SALVO] graficos/02_overfitting_treino_teste.png")

    # -----------------------------------------------------------------------
    # GRÁFICO 3: Matriz de Confusão do Modelo Otimizado
    # -----------------------------------------------------------------------
    cm = confusion_matrix(y_test, pred_opt)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Matriz de Confusão no Teste (Acurácia: {acc_opt:.1f}%)', fontsize=12, fontweight='bold')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=25)
    plt.yticks(tick_marks, class_names)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=14, fontweight='bold')
                     
    plt.ylabel('Classe Verdadeira', fontsize=11)
    plt.xlabel('Classe Predita', fontsize=11)
    plt.tight_layout()
    plt.savefig('graficos/03_matriz_confusao_mlp.png', dpi=300)
    plt.close()
    print("[SALVO] graficos/03_matriz_confusao_mlp.png")

    # -----------------------------------------------------------------------
    # GRÁFICO 4: Evolução Temporal dos Pesos Sinápticos
    # -----------------------------------------------------------------------
    plt.figure(figsize=(9, 4.5))
    plt.plot(range(1, 601), mlp_opt.w1_traj, label='Peso W1[0,0] (Entrada 0 -> Oculta 0)', color='#9467bd', linewidth=2)
    plt.plot(range(1, 601), mlp_opt.w2_traj, label='Peso W2[0,0] (Oculta 0 -> Saída 0)', color='#8c564b', linewidth=2, linestyle='--')
    plt.title('Dinâmica de Ajuste dos Pesos Sinápticos ao Longo do Treinamento', fontsize=12, fontweight='bold')
    plt.xlabel('Época', fontsize=11)
    plt.ylabel('Valor do Peso', fontsize=11)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig('graficos/04_evolucao_pesos.png', dpi=300)
    plt.close()
    print("[SALVO] graficos/04_evolucao_pesos.png")

    # -----------------------------------------------------------------------
    # GRÁFICO 5: Fronteiras de Decisão no Espaço das Pétalas (2D)
    # -----------------------------------------------------------------------
    # Treinando um submodelo em 2D (comprimento e largura da pétala) para plotar fronteiras
    X_petal = X[:, [2, 3]]
    scaler_2d = StandardScaler()
    X_petal_scaled = scaler_2d.fit_transform(X_petal)
    X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X_petal_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    mlp_2d = OptimizedMLP(input_dim=2, hidden_dim=6, output_dim=3, lr=0.1, epochs=500)
    mlp_2d.fit(X_tr2, y_tr2)
    
    x_min, x_max = X_petal_scaled[:, 0].min() - 0.5, X_petal_scaled[:, 0].max() + 0.5
    y_min, y_max = X_petal_scaled[:, 1].min() - 0.5, X_petal_scaled[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = mlp_2d.predict(grid_points).reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.viridis)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for idx, cname in enumerate(class_names):
        plt.scatter(X_petal_scaled[y == idx, 0], X_petal_scaled[y == idx, 1],
                    label=cname, color=colors[idx], edgecolors='k', s=50)
    plt.title('Fronteiras de Decisão da Rede Neural MLP (Espaço das Pétalas)', fontsize=12, fontweight='bold')
    plt.xlabel('Comprimento da Pétala (padronizado)', fontsize=11)
    plt.ylabel('Largura da Pétala (padronizado)', fontsize=11)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig('graficos/05_fronteiras_decisao.png', dpi=300)
    plt.close()
    print("[SALVO] graficos/05_fronteiras_decisao.png")

    print("\n" + "=" * 75)
    print("MÉTRICAS FINAIS DETALHADAS (MODELO OTIMIZADO):")
    print("=" * 75)
    print(classification_report(y_test, pred_opt, target_names=class_names))

if __name__ == "__main__":
    executar_experimentos()
