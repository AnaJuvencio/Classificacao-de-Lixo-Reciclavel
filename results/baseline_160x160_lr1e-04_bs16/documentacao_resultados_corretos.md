# Documentação dos Resultados - Classificação de Lixo Reciclável

## Configuração do Experimento Atual

```python
# Configuração atual do notebook
IMG_SIZE = (160, 160)          # Tamanho das imagens
BATCH_SIZE = 16                # Tamanho do batch
base_lr = 1e-4                 # Taxa de aprendizado
USE_CLASS_WEIGHT = False       # Balanceamento desativado
EPOCHS_CNN = 20                # Épocas CNN Baseline
EPOCHS_TL_FREEZE = 8           # Épocas fase freeze
EPOCHS_TL_FINETUNE = 10        # Épocas fase fine-tuning
```

## Dataset

- **Dataset**: TrashNet (dataset-resized)
- **Classes**: 6 categorias
  - cardboard, glass, metal, paper, plastic, trash
- **Divisão**: 80% treino / 20% validação+teste
- **Augmentação**: RandomFlip, RandomRotation(0.03), RandomZoom(0.05)

## Resultados dos Modelos

### Comparação Geral

| Modelo | Acurácia no Teste | Performance |
|--------|------------------|-------------|
| **MobileNetV2 TL** | **78.71%** | 🏆 **Melhor modelo** |
| CNN Baseline | 57.83% | Baseline |
| **Diferença** | **+20.88 pontos percentuais** | **Melhoria significativa** |

### CNN Baseline - Resultados Detalhados

**Acurácia no Teste**: 57.83%  
**Loss no Teste**: 1.1819

**Performance por Classe**:

| Classe | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| cardboard | 61.36% | 79.41% | 69.23% | 34 |
| glass | 61.76% | 44.68% | 51.85% | 47 |
| metal | 75.76% | 59.52% | 66.67% | 42 |
| paper | 49.02% | 83.33% | 61.73% | 60 |
| plastic | 66.67% | 48.00% | 55.81% | 50 |
| **trash** | **0.00%** | **0.00%** | **0.00%** | **16** |

**Problemas Identificados**:
- ❌ **Falha total na classe "trash"** (0% de performance)
- ⚠️ Baixa recall para "glass" (44.68%) e "plastic" (48.00%)
- ⚠️ Baixa precision para "paper" (49.02%)

### MobileNetV2 Transfer Learning - Resultados Detalhados

**Acurácia no Teste**: 78.71%  
**Loss no Teste**: 0.5292

**Performance por Classe**:

| Classe | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| cardboard | 81.82% | 79.41% | 80.60% | 34 |
| glass | 78.72% | 72.55% | 75.51% | 51 |
| metal | 88.24% | 73.17% | 80.00% | 41 |
| paper | 83.87% | 85.25% | 84.55% | 61 |
| plastic | 70.59% | 80.00% | 75.00% | 45 |
| **trash** | **54.55%** | **70.59%** | **61.54%** | **17** |

**Principais Melhorias**:
- ✅ **Resolução do problema da classe "trash"** (0% → 61.54% F1)
- ✅ Performance consistente em todas as classes
- ✅ Melhoria geral em precision e recall

## Arquivos Gerados pelo Notebook

### Localização dos Resultados
**⚠️ IMPORTANTE**: Todos os arquivos são salvos no **diretório raiz** (onde está o notebook), **não** em subpastas organizadas.

### Modelos Salvos
```
cnn_baseline_best.keras          # Melhor modelo CNN Baseline
```

### Históricos de Treinamento
```
cnn_baseline_history.csv         # Histórico CNN Baseline
mobilenetv2_tl_freeze_history.csv    # Histórico fase freeze
mobilenetv2_tl_finetune_history.csv  # Histórico fase fine-tuning
```

### Gráficos Gerados
```
acc_cnn_baseline.png             # Gráfico accuracy CNN
loss_cnn_baseline.png            # Gráfico loss CNN
acc_mobilenetv2_tl_freeze.png    # Accuracy fase freeze
loss_mobilenetv2_tl_freeze.png   # Loss fase freeze
acc_mobilenetv2_tl_finetune.png  # Accuracy fase fine-tuning
loss_mobilenetv2_tl_finetune.png # Loss fase fine-tuning
```

### Matrizes de Confusão
```
cm_abs_cnn_baseline.png          # Matriz confusão absoluta CNN
cm_norm_cnn_baseline.png         # Matriz confusão normalizada CNN
cm_abs_mobilenetv2_tl.png        # Matriz confusão absoluta TL
cm_norm_mobilenetv2_tl.png       # Matriz confusão normalizada TL
```

### Relatórios CSV
```
class_report_cnn_baseline.csv    # Métricas por classe CNN
class_report_mobilenetv2_tl.csv  # Métricas por classe TL
models_comparison.csv            # Comparação final dos modelos
```

## Análise dos Resultados

### Principais Descobertas

1. **Transfer Learning é Superior**: MobileNetV2 supera CNN baseline em +20.88% de acurácia
2. **Problema do Desbalanceamento**: Classe "trash" tem apenas 16-17 exemplos vs 34-61 das outras
3. **CNN Baseline Falha**: Não consegue classificar "trash" (0% performance)
4. **Transfer Learning Resolve**: Consegue classificar todas as classes, incluindo "trash"

### Limitações Identificadas

1. **Desbalanceamento Severo**: `USE_CLASS_WEIGHT = False` prejudica classes minoritárias
2. **Dataset Pequeno**: Especialmente classe "trash" com poucos exemplos
3. **Configuração Conservadora**: LR baixo (1e-4) pode limitar aprendizado

## Recomendações para Próximos Experimentos

### 1. Balanceamento de Classes
```python
USE_CLASS_WEIGHT = True  # Ativar balanceamento
```

### 2. Imagens Maiores
```python
IMG_SIZE = (224, 224)    # Melhor para Transfer Learning
BATCH_SIZE = 32          # Ajustar para imagens maiores
```

### 3. Learning Rate
```python
base_lr = 5e-5          # Para imagens maiores (mais conservador)
# ou
base_lr = 2e-4          # Para convergência mais rápida
```

### 4. Mais Épocas
```python
EPOCHS_CNN = 25
EPOCHS_TL_FREEZE = 10
EPOCHS_TL_FINETUNE = 15
```

## Como Reproduzir

1. **Execute o notebook** `Projeto_Aprendizado_Profundo.ipynb`
2. **Verifique os arquivos** gerados no diretório raiz
3. **Analise os gráficos** de accuracy e loss
4. **Examine as matrizes** de confusão para entender os erros
5. **Compare os CSVs** para métricas detalhadas

## Próximos Passos

1. **Teste com `USE_CLASS_WEIGHT = True`** - Prioridade máxima
2. **Aumente o tamanho das imagens** para (224, 224)
3. **Experimente learning rates diferentes**
4. **Considere augmentação mais agressiva** para classe "trash"
5. **Organize os resultados** em subpastas por experimento

---

*Documentação atualizada: 29 de novembro de 2025*  
*Baseada na execução atual do notebook Projeto_Aprendizado_Profundo.ipynb*