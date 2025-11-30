# CNN Avançada vs MobileNetV2 - Experimento 3

## 🎯 Objetivo
Desenvolver uma arquitetura CNN customizada capaz de competir com Transfer Learning (MobileNetV2) na classificação de lixo reciclável.

## 🏗️ Arquitetura CNN Avançada

### Inovações Implementadas:
- **Residual Blocks:** Skip connections para treinar redes profundas sem degradação
- **Squeeze-and-Excitation:** Attention mechanism nos canais para focar em features importantes
- **Regularização Adaptativa:** Dropout crescente (0.1 → 0.4) por camada
- **Learning Rate Scheduling:** Decaimento exponencial para convergência otimizada
- **Classificador Robusto:** Múltiplas camadas densas com BatchNorm

### Estrutura da Rede:
```
Input (224x224x3)
    ↓
Initial Conv + Pool (7x7, stride=2)
    ↓
Residual Block 1 (64 filters, 2 layers)
    ↓
Residual Block 2 (128 filters, 2 layers, downsample)
    ↓
Residual Block 3 (256 filters, 2 layers, downsample)  
    ↓
Residual Block 4 (512 filters, 2 layers, downsample)
    ↓
Global SE Block (ratio=8)
    ↓
GlobalAveragePooling2D
    ↓
Dense Classifier (512→256→6)
    ↓
Output (6 classes)
```

## ⚔️ Competição

### Meta: Superar MobileNetV2 (75.10%)

| Modelo | Accuracy | Parâmetros | Estratégia |
|--------|----------|------------|------------|
| **MobileNetV2 TL** | 75.10% | ~2.3M | Transfer Learning |
| **CNN Avançada** | TBD | ~3.5M | From Scratch + Arquitetura Avançada |

## 📊 Hipóteses

### Quando CNN Avançada pode vencer:
✅ **Arquitetura otimizada** para classificação de materiais  
✅ **Attention mechanisms** focam em texturas importantes  
✅ **Residual connections** permitem rede mais profunda  
✅ **Regularização adaptativa** controla overfitting  

### Desafios esperados:
❌ **Dataset pequeno** (~2.5k imagens) favorece Transfer Learning  
❌ **Treinamento from scratch** requer mais épocas  
❌ **Risco de overfitting** sem features pré-treinadas  

## 🔬 Metodologia

### Configuração Experimental:
- **Dataset:** TrashNet (6 classes: cardboard, glass, metal, paper, plastic, trash)
- **Tamanho:** 224×224 pixels (mesmo do MobileNetV2)
- **Batch Size:** 32
- **Learning Rate:** 5e-5 (com scheduling)
- **Class Weights:** Habilitado (mesmo balanceamento)
- **Early Stopping:** Múltiplos critérios (loss + accuracy)

### Estratégias de Otimização:
1. **Callbacks Avançados:** ModelCheckpoint + EarlyStopping + ReduceLROnPlateau
2. **Regularização Multi-nível:** Dropout, Weight Decay, BatchNorm
3. **Monitoring:** Accuracy + Top-3 Accuracy para análise detalhada
4. **Comparison Framework:** Avaliação lado-a-lado com MobileNetV2

## 📁 Estrutura de Resultados

```
advanced_cnn_224x224_lr5e-5_bs32/
├── models/
│   └── advanced_cnn_best.keras          # Melhor modelo treinado
├── plots/
│   ├── advanced_cnn_training.png        # Curvas de treinamento
│   └── cm_advanced_cnn.png              # Matriz de confusão
├── history/
│   └── advanced_cnn_history.csv         # Histórico épocas
├── reports/
│   ├── advanced_cnn_report.csv          # Métricas por classe
│   └── experiment_comparison.json       # Comparação vs MobileNetV2
└── README.md                            # Esta documentação
```

## 🎯 Métricas de Avaliação

### Critérios de Sucesso:
- **Vitória:** CNN > 75.10% (supera MobileNetV2)
- **Empate:** CNN entre 73-75% (competitiva)
- **Derrota honrosa:** CNN > 65% (melhoria significativa vs baseline)
- **Falha:** CNN < 50% (problemas arquiteturais)

### Análises Incluídas:
- ✅ **Performance geral:** Accuracy no conjunto de teste
- ✅ **Performance por classe:** Precision, Recall, F1-Score
- ✅ **Análise visual:** Curvas de treinamento e matrizes de confusão
- ✅ **Comparação arquitetural:** Parâmetros, complexidade, eficiência
- ✅ **Tempo de treinamento:** Convergência vs MobileNetV2

## 🚀 Próximos Passos

### Se CNN Avançada VENCER:
1. **Documentar inovações** que levaram ao sucesso
2. **Testar em outros datasets** para validar generalização  
3. **Otimizar arquitetura** para deployment mobile
4. **Investigar ensemble** CNN + MobileNetV2

### Se MobileNetV2 VENCER:
1. **Analisar gaps** da CNN customizada
2. **Testar técnicas adicionais:** Data augmentation, Mixup, CutMix
3. **Investigar arquiteturas híbridas** 
4. **Validar hipótese** sobre tamanho de dataset

## 💡 Valor Científico

Este experimento contribui para o entendimento de:
- **Transfer Learning vs From Scratch** em domínios específicos
- **Eficácia de attention mechanisms** em classificação de materiais
- **Arquiteturas customizadas** vs modelos pré-treinados
- **Otimizações específicas** para datasets pequenos

**Status:** 🟡 Experimento em andamento  
**Última atualização:** 29/11/2025