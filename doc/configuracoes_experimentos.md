# Configurações de Experimentos - Classificação de Lixo Reciclável

Este notebook contém diferentes configurações de parâmetros para facilitar a experimentação e comparação de resultados.

## Experimento Base (Atual)

```python
# Configuração atual do notebook principal
EXPERIMENT_NAME = "baseline_160x160_lr1e4"
IMG_SIZE = (160, 160)
BATCH_SIZE = 16
base_lr = 1e-4
USE_CLASS_WEIGHT = False
EPOCHS_CNN = 20
EPOCHS_TL_FREEZE = 8
EPOCHS_TL_FINETUNE = 10
```

## Experimento 1: Imagens Maiores + Class Weight

```python
EXPERIMENT_NAME = "baseline_224x224_lr5e-5_bs32"
IMG_SIZE = (224, 224)      # Imagens maiores para mais detalhes
BATCH_SIZE = 32            # Batch maior para melhor gradiente
base_lr = 5e-5             # LR menor para estabilidade
USE_CLASS_WEIGHT = True    # Balanceamento de classes
EPOCHS_CNN = 25
EPOCHS_TL_FREEZE = 10
EPOCHS_TL_FINETUNE = 15
```

## Experimento 2: Learning Rate Mais Alto

```python
EXPERIMENT_NAME = "exp2_lr2e4_classweight"
IMG_SIZE = (160, 160)
BATCH_SIZE = 16
base_lr = 2e-4             # LR mais alto para convergência rápida
USE_CLASS_WEIGHT = True    # Foco no balanceamento
EPOCHS_CNN = 30
EPOCHS_TL_FREEZE = 8
EPOCHS_TL_FINETUNE = 12
```

## Experimento 3: Batch Maior

```python
EXPERIMENT_NAME = "exp3_batch64"
IMG_SIZE = (160, 160)
BATCH_SIZE = 64            # Batch muito maior
base_lr = 2e-4             # LR ajustado para batch grande
USE_CLASS_WEIGHT = False
EPOCHS_CNN = 20
EPOCHS_TL_FREEZE = 8
EPOCHS_TL_FINETUNE = 10
```

## Experimento 4: Configuração Conservadora

```python
EXPERIMENT_NAME = "exp4_conservative"
IMG_SIZE = (128, 128)      # Imagens menores para treino rápido
BATCH_SIZE = 32
base_lr = 1e-4
USE_CLASS_WEIGHT = True
EPOCHS_CNN = 15            # Menos épocas
EPOCHS_TL_FREEZE = 6
EPOCHS_TL_FINETUNE = 8
```

## Experimento 5: Configuração Agressiva

```python
EXPERIMENT_NAME = "exp5_aggressive"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
base_lr = 5e-4             # LR bem alto
USE_CLASS_WEIGHT = True
EPOCHS_CNN = 40            # Mais épocas
EPOCHS_TL_FREEZE = 15
EPOCHS_TL_FINETUNE = 20
```

## Como Usar

1. Escolha uma das configurações acima
2. Copie os parâmetros para o notebook principal
3. Execute o treinamento
4. Compare os resultados na documentação

## Template para Novos Experimentos

```python
EXPERIMENT_NAME = "exp_custom_nome"
IMG_SIZE = (160, 160)      # (128,128), (160,160), (224,224)
BATCH_SIZE = 16            # 8, 16, 32, 64
base_lr = 1e-4             # 5e-5, 1e-4, 2e-4, 5e-4
USE_CLASS_WEIGHT = False   # True, False
EPOCHS_CNN = 20
EPOCHS_TL_FREEZE = 8
EPOCHS_TL_FINETUNE = 10

# Opcional: Modificações na augmentação
ROTATION_FACTOR = 0.03     # 0.01, 0.03, 0.05, 0.1
ZOOM_FACTOR = 0.05         # 0.02, 0.05, 0.1, 0.15
```

## Registro de Experimentos

| Experimento | Acurácia CNN | Acurácia TL | Tempo Treino | Observações |
|-------------|--------------|-------------|--------------|-------------|
| baseline_160x160_lr1e4 | 57.83% | 78.71% | ~X min | Baseline atual |
| baseline_224x224_lr5e-5_bs32 | - | - | - | A executar |
| exp2_lr2e4_classweight | - | - | - | A executar |
| exp3_batch64 | - | - | - | A executar |
| exp4_conservative | - | - | - | A executar |
| exp5_aggressive | - | - | - | A executar |

## Dicas de Experimentação

### Para Melhorar Acurácia:
- Aumentar IMG_SIZE
- Usar USE_CLASS_WEIGHT = True
- Mais épocas de fine-tuning
- LR menor e mais estável

### Para Treino Mais Rápido:
- Diminuir IMG_SIZE
- Aumentar BATCH_SIZE
- Menos épocas
- LR maior (mas cuidado com instabilidade)

### Para Melhor Generalização:
- USE_CLASS_WEIGHT = True
- Augmentação mais agressiva
- Dropout maior
- Early stopping mais restritivo

### Sinais de Overfitting:
- Gap grande entre train/val accuracy
- Val_loss aumentando enquanto train_loss diminui
- Performance no teste muito menor que validação

### Sinais de Underfitting:
- Train e val accuracy muito baixas
- Ambas as curvas ainda subindo no final
- Loss alto mesmo após muitas épocas