# Análise do Experimento 8

## Comparação: Experimento 7 → Experimento 8

### 📊 Fitness
- **Experimento 7**: 1853.35
- **Experimento 8**: 2006.54 ✅ **(+8.3% de melhoria)**
- **Melhor resultado até agora!** 🏆

---

## ⚙️ Mudanças nos Hiperparâmetros

### Elitismo e Seleção
- **elitism_count**: 3 → **10** (preserva mais elite, menos perda de boas soluções)
- **tournament_size**: 3 → **5** (pressão seletiva moderada)

### Mutação
- **Todos os parâmetros de mutação mantidos iguais**:
  - mutation_rate: 0.05
  - min/max_mutation_rate: 0.05/0.12
  - enable_adaptive_strength: true (0.25→0.05)

---

## 🔍 Análise dos Parâmetros Evoluídos

### Frequência
**Exp 7**: 10.82 Hz
**Exp 8**: 11.39 Hz (+5.2%)
- Movimento ligeiramente mais rápido
- Ainda dentro da faixa ótima (~10-12 Hz)

### Amplitudes
**Exp 7**: Range 0.000 a 0.222 (média ~0.14)
- Junta 6 zerada
- Juntas 2, 4, 7, 8 dominantes

**Exp 8**: Range 0.030 a 0.239 (média ~0.15)
- **Todas as juntas ativas** (nenhuma zerada!)
- Junta 6 agora contribui (0.099)
- Juntas 2, 4, 8 ainda dominantes (0.22-0.24)
- Distribuição mais equilibrada

**Comparação por junta**:
| Junta | Exp 7 | Exp 8 | Mudança |
|-------|-------|-------|----------|
| 1 | 0.066 | 0.030 | -55% |
| 2 | 0.186 | 0.239 | +29% |
| 3 | 0.114 | 0.052 | -55% |
| 4 | 0.195 | 0.217 | +11% |
| 5 | 0.173 | 0.120 | -30% |
| 6 | **0.000** | **0.099** | ∞ (reativada!) |
| 7 | 0.189 | 0.193 | +2% |
| 8 | 0.222 | 0.239 | +8% |

### Fases
**Exp 7**: Fases agrupadas
- Grupo 1 (juntas 1, 7): ~4.2 rad
- Grupo 2 (juntas 2, 8): ~3.6 rad
- Grupo 3 (juntas 3-6): ~1.3-2.5 rad

**Exp 8**: Fases mais concentradas
- **Junta 1 quase em zero**: 0.0065 rad (referência)
- **Maioria entre 2.3-3.7 rad** (mais sincronizadas)
- Distribuição mais uniforme

**Comparação**:
| Junta | Exp 7 | Exp 8 | Mudança |
|-------|-------|-------|----------|
| 1 | 4.166 | **0.007** | -99.8% (reset!) |
| 2 | 3.627 | 3.343 | -8% |
| 3 | 1.144 | 2.819 | +146% |
| 4 | 2.508 | 2.356 | -6% |
| 5 | 1.323 | 2.809 | +112% |
| 6 | 1.842 | 3.050 | +66% |
| 7 | 4.352 | 3.669 | -16% |
| 8 | 3.542 | 3.127 | -12% |

---

## 🎯 Por que o Experimento 8 foi Melhor?

### 1. **Elitismo Aumentado (3 → 10)**
- Preservou as **10 melhores soluções** a cada geração
- Evitou perda de boas características por mutação/crossover
- Convergência mais estável

### 2. **Pressão Seletiva Balanceada (tournament 5)**
- Meio termo entre exploração (3) e exploitation (7)
- Favorece bons indivíduos sem eliminar diversidade

### 3. **Reativação da Junta 6**
- Exp 7 havia zerado a junta 6
- Exp 8 descobriu que ela **contribui** (amplitude 0.099)
- Uso mais eficiente de todos os graus de liberdade

### 4. **Coordenação de Fase Melhorada**
- Junta 1 como **referência** (fase ~0)
- Outras juntas agrupadas em ~2.3-3.7 rad
- Padrão de marcha mais consistente

### 5. **Amplitudes Otimizadas**
- Juntas 2, 4, 8 aumentaram (principais motores)
- Juntas 1, 3, 5 reduziram (suporte/estabilização)
- Junta 6 reativada (contribuição moderada)

### 6. **Frequência Ajustada**
- 11.39 Hz (vs 10.82 Hz)
- Movimento ~5% mais rápido
- Provavelmente melhor trade-off velocidade/estabilidade

---

## 📈 Insights Importantes

### Papel do Elitismo
O aumento de elitismo de 3 para 10 foi **crucial**:
- **Exp 7** (elitismo 3): Perdeu boas características, junta 6 zerada
- **Exp 8** (elitismo 10): Preservou e refinou, todas as juntas ativas

### Distribuição de Trabalho
**Juntas principais** (amplitudes altas):
- Junta 2: 0.239
- Junta 4: 0.217
- Junta 8: 0.239

**Juntas de suporte** (amplitudes médias):
- Junta 5: 0.120
- Junta 6: 0.099
- Junta 7: 0.193

**Juntas de estabilização** (amplitudes baixas):
- Junta 1: 0.030
- Junta 3: 0.052

### Padrão de Coordenação
- **Junta 1**: Referência (fase ~0)
- **Juntas 2, 4, 7, 8**: Cluster principal (~3.1-3.7 rad)
- **Juntas 3, 5, 6**: Cluster secundário (~2.8-3.0 rad)
- **Diferença entre clusters**: ~0.5-0.9 rad

---

## 🏆 Conclusão

O **Experimento 8 alcançou o melhor fitness (2006.54)** devido a:

1. ✅ **Elitismo alto (10)**: Preservou as melhores soluções
2. ✅ **Tournament balanceado (5)**: Pressão seletiva moderada
3. ✅ **Todas as juntas ativas**: Uso completo dos graus de liberdade
4. ✅ **Coordenação refinada**: Fases bem agrupadas em clusters
5. ✅ **Distribuição inteligente**: Juntas principais, suporte e estabilização
6. ✅ **Frequência otimizada**: 11.39 Hz (sweet spot)

**Lição aprendida**: **Elitismo adequado (10-20% da população) é essencial** para preservar boas características enquanto permite exploração.

---

## 🔧 Recomendações para Próximos Experimentos

### Se quiser melhorar ainda mais:
1. **Testar elitismo 15**: Ver se preservar mais elite ajuda
2. **Aumentar gerações**: 300-500 gerações para refinamento
3. **Ajustar bounds de amplitude**: Permitir até 0.6-0.7 para juntas principais
4. **Testar representação de 8 genes**: Pernas simétricas + fases coordenadas

### Configuração atual está ótima:
- ✅ Elitismo: 10
- ✅ Tournament: 5
- ✅ Mutation rate: 0.05
- ✅ Adaptive strength: 0.25→0.05
- ✅ Representação: 17 genes (1 freq + 8 amps + 8 phases)