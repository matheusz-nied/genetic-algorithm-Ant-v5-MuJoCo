# Análise do Experimento 7 

## Comparação: Experimento 6 → Experimento 7

### 📊 Fitness
- **Experimento 6**: 1316.35
- **Experimento 7**: 1853.35 ✅ **(+40.8% de melhoria)**

---
## Mudanças algoritmo
- Força da mutação adaptativa
- Frequencia compartilhada: Afinal a sincronização do movimento das pernas é o que mais importa 

## 🔄 Mudança Fundamental: Representação dos Genes
### Experimento 6 (24 genes)
- **8 frequências independentes** (uma por junta)
- **8 amplitudes** (uma por junta)
- **8 fases** (uma por junta)
- **Total**: 24 parâmetros

### Experimento 7 (17 genes)
- **1 frequência compartilhada** (todas as juntas)
- **8 amplitudes** (uma por junta)
- **8 fases** (uma por junta)
- **Total**: 17 parâmetros

**Impacto**: Redução de **29% no espaço de busca** → convergência mais eficiente

---

## ⚙️ Mudanças nos Hiperparâmetros

### Elitismo e Seleção
- **elitism_count**: 10 → **3** (preserva menos elite, mais diversidade)
- **tournament_size**: 7 → **3** (pressão seletiva menor)

### Mutação
- **mutation_rate inicial**: 0.01 → **0.05** (5x maior, mais exploração)
- **min_mutation_rate**: 0.01 → **0.05** (piso mais alto)
- **max_mutation_rate**: 0.05 → **0.12** (teto 2.4x maior)
- **mutation_strength**: 0.1 → **0.25→0.05** (adaptativa linear)
- **enable_adaptive_strength**: ❌ → **✅** (novo recurso)

---

## 🔍 Análise dos Parâmetros Evoluídos

### Frequências
**Exp 6**: 8 frequências dispersas (9.05 a 12.18 Hz)
- Média: ~10.9 Hz
- Variação: alta (cada junta independente)

**Exp 7**: 1 frequência compartilhada
- **10.82 Hz** (próxima à média do Exp 6)
- Movimento sincronizado entre todas as juntas

### Amplitudes
**Exp 6**: Amplitudes pequenas e desbalanceadas
- Range: 0.028 a 0.220
- Média: ~0.11
- Junta 6 tinha amplitude alta (0.17)

**Exp 7**: Amplitudes mais balanceadas e maiores
- Range: 0.000 a 0.222
- Média: ~0.14 (+27%)
- **Junta 6 zerada** (GA descobriu que ela não contribui)
- Juntas 2, 4, 7, 8 com amplitudes maiores (0.18-0.22)

### Fases
**Exp 6**: Fases dispersas
- Range: 0.65 a 4.38
- Pouca coordenação aparente

**Exp 7**: Fases mais coordenadas
- Range: 1.14 a 4.35
- Grupos de coordenação:
  - Juntas 1, 7: ~4.2 rad (próximas)
  - Juntas 2, 8: ~3.6 rad (próximas)
  - Juntas 4, 5, 6: ~1.3-2.5 rad (intermediárias)

---

## 🎯 Por que o Experimento 7 foi Melhor?

### 1. **Representação Mais Eficiente**
- Frequência compartilhada força movimento sincronizado
- Espaço de busca 29% menor → GA explora melhor
- Menos parâmetros conflitantes

### 2. **Mutação Adaptativa de Força**
- Início: mutações fortes (0.25) para exploração ampla
- Final: mutações suaves (0.05) para refinamento
- Balanceou exploração e exploitation automaticamente

### 3. **Taxa de Mutação Mais Alta**
- Começou 5x maior (0.05 vs 0.01)
- Evitou convergência prematura
- Manteve diversidade por mais tempo

### 4. **Descoberta Inteligente**
- GA descobriu que **Junta 6 não contribui** (amplitude = 0)
- Concentrou esforço nas juntas mais efetivas
- Amplitudes maiores nas juntas críticas (2, 4, 7, 8)

### 5. **Movimento Coordenado**
- Frequência única (10.82 Hz) criou ritmo consistente
- Fases bem distribuídas geraram padrão de marcha eficiente
- Sincronização natural entre juntas

---

## 📈 Conclusão

O **Experimento 7 superou o 6 em 40.8%** devido a:

1. ✅ **Representação compacta** (17 vs 24 genes)
2. ✅ **Mutação adaptativa de força** (0.25→0.05)
3. ✅ **Taxa de mutação alta** (0.05 inicial)
4. ✅ **Frequência compartilhada** forçou sincronização
5. ✅ **Descoberta automática** de juntas irrelevantes

**Lição aprendida**: Reduzir o espaço de busca com **constraints inteligentes** (frequência compartilhada) é mais eficaz que dar liberdade total ao GA.

---

## 🔧 Próximos Passos Sugeridos

1. **Testar representação de 8 genes** (pernas simétricas + fases coordenadas)
2. **Aumentar elitismo** para 5-10 (preservar melhores soluções)
3. **Ajustar bounds de amplitude** se movimento estiver saturando
4. **Analisar vídeo** para validar qualidade do movimento