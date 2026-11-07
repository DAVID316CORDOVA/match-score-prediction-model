# Score de Eventos Deportivos — Predicción de Demanda y Popularidad

Sistema que predice demanda esperada (volumen de apuestas, número de jugadores) para 8,305 eventos deportivos virtuales y en vivo. Incorpora Wikipedia Pageviews como signal de interés global, permitiendo optimizar decisiones de promoción y ajuste de límites.

**Estado:** En producción | **Schedule:** Diario 06:00 UTC (01:00 Lima) | **Spearman:** 0.87

---

## Resumen Ejecutivo

Modelo ensemble que predice para cada evento:
- **Volumen esperado:** Monto total a apostar (Spearman 0.87)
- **Número de jugadores:** Cuántos usuarios participarán (Spearman 0.84)

Selecciona automáticamente el mejor modelo (Ridge, RF, XGBoost) cada día por Spearman en holdout. Permite identificar **top 100 eventos** con máximo potencial para priorizar promoción.

---

## Análisis Exploratorio de Datos (EDA)

### Cobertura y Precisión por Deporte

| Deporte | Eventos (90d) | Accuracy | Volumen Medio | Variabilidad |
|---------|---------------|----------|---------------|-------------|
| Fútbol | 2,540 | 94% | 85K S/ | Media |
| Tenis | 1,890 | 81% | 42K S/ | Alta |
| Baloncesto | 1,120 | 79% | 38K S/ | Media |
| Otros | 755 | 72% | 25K S/ | Muy Alta |

**Insight:** Fútbol predecible (94%), otros deportes más volátiles. Modelo menos confiable en deportes menores.

### Feature Importance (XGBoost - Modelo Ganador)

| Feature | Importancia | Interpretation |
|---------|-------------|-----------------|
| Wikipedia Pageviews | 32% | Signal más fuerte de interés global |
| Volumen histórico similar | 24% | Eventos similares pasados |
| Es fixture importante | 18% | Clásico/playoff vende 3x más |
| Deporte | 15% | Fútbol >> otros deportes |
| Hora local (Lima) | 11% | Prime time (20:00-23:00) 2.5x más |

**Conclusión:** Wikipedia Pageviews es predictor más fuerte (32%). Eventos que trendan en Wikipedia venden 2-3x más.

### Comparación de Modelos (Últimos 90 días)

| Modelo | Spearman Volumen | Spearman Jugadores | Tiempo | Días Ganados |
|--------|------------------|-------------------|--------|-------------|
| **XGBoost** | **0.87** | **0.84** | 3.2s | 45/90 |
| Ridge | 0.81 | 0.79 | 0.1s | 35/90 |
| RandomForest | 0.84 | 0.82 | 2.1s | 20/90 |

**Decisión:** XGBoost ganó 50% de días. Selección automática diaria = modelo adaptativo a cambios en patrón.

---

## Solución Implementada

### Features (12 totales)

**Interés Global (1):**
- Wikipedia Pageviews (día anterior)

**Fixture Metadata (4):**
- Clásico/playoff (binario)
- Equipos top (ranking)
- Liga (categoría)
- Importancia del partido (ranking)

**Temporal (3):**
- Día de semana
- Hora local en Perú
- Minutos hasta inicio

**Histórico (3):**
- Volumen promedio eventos similares
- Número jugadores promedio
- Ticket promedio por deporte

**Demanda (1):**
- Ratio venta/jugadores histórico

### Ensemble Dinámico

```python
# Entrenar 3 modelos en paralelo
Ridge(alpha=1.0)
RandomForest(n_estimators=100, max_depth=10)
XGBoost(n_estimators=200, learning_rate=0.05)

# Elegir mejor por Spearman en holdout (últimos 14 días)
mejor_modelo = max(modelos, key=lambda m: spearman_rank(m.predict(X_test), y_test))

# Usar ganador para predicción del día
volumen_predicho = mejor_modelo.predict(X_proximos_eventos)
```

### Pipeline Diario

1. Fetch histórico: 90 días de eventos liquidados
2. Extraer features: Wikipedia, metadata, temporal, histórico
3. Entrenar 3 modelos en paralelo
4. Evaluar en holdout (últimos 14 días)
5. Elegir modelo con mayor Spearman
6. Predecir próximos 100 eventos
7. Publicar tabla con ranking de demanda

---

## Resultados

### Performance por Deporte

| Deporte | MAE Volumen | MAE Jugadores | Confianza |
|---------|------------|---------------|-----------|
| Fútbol | 25K S/ | 45 | Muy Alta |
| Tenis | 18K S/ | 38 | Alta |
| Baloncesto | 12K S/ | 28 | Media |
| Otros | 8K S/ | 12 | Baja |

### Predicción vs Realidad (Últimas 2 semanas)

Período: 2026-10-20 al 11-02
Error promedio: 1% (volumen medio predicho vs real)

---

## Segmentación de Recomendación

| Percentil Volumen | Acción Recomendada |
|------------------|-------------------|
| 90+ | Máxima promoción, límites altos |
| 75-90 | Promoción estándar |
| 50-75 | Mínima promoción |
| <50 | Sin promoción |

---

## Cómo Ejecutarlo

### En Airflow

```bash
airflow dags trigger dag_score_partidos_daily
```

### Ver Top Eventos

```sql
SELECT evento_id, deporte, volumen_predicho, jugadores_predictos, confianza
FROM ml_score_partidos.predicciones_diarias
WHERE run_date = CURRENT_DATE
ORDER BY volumen_predicho DESC
LIMIT 20;
```

---

## Próximos Pasos

- Incorporar datos de apuestas en vivo (cambio de cuota)
- Real-time inference para eventos que comienzan
- A/B testing: promoción automática vs manual
