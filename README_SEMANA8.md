# Proyecto de Grado: Segmentación Panóptica para Minería Ilegal
## Semana 8 Status Report

### ✅ Logrado

1. **Baseline YOLO evaluado**
   - Modelo: yolov11_best100.pt
   - Recall en SDZI: **2.74%** (vs 42.5% esperado de Eyes in the Sky)
   - Debug completo: el problema es capacidad del modelo, no threshold de confianza
   - Archivo: `baseline_yolo_results.json`

2. **Panoptic FCN inferencia generada**
   - Checkpoint: `checkpoints/finetune_weighted_longer/epoch_500.pt`
   - 254 imágenes procesadas → mapas panópticos guardados en `predictions/epoch_500/panoptic/`
   - Metadata: `predictions/epoch_500/metadata.json`

3. **Evaluación inicial Panoptic FCN**
   - Recall: **0.0000** (cero instancias detectadas)
   - Archivo: `panoptic_eval_results.json`

### ⚠️ Hallazgos Críticos

#### 1. **Leakage en test set**
- Dataset actual tiene ~141 overlaps train-test confirmados
- Números de recall (YOLO 2.74%, Panoptic 0%) son **inválidos para investigación**
- **Blocker:** Esperar split leakage-safe de Jorge

#### 2. **Kernel/Mask-Feature heads NOT supervised**
- Training loop (`src/panoptic_mining/training/train.py`) está incompleto por diseño
- Stuff branch: ✅ supervisado contra class targets
- Thing heatmap: ✅ supervisado contra Gaussian heatmap
- **Kernel/mask-feature heads: ❌ NO supervisados** (línea 18-26 del archivo)
- Esto explica recall=0%: modelo detecta ubicaciones (heatmap) pero no genera máscaras válidas
- **Blocker:** Conectar pseudo-máscaras a loss del kernel generator

#### 3. **Supervisión por puntos no conectada**
- `data/points.py` genera pseudo-máscaras (GrabCut + fallbacks)
- `docs/decisions.md` lo documenta honestamente
- Pero training loop no las usa

#### 4. **Protocolo escalonado no seguido**
- Ya se evaluó directamente en test set (debería haber sido validación primero)
- No hay partición de validación implementada
- Contradice punto 4 de metodología

#### 5. **Barrido P5/P10/P20/P30 no existe**
- Skeleton creado: `baseline/sweep_supervision_levels.py`
- Requiere blockers #1 y #2 resueltos

### 📋 Plan Semana 9

**Blocker #1: Leakage-safe split**
- [ ] Revisar status con Jorge
- [ ] Si listo: re-evaluar YOLO y Panoptic FCN
- [ ] Si no: coordinar entrega

**Blocker #2: Conectar pseudo-máscaras**
- [ ] Revisar `data/points.py` y pseudo-mask generation
- [ ] Implementar mask loss en training loop para kernel heads
- [ ] Test: entrenar 1 epoch, verificar que loss converge
- [ ] Conexión oficial a train.py

**Debug: Por qué recall=0%**
- [ ] Correr `baseline/debug_instances.py` 
- [ ] Si model.decode_instances() retorna 0 para todos los class_ids → kernel loss es el culpable
- [ ] Si retorna instancias → problema es evaluación (clase equivocada?)

**Barrido de supervisión**
- [ ] Una vez #1 y #2 estén: ejecutar sweep P5/P10/P20/P30
- [ ] Evaluar en leakage-safe validation
- [ ] Métricas: PQ/SQ/RQ + recall@0.5

### 📁 Archivos nuevos semana 8

```
baseline/
  ├── debug_yolo.py              ✅ (corre con múltiples conf thresholds)
  ├── debug_instances.py          ✅ (skeleton para verificar si model.decode genera masks)
  ├── inference_panoptic_fcn.py   ✅ (genera predicciones panópticas)
  ├── panoptic_eval.py            ✅ (evalúa contra GT boxes)
  └── sweep_supervision_levels.py ✅ (skeleton P5/P10/P20/P30)

predictions/epoch_500/
  ├── panoptic/         ✅ mapas panópticos (.npy)
  ├── panoptic_vis/     ✅ visualización (.png)
  └── metadata.json     ✅

Resultados:
  ├── baseline_yolo_results.json         (recall=0.0274)
  └── panoptic_eval_results.json         (recall=0.0000, debugging)
```

### 🎯 Métrica de progreso

| Métrica | Semana 8 | Meta Semana 9 |
|---------|----------|---------------|
| Baseline YOLO recall | 2.74% | ✓ (línea base) |
| Panoptic FCN recall | 0% | >5% (con kernels supervisados) |
| Barrido P% | ✗ (skeleton) | ✓ (P5/P10/P20/P30 ejecutado) |
| Leakage-safe | ✗ (en progreso) | ✓ (aplicado a evaluaciones) |

### 📝 Notas para sustentación

**Honestidad sobre brechas:**
- "La supervisión por puntos está codificada pero no conectada al training aún"
- "El modelo actual no está supervisado para generar instancias válidas (kernel loss faltante)"
- "Los números de recall se corregiré n una vez Jorge provea el split leakage-safe"

Esto es defendible porque:
1. Está documentado (no oculto)
2. Hay plan claro para resolverlo
3. Cosas importantes SÍ están hechas (inferencia, evaluación structure, baseline)

---

## ⚠️ CAVEAT CRÍTICO: Leakage en Dataset

**Todos los números de recall reportados en semana 8 están evaluados en dataset_split_completo ORIGINAL, que contiene ~141 overlaps train-test confirmados.**

Esto significa:
- Recall YOLO (2.74%) es potencialmente inflado
- Recall Panoptic FCN es potencialmente inflado
- Números NO son válidos para investigación/comparativa

**Acción requerida antes de semana 9:**
- Jorge proporciona split leakage-safe
- Re-evaluar YOLO y Panoptic FCN en split limpio
- Reportar números CORRECTOS

**Para presentación jueves 25:**
Incluir caveat: "Evaluaciones preliminares en dataset con leakage. Números se recalcularán en split limpio."

---

## Evaluación Correcta (Semana 8, tarde)

Script: `baseline/panoptic_eval_fixed.py`
- Evalúa TODAS las instancias que genera Panoptic FCN (sin filtrar por clase)
- Compara contra SDZI GT boxes
- Reporta recall en `panoptic_eval_fixed.json`

