# Protocolo de verificación de puntos de referencia

Las métricas de este proyecto miden cada mapa contra **lo que una persona ve en imágenes de alta resolución**, no contra WorldCover. WorldCover también tiene errores, y el modelo se entrenó con él. Este documento explica cómo etiquetar los puntos para que la comparación sea justa y repetible.

## Diseño de la muestra

- **Muestreo aleatorio estratificado:** 70 puntos por clase de WorldCover, 490 en total, **solo dentro de los bloques de prueba**. Ningún modelo vio esos bloques durante el entrenamiento.
- Los estratos no coinciden con las clases de ningún mapa evaluado. Por eso las exactitudes y áreas se estiman con los estimadores de Stehman (2014), que corrigen el peso de cada estrato (`src/sabana_lc/evaluate.py`).
- **Unidad de evaluación:** el píxel Sentinel-2 de 10 × 10 m que contiene el punto.

## Etiquetado a ciegas

1. En QGIS, carga `ref_points.csv` como *texto delimitado* (X = `lon`, Y = `lat`, EPSG:4326).
2. **No abras `ref_strata.csv` hasta terminar.** Si sabes qué clase dijo WorldCover, la etiqueta se sesga.
3. Imágenes de apoyo, en este orden:
   - la ortofoto de Bogotá de 2021–2022 (Mapa de Referencia de IDECA, para la zona urbana y los Cerros);
   - Google Satellite o ESRI World Imagery, revisando con la línea de tiempo de Google Earth Pro la fecha más cercana a **2021**;
   - el compuesto `s2_2021_hls6.tif` en falso color (B8A-B11-B4), para confirmar vegetación y agua.
4. En cada punto, mira un cuadro de **10 × 10 m** centrado en él y asigna la clase que cubre **más de la mitad** del cuadro.
5. Llena `ref_labels.csv` con estas columnas:

```
id,ref_class,confianza,nota
1,3,alta,
2,5,media,invernaderos
3,,baja,nube en todas las imágenes de 2021
```

`ref_class` vacío significa que el punto no se puede interpretar: queda excluido y se reporta cuántos fueron.

## Clases

| Código | Clase | Incluye | No incluye |
|---|---|---|---|
| 0 | Agua | embalses, ríos anchos, humedales con lámina de agua visible | humedal cubierto de juncos (→ 3) |
| 1 | Bosque / árboles | bosque nativo, plantaciones de eucalipto y pino, arbolado denso urbano (dosel > 50 %) | arbolado disperso en pastos (→ 3) |
| 2 | Arbustal | matorral alto andino, rastrojo, frailejonal denso | pasto con arbustos aislados (→ 3) |
| 3 | Pastos / herbazal | potreros, canchas de césped, juncales, páramo de pajonal | cultivos con surcos visibles (→ 4) |
| 4 | Cultivos | papa, hortalizas, cereales, flores **a cielo abierto**, lotes arados | invernaderos (→ 5) |
| 5 | Construido | edificaciones, vías pavimentadas, parqueaderos, **invernaderos de flores** | lotes sin construir (→ 6 o 3) |
| 6 | Suelo desnudo | canteras, áreas de extracción, lotes en obra sin cubierta, roca | lote arado para siembra (→ 4) |

**Casos frecuentes en la Sabana:** invernaderos → 5. Canteras de los Cerros → 6. Eucalipto en los Cerros → 1. Humedales como Juan Amarillo, La Conejera o Jaboque: lámina de agua → 0, juncal → 3.

## Tiempo estimado

Unas 4 horas para 490 puntos, a unos 30 s por punto. Se puede hacer por partes, porque el archivo se va llenando por `id`.

## Referencias

- Olofsson, P., Foody, G. M., Herold, M., Stehman, S. V., Woodcock, C. E., & Wulder, M. A. (2014). *Good practices for estimating area and assessing accuracy of land change.* Remote Sensing of Environment, 148, 42–57.
- Stehman, S. V. (2014). *Estimating area and map accuracy for stratified random sampling when the strata are different from the map classes.* International Journal of Remote Sensing, 35(13), 4923–4939.
