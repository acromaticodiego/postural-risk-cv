# Los datos

Nada de esta carpeta va a git (ver `.gitignore`). Aquí queda escrito **de dónde
salió cada cosa**, que es lo que permite que otro reproduzca los números.

```
data/
  raw/         los datasets tal como se descargan, sin tocar
  keypoints/   lo que produce el extractor: arrays (T, 17, 3) por secuencia
```

---

## 1. UW-IOM — el conjunto principal · DESCARGAR ESTE PRIMERO

**University of Washington Indoor Object Manipulation Dataset**

- **Descarga:** https://data.mendeley.com/datasets/xwzzkxtf9s/2
- Versión 2, publicada el 15 de junio de 2019.
- Autores: Behnoosh Parsa, Ekta U. Samani, Rose Hendrix, Shashi M. Singh,
  Santosh Devasia, Ashis G. Banerjee.
- Paper que lo acompaña: *Toward Ergonomic Risk Prediction via Segmentation of
  Indoor Object Manipulation Actions Using Spatiotemporal Convolutional
  Networks* — https://arxiv.org/pdf/1902.05176

**Qué contiene**, según la ficha oficial:

- **20 participantes** de 18 a 25 años: 15 hombres y 5 mujeres.
- Grabado con **Kinect Sensor for Xbox One** a unos **12 fotogramas por segundo**.
- **Vídeo e información de seguimiento esquelético**, o sea que trae esqueletos
  además de las imágenes.
- **17 etiquetas de acción en una jerarquía de cuatro niveles**: (1) si se
  manipula la caja o la varilla, (2) el movimiento del cuerpo —caminar, estar de
  pie, doblarse—, (3) el tipo de manipulación —alcanzar, recoger, colocar,
  sostener— y (4) **la altura relativa de la superficie** donde ocurre: baja,
  media o alta.

**Por qué este y no otro:** la altura de la superficie es la variable que cambia
el puntaje ergonómico de un levantamiento real, y son 20 sujetos distintos, que
es lo que permite partir por persona y no por clip.

### LO QUE HAY DE VERDAD DENTRO (verificado el 2026-09-29)

Descargado y abierto. La estructura real es:

```
xwzzkxtf9s-2/UW IOM Dataset/
  JointPositions/   1.mat … 20.mat     15 MB    esqueletos
  VideoLabels/      01.txt … 20.txt   756 KB    etiquetas, una por fotograma
  Videos/           01.7z … 20.7z     8,5 GB    los vídeos, SIN descomprimir
```

**Ojo al emparejar: los esqueletos van sin cero (`1.mat`) y las etiquetas con cero
(`02.txt`).** Ordenar alfabéticamente pone `10.mat` antes de `2.mat`, y emparejar
el esqueleto del sujeto 10 con las etiquetas del 2 no da ningún error: da un
dataset mezclado que rinde mal sin decir por qué.

Cada `.mat` es MATLAB v7.3, o sea HDF5: se lee con `h5py`, no con `scipy.io`.
Contiene cuatro cosas, y hay más de lo prometido:

| campo | forma | qué es |
|---|---|---|
| `bodylogger3D` | (T, 25, 3) | esqueleto **3D** del Kinect, 25 articulaciones, en metros |
| `pos2Dcolor` | (T, 25, 2) | las mismas articulaciones proyectadas sobre la imagen de color |
| `bodytimelogger` | (T, 1) | marca de tiempo de cada esqueleto |
| `videotimelogger` | (Tv, 1) | marca de tiempo de cada fotograma de vídeo |

El orden de articulaciones es el del SDK del **Kinect v2** (0 = base de la
columna, 20 = columna a la altura de los hombros). No está documentado en el
dataset: se comprobó midiendo la inclinación del tronco, que da p50 de 8,1° y p95
de 69,1° en el sujeto 1 —erguido de pie, doblado al doblarse—, que es lo que
tendría que salir si el orden es el supuesto.

**Las etiquetas son las 17 de acción, NO son puntajes REBA.** Varios trabajos que
usan este dataset hablan de anotaciones REBA por fotograma y aquí no vienen. El
formato es una línea por fotograma con cuatro campos separados por guiones bajos:

```
objeto _ movimiento _ manipulación _ altura
box|rod|none _ walk|stand|bend _ pick-up|place|hold|reach|none _ low|mid|top|none
```

El puntaje de riesgo se calcula de los ángulos del esqueleto, que era el plan de
todas formas, y estas etiquetas son lo que aprende el modelo.

### Los dos hallazgos que cambian el diseño

**1. El fps real va de 7,81 a 10,58 y cambia con cada sujeto.** La ficha dice unos
12. Medido de `videotimelogger` en los 20 sujetos. Consecuencia: una ventana de 30
fotogramas son 3,8 s en el sujeto 20 y 2,8 s en el sujeto 9, así que **las
ventanas temporales se definen en segundos y se remuestrea**; definirlas en
fotogramas convierte al sujeto en una variable oculta del experimento.

**2. Las etiquetas están alineadas al FINAL de la secuencia.** Hay entre 28 y 109
etiquetas menos que fotogramas de esqueleto, y el hueco está al principio. No se
supuso: se midió con `scripts/probe_alignment.py`, comparando cuánto más inclinado
está el tronco en los fotogramas `bend` que en los `stand` bajo cada hipótesis.

| hipótesis | separación, mediana de 20 sujetos |
|---|---|
| etiquetas desde el primer fotograma | 4,11° |
| **etiquetas alineadas al final** | **21,75°** |
| por marca de tiempo, desde el principio | 1,24° |
| por marca de tiempo, al final | 21,75° |

Gana alinear al final, y en 18 de 20 sujetos por separado. El mapeo por marca de
tiempo da **el mismo resultado exacto** que el directo, así que los dos relojes ya
van alineados y el adaptador no necesita interpolar por tiempo.

**El sujeto 3 es la excepción y queda marcado como sospechoso:** da −2,67°, o sea
que sus fotogramas `bend` salen *menos* inclinados que los `stand`, que es
imposible con la alineación correcta. Qué hacer con él es una decisión de criterio
sin tomar; lo que no se hace es meterlo en el conjunto como si nada.

### Cómo dejarlo

Descomprimir dentro de `data/raw/uw-iom/`, conservando la estructura original
tal como viene. No renombrar carpetas: el adaptador se escribe contra la
estructura original a propósito, para que se pueda volver a descargar y cuadre.

Cuando termine, decírselo a Claude y **pegar la salida de**:

```powershell
Get-ChildItem -Recurse data\raw\uw-iom | Select-Object -First 30 FullName, Length
```

Eso es lo que hace falta para escribir el adaptador sin adivinar.

---

## 2. InHARD — el conjunto de fábrica · MÁS ADELANTE

**Industrial Human Action Recognition Dataset**, para comprobar si lo aprendido
en laboratorio aguanta un entorno industrial real con cobots.

- **Descarga:** https://zenodo.org/records/4003541 — **50 GB** repartidos en
  varios archivos `.7z`.
- Repositorio: https://github.com/vhavard/InHARD
- 16 sujetos, 13 clases de acción industrial, más de 2 millones de fotogramas,
  más de 4800 muestras de acción, con RGB, esqueleto y profundidad.

**No descargar todavía.** Son 50 GB y no hace falta hasta la fase 4.

---

## 3. RULA_2DImage — referencia de contraste · OPCIONAL

Posturas de levantamiento etiquetadas con RULA, tomadas en laboratorio y
completadas con Human3.6M: https://github.com/LLDavid/RULA_2DImage

Sirve para contrastar el cálculo geométrico contra etiquetas RULA de otra fuente.
No es necesario para ninguna fase; es material de verificación cruzada.

---

## 4. Grabaciones propias — la prueba de la realidad · CUANDO TOQUE LA FASE 4

La pregunta que haría un cliente es si el sistema aguanta **otra sala, otra
cámara y otras personas**. Eso se responde con material grabado fuera del
laboratorio, y es lo único de esta lista que hay que producir.

Cuando llegue el momento se escribe aquí el protocolo exacto: tareas, ángulos de
cámara, distancias y número de personas. No se graba antes de tener el sistema
midiendo, para no grabar lo que luego resulte que no servía.
