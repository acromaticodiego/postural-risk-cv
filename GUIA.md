# GUÍA DEL PROYECTO — qué hay que hacer, en orden

Este fichero es la fuente de verdad del proyecto. Se actualiza cada sesión.
Si abres el proyecto después de días y no recuerdas nada, lee solo esto.

**El reparto:** Juan Diego entrena los modelos y toma las decisiones de criterio.
Claude construye el arnés de datos, la evaluación, el pipeline de inferencia y la
aplicación, y mantiene esta guía diciendo explícitamente qué toca hacer.

---

## VER EL PANEL

```powershell
Set-Location C:\Users\ASUS\Desktop\pose_stimation
.\.venv\Scripts\python.exe scripts\build_demo_data.py      # genera los datos, ~3 min
.\.venv\Scripts\python.exe -m uvicorn src.api.server:app --port 8000
# y abrir http://127.0.0.1:8000/
```

Tres pantallas: el ranking de puestos, el informe de un puesto con su
recomendación y su bandeja de eventos, y el replay en esqueleto de cada evento.

**Dos cosas del panel que son decisiones, no detalles:**

  · **Cada turno se predice con el modelo del pliegue que NO lo vio entrenando.**
    La demo enseña predicciones sobre gente que el modelo no conoce, igual que en
    una planta. Enseñar predicciones sobre los datos de entrenamiento daría una
    demo más lucida y sería mentir.
  · **Ningún recurso externo.** Una librería por CDN puede tardar o fallar justo
    en la toma del vídeo y dejar la página sin estilos. Hay una prueba que se
    niega a dejar entrar un `src` o `href` remoto.

### El suelo de riesgo de un puesto, y por qué importa

Medido el 30/09: una persona **de pie, erguida, sin hacer nada**, en un puesto
configurado con 18 kg, agarre malo y carga brusca, ya puntúa **REBA 4**, que es el
umbral de acción.

| configuración del puesto | REBA de una postura neutra |
|---|---|
| sin configurar | 1 |
| 12 kg, agarre regular | 3 |
| 18 kg, agarre malo | 3 |
| 18 kg, agarre malo, carga brusca | **4 — ya es riesgo medio** |

No es un fallo: la norma dice que manipular 18 kg con mal agarre es riesgo aunque
estés erguido, y tiene razón. Pero la consecuencia para el producto es incómoda y
hay que decirla: **cuanto más pesada es la carga declarada, menos aporta la
cámara**, porque el componente fijo ya decide por encima del umbral y la postura
deja de cambiar el resultado.

La salida, apuntada y **no implementada**: reportar además el **exceso postural**
—cuánto añade la postura sobre el REBA que ese puesto tendría con una postura
neutra—, que es una cifra en la que la visión siempre aporta. Mientras tanto, el
puesto de la demo se configuró sin `sudden_load` para que no saliera el 100% del
tiempo en rojo.

## DÓNDE ESTAMOS (actualizado 2026-10-02)

**El sistema está completo de punta a punta y funcionando**: vídeo → esqueletos →
riesgo según REBA → peso aceptable según NIOSH → tarea reconocida por el modelo →
informe accionable → panel web con modo en vivo. **126 pruebas en verde**, 33
commits, y el README ya existe.

```powershell
Set-Location C:\Users\ASUS\Desktop\pose_stimation
.\.venv\Scripts\python.exe -m uvicorn src.api.server:app --port 8010
# http://127.0.0.1:8010/     (el 8000 lo ocupa el agente de voz de Juan Diego)
```

### Las piezas

| módulo | qué hace |
|---|---|
| `src/pose/` | de vídeo a esqueletos. El fotograma se destruye en la misma iteración |
| `src/baseline/reba.py` | REBA geométrico, consciente de la confianza y del ángulo de cámara |
| `src/baseline/niosh.py` | la ecuación de levantamiento: el peso aceptable en kilos |
| `src/baseline/tecnica.py` | si el riesgo es evitable o es el mínimo de la tarea |
| `src/baseline/rules.py` | la línea base sin aprendizaje: el número a batir |
| `src/models/tcn.py` | el TCN causal que reconoce la tarea |
| `src/load/carga.py` | la fusión de persona y carga (detector entrenado, sin validar) |
| `src/load/prelabel.py` | SAM guiado por el esqueleto y los filtros que deciden qué máscara vale |
| `src/product/` | puesto de trabajo, exposición, informe por tareas, pipeline |
| `src/api/` | FastAPI + panel web, con modo en vivo por WebSocket |
| `src/eval/` | partición por sujeto, métricas, arnés común |

### Los números publicables

| campo | degenerado | reglas | **TCN** | rango |
|---|---|---|---|---|
| movimiento | 0,288 | 0,670 | **0,870** | 0,829–0,882 |
| altura de trabajo | 0,174 | 0,639 | **0,798** | 0,765–0,823 |
| **manipulación** | 0,114 | 0,046 | **0,741** | 0,650–0,788 |
| objeto | 0,195 | 0,123 | **0,784** | 0,738–0,812 |

4 pliegues de validación cruzada por sujeto, 10 Hz, ventana causal de 2 s, 78.095
parámetros, 17 s de entrenamiento por pliegue. **El reservado (sujetos 11, 14, 19,
20) sigue sin tocarse.**

### Lo que se descubrió grabando con la webcam (29–30/09)

Cuatro vídeos de Juan Diego probando el sistema en su salón. Cada uno destapó algo:

1. **El modelo de tareas no generaliza al ángulo de su cámara.** Decía `bend` con
   el tronco a 2°. Arreglado con un guardia de coherencia: cuando la tarea
   contradice la geometría, el sistema lo dice en vez de mostrarla como si tal.
2. **La pose falla donde REBA más la necesita.** Codo, muñeca y tobillos caen por
   debajo de 0,5 de confianza en el 21–24% de los fotogramas, y el cálculo no
   miraba la confianza. **Subir de modelo no lo arregla** (yolo11n 23% de malas,
   yolo11m 18% por 70 ms más): es oclusión. Arreglado usando el lado más visible y
   marcando lo que no se ve.
3. **El cuello dependía de las orejas**, que es lo que peor detecta YOLO. Pasó de
   60% de fotogramas no fiables a 6% usando cualquier articulación facial visible.
4. **REBA da el mismo puntaje a la técnica buena y a la mala.** Medido en su
   vídeo: agacharse doblando la espalda da REBA 4, y en cuclillas con la espalda a
   12° también da 4. La norma compensa lo que se gana en tronco con lo que se
   pierde en piernas. Arreglado separando el riesgo **evitable** del **inherente**.
5. **Y el peor de todos: de frente, la flexión del tronco es invisible.** Agacharse
   doblando la espalda sale naranja de perfil y VERDE de frente. Inclinarse hacia
   la cámara no produce ningún ángulo en la imagen. Era un falso negativo de
   seguridad, que es lo peor que puede hacer un sistema así. Ahora el sistema
   detecta la vista frontal y declara el tronco como no medible.

## LO QUE TIENES QUE HACER AHORA

**Una sola cosa, y sin ella no avanza nada de la carga: grabar dos o tres vídeos con
la cámara en crudo.** Todo lo demás está montado y esperando material.

```powershell
.\.venv\Scripts\python.exe scripts\record_loads.py --nombre carga_lateral
# q o Esc para parar. Un nombre por objeto, un vídeo por objeto.
```

Qué grabar importa más que el guion, y está entero en la cabecera del fichero: **la
carga de verdad**, **dos o tres objetos distintos y cada uno en su propio vídeo**
—la partición del afinado es por vídeo—, **levantando, cargando y soltando**, desde
el suelo y desde una mesa, y **de lado**, que es además la única vista desde la que
REBA se puede medir. Dos o tres minutos por vídeo basta: de 50 segundos salieron 138
fotogramas candidatos.

Después, y eso ya es una tarde:

```powershell
.\.venv\Scripts\python.exe scripts\extract_load_frames.py --video "data\carga\videos\carga_lateral.mp4" --fuente camara
.\.venv\Scripts\python.exe scripts\prelabel_sam.py                 # SAM propone
# abrir data\carga\revision.html, descartar las malas, guardar ahí rechazadas.txt
.\.venv\Scripts\python.exe scripts\finetune_load.py --val-video carga_lateral
```

El último es un entrenamiento y **lo lanzas tú**, como todos. El lote por defecto es
16, el mismo con el que entrenaste el modelo base, y en la 3050 de 6 GB no cabe mucho
más; `--batch` está expuesto por si acaso.

### 1. La detección de carga: el arnés está hecho, falta el material

Juan Diego entrenó un detector de segmentación con `package-seg` (2.197 imágenes de
cajas de almacén, 100 épocas, **mAP50 de 0,935 en máscaras**). Los pesos están en
`artifacts/modelo/carga.pt`.

**Pero no se sabe si sirve aquí.** Probado sobre su vídeo: 0 de 89 fotogramas con
caja detectada. La comprobación **no concluye nada**, porque lo que sostenía era un
organizador de plástico transparente y el dataset son cajas de cartón opacas: el 0%
tiene dos causas indistinguibles. Está escrito como la medición falsa nº 5.

La salida era afinar ese modelo con sus propios vídeos, pre-etiquetando **con SAM
guiado por el esqueleto** en vez de a mano. **El flujo está montado entero**, y las
decisiones con lo que se descartó están en el
[ADR 0002](docs/adr/0002-afinar-el-detector-de-carga-con-el-material-del-cliente.md):

| guion | qué hace |
|---|---|
| `scripts/record_loads.py` | graba la cámara en crudo, sin pasar por el navegador |
| `scripts/extract_load_frames.py` | saca los fotogramas con persona y muñecas a la vista |
| `scripts/prelabel_sam.py` | SAM propone, los filtros descartan, y sale `revision.html` |
| `scripts/finetune_load.py` | partición por vídeo, mezcla con las de fábrica, mide antes y después |

#### Lo que apareció al mirar el material antes de construir nada encima

**Los cuatro vídeos de la webcam son capturas de pantalla del panel**, con el
esqueleto PINTADO encima de la persona y de la carga. Afinar con eso enseñaría al
modelo a buscar líneas de colores, y **no se habría notado**, porque la validación
lleva las mismas líneas. Es la medición falsa nº 6.

Por eso `extract_load_frames.py` exige `--fuente` y `finetune_load.py` **se niega** a
entrenar con material de pantalla: la garantía vive en el código, no en un aviso.

Y lo que el ensayo sobre ese material deja medido, que orienta sin ser una tasa:

| | |
|---|---|
| fotogramas candidatos de los 4 vídeos | 630 |
| propuestas que pasan los filtros | **81** |
| veces que SAM devolvió la silueta de la persona | **335 de 630**, con hasta 7 puntos negativos encima |

Ese último número es el que justifica que los filtros existan: **los puntos negativos
son una sugerencia, no una garantía.** Lo que garantiza es `judge_mask`, y encima de
él, una persona mirando.

### 2. Lo que falta para el vídeo de LinkedIn

  · ~~**El README del repositorio.**~~ HECHO el 2026-10-02. Abre con el problema y su
    coste, lleva el replay en esqueleto como portada —que *es* el argumento de
    privacidad— y una sección de lo que el sistema NO hace. **Falta una cosa tuya**:
    enlazar una por una las cifras de incidencia y coste (Sistema General de Riesgos
    Laborales, Fasecolda). Están marcadas en el propio README como pendientes y
    citadas como orden de magnitud mientras tanto.
  · **Grabar con la cámara de lado.** Es la única vista desde la que REBA se puede
    medir, y ahora el sistema avisa cuando no lo está. **Sale gratis con los vídeos de
    la carga** si los grabas de lado, que es lo que pide el guion de arriba.
  · **El nombre del repositorio**, que sigue sin decidirse y es tuyo: `pose_stimation`
    describe la técnica en vez del problema, y lleva una errata.

### 3. Lo apuntado y no hecho

  · **El exceso postural**: cuánto añade la postura sobre el mínimo de ese puesto.
    Nace de que un puesto con carga pesada declarada sale en riesgo aunque la
    persona esté erguida, y entonces la cámara deja de aportar.
  · **Las fases 4 y 5** de este documento. Ojo con la 5: la medición del 30/09 dice
    que el tiempo no está en la red, así que cuantizar podría no arreglar nada.

---

## EL PROBLEMA, y por qué alguien paga por resolverlo

Los **desórdenes musculoesqueléticos son la primera causa de enfermedad laboral
en Colombia**: más del 65% de todos los diagnósticos reportados al Sistema
General de Riesgos Laborales, más de **3 millones de días de incapacidad al año**,
y cerca del **35% de las solicitudes de incapacidad**. Un caso calificado como
enfermedad laboral pasa de **$25 millones de pesos** entre tratamiento,
rehabilitación e incapacidades, y el dolor lumbar concentra alrededor del **80%
de las indemnizaciones de origen laboral**.

**Cómo se mide hoy:** un ingeniero de seguridad y salud en el trabajo observa a
un operario durante media hora con un portapapeles, calcula un puntaje REBA a
mano, y extrapola esa muestra a un turno de ocho horas y a cincuenta
trabajadores. El problema no es que no sepan medirlo. El problema es que **solo
pueden medir una muestra ridícula**, y las intervenciones se priorizan con eso.

**Qué hace el sistema:** mide la exposición postural de todos los operarios
durante todo el turno, puntúa contra la norma, y dice qué puesto de trabajo
concentra el riesgo y si una capacitación cambió algo.

### La objeción que mata a estos sistemas, y que este ataca de frente

La literatura del área es explícita: los sistemas de ergonomía por visión están
frenados por **las objeciones de privacidad del trabajador** —hay grupos
publicando alternativas con radar de ondas milimétricas solo para no poner una
cámara—. Un sistema que graba a los operarios no lo aprueba ningún sindicato ni
ningún comité de convivencia.

Aquí el fotograma se destruye en cuanto se extraen las articulaciones, y lo único
que se puede persistir son esqueletos. **No es una promesa del README: el código
no tiene forma de guardar una imagen.** Eso es lo que hace el sistema instalable,
y es un argumento de venta antes que una decisión técnica.

---

## EL ENTREGABLE ES UN VÍDEO CON INTERFAZ, Y ESO CONDICIONA TODO

Decidido por Juan Diego el 29/09: el proyecto termina en **un vídeo de todo el
funcionamiento** que sube a LinkedIn, con **una interfaz que muestre lo que el
sistema hace**, y tiene que leerse como algo escalable y de alto impacto.

Eso no es un requisito de la última fase, es una restricción de diseño desde la
primera, y tiene tres consecuencias concretas:

  · **Lo que el vídeo tenga que mostrar hay que guardarlo mientras se procesa.**
    Por eso el `.npz` lleva los keypoints crudos y todos los componentes de REBA
    se devuelven por separado en vez de solo el puntaje final: un informe que
    solo dice «riesgo alto» no se puede grabar de forma interesante, y uno que
    dice «riesgo alto porque el tronco va a 62° recogiendo de una superficie
    baja» sí.
  · **La interfaz tiene que hacer visible el MECANISMO, no solo el resultado.**
    Lo que impresiona en treinta segundos es ver al sistema decidir: el esqueleto
    encima de la persona, los ángulos en vivo, el componente que dispara el
    puntaje, y el contador de exposición subiendo por puesto de trabajo.
  · **La pieza que hace la demo es el replay en esqueleto.** Se reproduce el
    incidente sin vídeo: se ve exactamente qué pasó y no hay imagen de nadie. Es
    la garantía de privacidad demostrada en pantalla en vez de explicada en el
    README, y es lo que un jefe de planta necesita ver para creérsela.

## POR QUÉ ESTO NO ES EL TUTORIAL DE YOUTUBE

"Pose estimation + calcular un ángulo" es una tarde de trabajo. Lo que separa
esto de un repo más:

1. **La vara la pone una norma, no yo.** REBA y RULA son los estándares del
   sector y REBA es el indicado para tareas industriales dinámicas. No se invento
   una métrica: se automatiza una que ya se factura. Eso también significa que
   **REBA geométrico es la línea base, no el producto**: el modelo tiene que
   aportar algo por encima de la norma, y si no lo aporta, se dice.
2. **Partición por sujeto, no por clip.** Casi todos los repos parten los clips
   al azar, así que la misma persona está en entrenamiento y en prueba: el modelo
   memoriza cuerpos y el número sale inflado. Aquí ninguna persona cruza la
   frontera, y se publica el número más bajo explicando por qué el otro miente.
3. **Falsas alarmas por hora de vídeo continuo.** Nadie la publica porque hunde
   los números, y es la única que dice si el sistema se puede instalar: un
   detector que grita cada diez minutos lo apagan el primer día.
4. **Generalización a otra sala y otra cámara.** Entrenar con el dataset de
   laboratorio y evaluar sobre grabaciones propias, con otra cámara y otras
   personas. Es la pregunta que haría un cliente, y si el número se cae —lo más
   probable— eso no es un fracaso: es el hallazgo del proyecto.
5. **Anticipación en vez de informe.** Un sistema que puntúa después es un
   reporte; uno que avisa **antes** de que el levantamiento malo se complete es
   prevención. Esa es la ambición del modelo entrenado, y es exactamente la línea
   del paper que acompaña al dataset.

---

## LAS FASES

### Fase 0 — Entorno y datos · EN CURSO

| quién | qué | estado |
|---|---|---|
| Claude | estructura, venv, torch 2.11 + CUDA 12.8, ultralytics 8.3.253 | hecho y verificado en la 3050 |
| Claude | prueba de humo de YOLO-pose: 4 personas, keypoints `(4,17,3)` | hecho |
| Claude | formato de esqueletos `src/pose/schema.py` y sus 6 pruebas | hecho, y la mutación tumba la prueba correcta |
| Claude | extractor `src/pose/extractor.py` | hecho, verificado sobre vídeo real |
| **Juan Diego** | **descargar UW-IOM** (`data/README.md`) | **pendiente** |
| Claude | adaptador de UW-IOM | bloqueado por la descarga |

**Comprobaciones (pasan hoy):**
```powershell
.\.venv\Scripts\python.exe -m tests.test_schema      # 6/6, sin GPU
.\.venv\Scripts\python.exe -m src.pose.extractor VIDEO --out salida.npz --max-frames 120
```

Lo medido el 29/09 sobre un vídeo cualquiera de 30 fps, 120 fotogramas, en la
3050: **77 ms por fotograma** con seguimiento incluido, frente a los 38 ms del
modelo solo. El seguidor y la lectura del vídeo duplican el coste, lo que apunta
—otra vez— a que el tiempo no está en la red.

**El formato de keypoints es la decisión de diseño de la fase**, porque todo lo
demás se construye encima: una secuencia es un array `(T, 17, 3)` —fotogramas ×
articulaciones × (x, y, confianza)— normalizado por la caja de la persona, para
que no dependa de la distancia a la cámara ni de la resolución. Guardado así, el
dataset entero de keypoints pesa megabytes en vez de gigabytes, el entrenamiento
cabe de sobra en la 3050, y la garantía de privacidad es estructural.

### Fase 1 — REBA geométrico: el número a batir, sin aprendizaje

| quién | qué |
|---|---|
| Claude | adaptador de UW-IOM y extracción de keypoints |
| Claude | REBA por fotograma calculado de los ángulos del esqueleto: tronco, cuello, piernas, brazos |
| Claude | arnés de evaluación: partición por sujeto, precisión/recall por nivel de riesgo, falsas alarmas por hora |
| **Juan Diego** | **decisiones de criterio** (abajo) |

Las decisiones que te tocan aquí, y hay que tomarlas **antes** de ver cualquier
resultado para no ajustar la vara:

- ¿A partir de qué puntaje REBA se considera exposición de riesgo que hay que
  contar? La norma da niveles; elegir el corte es una decisión de producto.
- ¿Cuánto tiene que durar una postura para contar como exposición, y no como un
  gesto de paso?
- ¿Qué cuesta más al cliente: perder una exposición real o levantar una falsa
  alarma? Eso fija hacia dónde se inclina el sistema.

### Fase 2 — El conjunto completo y la partición

| quién | qué |
|---|---|
| Claude | extracción masiva, partición por sujeto con reservado bajo llave |
| **Juan Diego** | decidir qué sujetos son reservado, y no mirarlos |

El reservado se decide aquí, con el conjunto sin tocar, que es la única ventana
para decidirlo sin trampa. Un reservado usado no vuelve a ser un reservado.

### EL NÚMERO A BATIR (2026-09-29)

Medido sobre los 20 sujetos, 4 pliegues de validación cruzada por sujeto, 10 Hz,
reservado sin tocar. La métrica es **F1 macro**, no *accuracy*: las clases están
brutalmente desbalanceadas —`stand/place` tiene 12.010 fotogramas y `walk/hold`
388— y un clasificador que conteste siempre lo mismo saca un *accuracy* decente sin
haber aprendido nada.

| campo | degenerado | reglas | **a batir** |
|---|---|---|---|
| **movimiento** (andar/de pie/agachado) | 0,287 | **0,673** | **0,673** |
| **altura de trabajo** | 0,172 | **0,641** | **0,641** |
| manipulación (recoger/colocar/sostener/alcanzar) | **0,113** | 0,048 | **0,113** |
| objeto (caja/varilla) | **0,193** | 0,127 | **0,193** |
| etiqueta completa, los 4 campos | **0,015** | 0,012 | **0,015** |

**La línea base a batir es el máximo de los dos brazos**, no la de las reglas: en
dos campos las reglas pierden contra el degenerado, y presentarlas como rival ahí
sería ponerle el listón bajo al modelo.

**Y donde el modelo tiene que ganar está identificado.** Las reglas hacen bien lo
que se ve en un fotograma —si el tronco está doblado, a qué altura están las
manos— y se estrellan en lo que necesita ver el tiempo:

  · **`manipulation` es el hueco grande.** Recoger y colocar son **el mismo gesto
    en dos sentidos**: un esqueleto suelto no sabe hacia dónde va el movimiento, y
    por eso las reglas ni lo intentan. Una ventana temporal sí puede. Si el modelo
    gana en algún sitio, tiene que ser aquí, y eso es exactamente la tesis del
    proyecto.
  · **`object` probablemente no es ganable**, y conviene decirlo por adelantado:
    si lo que se levanta es una caja o una varilla no está en el esqueleto. Si el
    modelo acierta ahí, la primera hipótesis no es que haya aprendido a ver el
    objeto, sino que **ha memorizado el orden del guion del experimento** —todos
    los participantes hacen las tareas en la misma secuencia—, y habrá que
    comprobarlo antes de celebrarlo.

**Los umbrales se calibran en entrenamiento**, no se ponen a ojo, para que la
línea base no salga artificialmente débil. Medido: los valores calibrados (30°,
25°, 30°, 30° de tronco según el pliegue) coinciden con los que se habían puesto
a ojo, y el F1 macro pasa de 0,674 a 0,673. **Calibrar no cambió nada, y ahora se
sabe en vez de suponerse.** La desviación del umbral entre pliegues es de 2,2°, o
sea que las reglas no dependen de a quién le tocó entrenar.

### EL TCN, MEDIDO (2026-09-30)

4 pliegues de validación cruzada por sujeto, 10 Hz, ventana causal de 2 s, 30
épocas, **78.095 parámetros**, 16–19 segundos de entrenamiento por pliegue en la
RTX 3050. Reservado sin tocar. Artefacto `artifacts/tcn-20260930.json`.

| campo | degenerado | reglas | **TCN** | rango TCN |
|---|---|---|---|---|
| movimiento | 0,288 | 0,670 | **0,870** | 0,829–0,882 |
| altura de trabajo | 0,174 | 0,639 | **0,798** | 0,765–0,823 |
| **manipulación** | 0,114 | 0,046 | **0,741** | 0,650–0,788 |
| objeto | 0,195 | 0,123 | **0,784** | 0,738–0,812 |
| etiqueta completa | 0,013 | 0,011 | **0,621** | 0,560–0,663 |

**El modelo gana donde estaba previsto que ganara, y esa es la tesis del proyecto
convertida en número.** En `manipulación` —distinguir recoger de colocar, que son
el mismo gesto en dos sentidos y un fotograma no puede separar— pasa de 0,114 a
**0,741**. Las reglas ahí sacan 0,046 porque ni lo intentan, y está dicho en su
código desde antes de medir.

Los rangos son estrechos, así que el resultado no depende de qué cuatro
participantes cayeron en la prueba.

**Lo que se hizo para que la comparación valga**, y sin lo cual estos números no
significarían nada:

  · **Los tres brazos predicen exactamente los mismos fotogramas.** El modelo
    necesita 2 s de pasado, así que no puede predecir los primeros 20 fotogramas
    de cada sujeto; si las reglas los hubieran predicho y el modelo no, la
    diferencia incluiría esa diferencia de material — y esos primeros fotogramas
    son los fáciles, la persona entrando en escena y colocándose. El arnés
    rechaza con error a un brazo que devuelva de más o de menos.
  · **La red es causal**: cada convolución solo mira hacia atrás. Una que mirara
    el futuro daría mejores números y no se podría instalar.
  · **Las reglas se calibran en entrenamiento**, para que la línea base no salga
    débil por descuido y le regale ventaja al modelo.

#### La alarma del campo `object`, comprobada y descartada

Esta guía tenía escrito **antes de medir** que si el modelo acertaba el objeto
—caja o varilla, que no está en un esqueleto— la primera hipótesis no debía ser
que hubiera aprendido a verlo, sino que hubiera memorizado el orden del guion del
experimento. Sacó 0,784, así que tocaba comprobarlo.

Medido en los datos crudos, sin ningún modelo de por medio
(`scripts/probe_object_leak.py`), la separación entre las muñecas normalizada por
la altura del cuerpo:

| objeto | mediana | n |
|---|---|---|
| caja | **0,105** (manos juntas) | 11.352 |
| varilla | **0,237** (manos separadas) | 9.798 |

**La señal está en la postura: el modelo lee cómo se agarra, no qué se agarra.**
Una varilla larga se coge con las manos separadas a lo largo de ella y una caja
con las manos a los lados. Para un sistema de ergonomía eso es exactamente lo que
interesa. No es una prueba cerrada —el solapamiento intercuartílico es del 44% y
es una sola característica— pero basta para no tratar el resultado como
sospechoso. De paso corrigió una predicción mía, que era la contraria.

### EL INFORME POR TAREAS, Y LA PREGUNTA QUE DECIDE SI EL MODELO SIRVE (2026-09-30)

Las dos mitades juntas: REBA dice cuánto riesgo hay, el modelo dice de qué tarea
viene. Por separado no se puede accionar ninguna de las dos.

**Y de ahí sale la medida que importa para el cliente, que no es el F1:** el
informe se genera dos veces sobre los mismos sujetos, una con las etiquetas
verdaderas y otra con lo que predice el modelo. El riesgo total es idéntico —REBA
se calcula de la geometría— así que lo único que puede cambiar es la atribución.

| | resultado (pliegue 1, puesto configurado con 12 kg y agarre regular) |
|---|---|
| la tarea número 1 coincide | **sí** |
| de las 3 peores, coinciden | **3 de 3** |
| riesgo atribuido a la peor tarea | 26% real contra **23%** del modelo |

**Un modelo con 0,741 de F1 lleva a la misma decisión que la verdad.** Para este
producto eso es suficiente: el jefe de planta interviene en el mismo sitio. Y es
un criterio de aceptación mucho más honesto que un umbral de F1 elegido a ojo.

#### El hallazgo que justifica el informe entero

| tarea | REBA | tiempo | % del riesgo |
|---|---|---|---|
| `bend / place / low` (agacharse al suelo) | **9**, el más alto | 0,6 min | 9% |
| `stand / place / mid` (colocar a media altura) | 4 | 3,1 min | **26%** |

**La tarea más peligrosa no es la que más daño acumula.** Quien mirara solo el
pico de REBA mandaría a rediseñar el agacharse, y el 26% del problema está en una
tarea de riesgo moderado que dura cinco veces más. Eso solo aparece cruzando
severidad con duración y atribuyéndolo a tareas, y tiene su prueba: si el informe
ordenara por severidad en vez de por riesgo acumulado, caería
`test_long_moderate_task_outranks_short_severe_one`.

### Fase 3 — Los modelos, que entrenas tú

| quién | qué |
|---|---|
| Claude | data loaders, bucle de entrenamiento, registro de experimentos, evaluación idéntica a la de la línea base |
| **Juan Diego** | **entrenar**: TCN/GRU sobre keypoints, y ST-GCN sobre el esqueleto como grafo |

El TCN va primero porque entrena en minutos. ST-GCN trata el cuerpo como grafo
espacio-temporal y es el estándar del reconocimiento de acción sobre esqueletos:
es el que cambia la conversación en una entrevista de visión.

### Fase 4 — Las dos preguntas que hacen el proyecto

1. **¿Aporta el modelo algo sobre la norma?** REBA geométrico ya da un puntaje.
   El modelo tiene que ganar en algo concreto: anticipar la postura de riesgo
   antes de que se complete, o aguantar oclusiones que tumban el cálculo
   geométrico.
2. **¿Sobrevive a otra sala?** Entrenado con el laboratorio, evaluado sobre
   grabaciones propias con otra cámara y otras personas.

### Fase 5 — Latencia, con una advertencia ya medida

**Antes de optimizar nada hay que mirar de qué está hecho el tiempo.** En la
prueba del 29/09, `yolo11s-pose` salió **más rápido** que `yolo11n-pose` (p50 de
33,4 contra 38,8 ms en la 3050, n=30, imagen en memoria): el modelo grande
ganando al pequeño significa que el tiempo se lo come el preprocesado y el
envoltorio, no la red. Si eso se confirma, cuantizar el modelo no arreglaría
nada. Se mide primero por etapas.

### Fase 6 — La aplicación

Pipeline en vivo, la garantía de privacidad en el código, y el panel con lo que
compra un jefe de SST: horas-hombre de exposición por puesto, ranking de puestos
de trabajo, y la tendencia después de una intervención. La pieza que hace la
demo es el **replay del evento en esqueleto**: se ve exactamente qué pasó y no
hay imagen de nadie.

### Fase 7 — Documentación de portafolio

README con los números y su procedencia, los ADRs de las decisiones, y el vídeo.

---

## EL MÉTODO, heredado y no negociable

- **Todo número va con su tamaño de muestra, su procedencia y la fecha.**
- **Partición por sujeto.** Si una persona cruza la frontera, el número no vale.
- **La métrica no es *accuracy*.** Precisión y recall por nivel de riesgo, y
  falsas alarmas por hora de vídeo continuo.
- **Mira de qué está hecho el tiempo antes de optimizarlo.**
- **Rompe tus propios tests a propósito.**
- **Coherente no es correcto.**
- **No ajustes la vara al resultado.** Las decisiones de criterio, antes de ver
  los números.
- **Verifica contra el stack levantado**, no solo que compile.

---

## DECISIONES PENDIENTES DE JUAN DIEGO

- Los tres criterios de la fase 1 (corte de riesgo, duración mínima, hacia dónde
  se inclina el sistema).
- El nombre del proyecto y del repositorio. La carpeta se llama `pose_stimation`,
  que describe la técnica y no el problema, y lleva una errata (`stimation` por
  `estimation`). Para un repositorio que va a LinkedIn conviene un nombre que
  nombre el problema.
