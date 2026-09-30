# GUÍA DEL PROYECTO — qué hay que hacer, en orden

Este fichero es la fuente de verdad del proyecto. Se actualiza cada sesión.
Si abres el proyecto después de días y no recuerdas nada, lee solo esto.

**El reparto:** Juan Diego entrena los modelos y toma las decisiones de criterio.
Claude construye el arnés de datos, la evaluación, el pipeline de inferencia y la
aplicación, y mantiene esta guía diciendo explícitamente qué toca hacer.

---

## LO QUE TIENES QUE HACER AHORA

El dataset ya está descargado y verificado. Lo que bloquea ahora son **tres
decisiones de criterio que son tuyas**, y hay que tomarlas antes de ver cualquier
resultado para no ajustar la vara:

1. **¿A partir de qué puntaje REBA cuenta como exposición de riesgo?** La norma
   da niveles (1 despreciable, 2–3 bajo, 4–7 medio, 8–10 alto, 11+ muy alto).
   Elegir dónde salta el contador es una decisión de producto, no de la norma.
2. **¿Cuánto tiene que durar una postura para contar?** Sin un mínimo, cada gesto
   de paso al agacharse cuenta como exposición y el informe se llena de ruido.
3. **¿Qué le cuesta más al cliente: perder una exposición real o levantar una
   falsa alarma?** Eso decide hacia dónde se inclina el sistema, y es lo que
   convierte una curva de precisión y recall en un punto de operación.

Mientras las piensas, Claude extrae los esqueletos de los 20 sujetos y monta el
cálculo de REBA geométrico.

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
