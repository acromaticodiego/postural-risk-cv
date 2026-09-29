# GUÍA DEL PROYECTO — qué hay que hacer, en orden

Este fichero es la fuente de verdad del proyecto. Se actualiza cada sesión.
Si abres el proyecto después de días y no recuerdas nada, lee solo esto.

**El reparto:** Juan Diego entrena los modelos y toma las decisiones de criterio.
Claude construye el arnés de datos, la evaluación, el pipeline de inferencia y la
aplicación, y mantiene esta guía diciendo explícitamente qué toca hacer.

---

## LO QUE TIENES QUE HACER AHORA

**Descargar el dataset UW-IOM.** Instrucciones exactas en `data/README.md`.
Es lo único que bloquea todo lo demás, y no hay que grabar nada todavía.

Mientras descarga, Claude escribe el extractor de keypoints y la línea base REBA.

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
| **Juan Diego** | **descargar UW-IOM** (`data/README.md`) | **pendiente** |
| Claude | extractor de keypoints y formato `.npz` | pendiente |

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
