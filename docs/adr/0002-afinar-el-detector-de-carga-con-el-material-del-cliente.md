# ADR 0002 — Afinar el detector de carga con el material del cliente, y con quién se valida

Fecha: 2026-10-02
Estado: aceptado, pendiente de veto de Juan Diego

## Contexto

El detector de carga está entrenado con `package-seg` —2.197 imágenes de cajas de
cartón de almacén, mAP50 de 0,935 en máscaras— y **no se sabe si sirve en las manos
de una persona**. La comprobación que tenía que responderlo dio 0 de 89 fotogramas y
no concluye nada, porque lo que el vídeo mostraba era un organizador de plástico
transparente y el dataset son cajas de cartón opacas: el 0% tiene dos causas que ese
material no separa. Está escrito como la medición falsa nº 5.

Juan Diego propuso afinar el modelo con sus propios vídeos, etiquetándolos con SAM
guiado por el esqueleto en vez de a mano, y la prueba de concepto funcionó: con un
punto positivo entre las manos y puntos negativos en el cuerpo, la máscara cae sobre
la caja y no sobre la persona (`artifacts/samneg-1.png`). El flujo acordado fue
extraer fotogramas, pre-etiquetar con SAM, validar a mano y afinar.

Estas decisiones las tomé yo aplicando su criterio —«que sea escalable y cercano a un
comportamiento de la vida real, como si fuera un producto real»— y quedan escritas
para que pueda vetar cualquiera. No se le atribuyen a él.

## Decisión

### 1. Se AFINA el modelo de fábrica mezclando su material, no se sustituye

Los vídeos de un cliente son una habitación, una luz y dos o tres objetos. Entrenar
solo con eso aprende *esas* cajas y se cae con la siguiente. Afinar parte de
`package-seg` y le añade las cargas del cliente.

Y lo que empezó siendo una limitación del material resulta ser la arquitectura
correcta: **el modelo viene de fábrica y se calibra en cada planta con las cargas de
ese cliente**, que es el ciclo de vida de un producto instalable y no el de un
experimento. La calibración por puesto del ADR 0001 ya iba en esa dirección; esto es
la misma idea aplicada a los pesos.

Se mezclan **3 imágenes de fábrica por cada imagen nueva**. Con las 1.920 enteras las
del cliente serían el 5% del lote y el afinado apenas se notaría; sin ninguna, el
modelo olvida las cajas de almacén. Es un parámetro, no una constante escondida, y el
artefacto guarda cuál se usó.

**Descartado:** entrenar un detector desde cero con el material del cliente. Daría un
número excelente sobre su propia habitación y ninguna garantía fuera de ella, que es
justo lo que un cliente no puede comprobar antes de comprar.

### 2. La partición es POR VÍDEO, y el programa se niega si solo hay uno

Dos fotogramas del mismo vídeo separados por medio segundo son casi la misma imagen.
Repartidos entre entrenamiento y validación, el modelo aprueba por haberlos
memorizado — es exactamente la partición por sujeto del modelo de tareas, un piso más
abajo, y por el mismo motivo.

Con un solo vídeo no hay partición honesta posible, así que `finetune_load.py` **se
niega a entrenar** en vez de partir por fotograma y dar un número inflado.

### 3. Se mide antes y después, y también sobre el conjunto de fábrica

Un mAP a secas no dice si afinar sirvió. Y afinar con cien imágenes de una habitación
puede subir el número del cliente y destrozar el de fábrica —olvido catastrófico—,
cosa que **no se vería mirando solo el conjunto del cliente**, donde el número
seguiría siendo excelente. Por eso se evalúan los dos frentes en las dos fases.

### 4. La procedencia del vídeo se DECLARA, y el entrenamiento la hace cumplir

Los cuatro vídeos que existían son **capturas de pantalla del panel**, no grabaciones
de la cámara. Eso los inhabilita como material de entrenamiento por una razón que se
ve al mirarlos: el esqueleto va **pintado encima** de la persona y de la carga, así
que el modelo aprendería a buscar líneas de colores, y la validación —que lleva las
mismas líneas— no lo delataría. Lo medido sobre ese material, en el recuadro de la
cámara recortado a su resolución nativa (687×441):

| | |
|---|---|
| confianza de YOLO-pose sobre la persona | **0,13** (`postura prueva 3`, fotograma al 55%) |
| fotogramas examinados sin encontrar a nadie | **148 de 317** en ese vídeo |

Así que `extract_load_frames.py` exige `--fuente {camara,pantalla}` sin valor por
defecto, lo escribe en el manifiesto de cada fotograma, y `finetune_load.py`
**se niega** a entrenar con material declarado `pantalla` salvo que se le pase
`--incluir-pantalla`, que marca el artefacto como ensayo.

**Descartado:** detectar la captura de pantalla mirando los píxeles, contando trazos
del color del panel. Una planta verde en la escena daría el mismo resultado, y de qué
cámara salió un vídeo es un dato que quien lo grabó conoce. Es el criterio del ADR
0001 otra vez: **lo que se sabe se declara, no se estima.**

### 5. SAM propone y una persona valida; los filtros no son el revisor

Los puntos negativos sobre el cuerpo **no son una garantía, son una sugerencia**, y
hay con qué decirlo: sobre 630 fotogramas, SAM devolvió la silueta de la persona en
335 pese a tener hasta siete puntos negativos encima de ella. Casi siempre era lo
correcto —en esos fotogramas no había ninguna carga que segmentar— pero el dato que
importa es que **SAM no obedece los negativos cuando no hay alternativa**.

Lo que garantiza es el filtro de después (`judge_mask`): tamaño, una sola pieza, que
no tape la cara, que no sea tan alta como la persona, y que esté en las manos con el
mismo radio que usa el sistema en producción. Y por encima del filtro, una persona
mira la hoja de contactos y descarta. Un pre-etiquetado es una propuesta: las malas
de SAM son plausibles —segmenta algo, con un contorno limpio, y a veces ese algo es
el sofá—, y entrenar con lo que salga sin que nadie lo haya mirado es la forma más
directa de obtener un número limpio sobre material falso.

**Un filtro que se corrigió al escribirlo, porque rechazaba el caso bueno:** la
primera versión exigía que la máscara no tocara ninguna de las articulaciones que se
le pasan a SAM como negativas, caderas incluidas. Una caja sujetada contra el vientre
**sí** tapa las caderas en la imagen, y eso es lo normal al levantar. Lo cazó la
prueba `test_a_box_in_the_hands_is_accepted` a la primera ejecución.

### 6. La variedad se busca entre los fotogramas que TIENEN carga, no antes

Elegir primero los fotogramas más distintos en postura y segmentar después parece el
orden natural y es el equivocado: **11 de 12 elegidos por variedad no tenían nada en
las manos**, porque una postura rara es justamente lo que hace alguien cuando no está
cargando nada. Así que SAM se pasa por todos los candidatos y la selección de los más
variados se hace entre los que sobreviven a los filtros.

Y la variedad sigue siendo por **postura**, no por reparto en el tiempo: un fotograma
cada N segundos da un conjunto donde la mitad son la misma persona de pie.

## Consecuencias

- **El flujo queda montado y sin material con el que correrlo de verdad.** Hace falta
  que Juan Diego grabe con `scripts/record_loads.py`. Es lo único que lo bloquea.
- **Un guion del proyecto escribe imágenes al disco**, y conviene decir dónde queda la
  frontera: la garantía de privacidad es sobre el sistema que se instala
  —`src/pose/extractor.py` no tiene forma de guardar un fotograma, y por ahí pasa todo
  lo que el producto procesa—. El etiquetado es una herramienta que se corre a mano,
  sobre vídeos propios, una vez por cliente, y escribe bajo `data/`, que no entra al
  repositorio. La frase publicable sigue siendo cierta y conviene decirla completa:
  **el sistema no guarda imágenes; calibrarlo para una planta nueva sí necesita unos
  minutos de vídeo, y ese vídeo no sale de la máquina del cliente.**
- El rendimiento del pre-etiquetado sobre el material que hay es de **81 propuestas
  válidas de 630 fotogramas candidatos**. Es un ensayo sobre capturas de pantalla, no
  una tasa: dice que el flujo funciona, no cuánto rinde.
