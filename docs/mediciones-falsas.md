# Veces que algo salió limpio y era falso

La lista más útil del proyecto. **Coherente no es correcto.**

Se escribe aquí cada vez que una medición, una prueba o una cifra parecía buena y
no lo era, con lo que la destapó. En el proyecto del agente de voz esta lista
llegó a quince entradas y acabó siendo el mejor material de entrevista que tenía.

---

## 1. La prueba de la normalización pasaba con la normalización rota (2026-09-29)

**Qué se creía.** Que `test_isotropic_normalization_preserves_angles` protegía la
decisión de diseño más importante del formato de datos: que normalizar los
esqueletos no puede deformar los ángulos del cuerpo, porque el riesgo ergonómico
**son** ángulos. La prueba construía un cuerpo con un ángulo conocido de 45
grados, lo normalizaba, y comprobaba que seguían siendo 45.

**Qué pasaba.** Se sustituyó `normalize_isotropic` por la versión equivocada a
propósito —dividir la x por el ancho de la caja y la y por el alto, que es lo que
se hace por costumbre— y **las seis pruebas siguieron en verde**.

**Por qué.** El cuerpo de prueba tenía tres articulaciones colocadas a mano, en
(1,1), (0,0) y (1,0), y las demás en el centro de la cadera. De ahí sale una caja
de 1 por 1: **cuadrada**. Con una caja cuadrada, dividir por el ancho y dividir
por el alto son la misma operación, así que la normalización mala coincidía
exactamente con la buena y no había nada que detectar.

Y había un agravante escrito por la misma mano: la prueba llevaba una aserción
que intentaba comprobar justo eso, anulada con un `or True` al final, o sea que no
comprobaba nada y además ocupaba el sitio de la que sí hacía falta.

**El arreglo.** El cuerpo de prueba lleva ahora un tobillo que alarga la caja a 1
por 3, y la condición previa —que la caja **no** sea cuadrada— se comprueba en una
función aparte, `_demand_non_square`, con un mensaje que explica por qué sin ella
la prueba no prueba nada. Con eso, la mutación tumba exactamente una prueba y solo
esa: el ángulo pasa de 45,00 a 18,43 grados.

**La lección.** Una prueba que sigue pasando cuando rompes lo que dice proteger no
está protegiendo nada, y el fallo casi nunca está en la aserción: está en que **el
material de prueba no ejercita la condición en la que el error aparece**. Aquí el
caso elegido era degenerado, igual que en el proyecto del agente de voz el
material de silencio resultó no ser silencio.

Lo que la hace valiosa es que se escribió **sabiendo** que esto pasa —la regla de
romper los propios tests venía heredada y el módulo llevaba el mecanismo
explicado en su cabecera— y pasó igualmente, a la primera prueba escrita del
proyecto. Leer la regla no basta: hay que ejecutar la mutación.

---

## 2. Un número plausible calculado sobre el 35% del vídeo (2026-09-29)

**Qué se creía.** Que el extractor sacaba el esqueleto del participante de un
vídeo de UW-IOM. La primera medida del sujeto 01 dio una separación bend/stand de
+29,61 grados: un número razonable, del orden del resto de los sujetos, sin
ninguna señal de alarma.

**Qué pasaba.** El extractor usaba el seguidor de ultralytics, y en un vídeo con
UN SOLO participante el seguidor creó **14 identidades distintas**. La mayor
cubría 515 de 1474 fotogramas, y había otra de 500 que era la misma persona
partida por la mitad. El criterio de quedarse con la identidad de más fotogramas
—razonable por sí solo— tiraba el **65% del vídeo**, y el número salía igual de
plausible con un tercio de los datos.

**Cómo se destapó.** No revisando el extractor, sino porque el diagnóstico del
sujeto 3 llevaba un sujeto sano como CONTROL. El control existía para comprobar
que el cálculo con YOLO era comparable al del Kinect, y de paso dejó a la vista
que en el sujeto sano solo había 515 fotogramas de 1474. Sin el control, el
extractor habría seguido perdiendo dos tercios del dataset en silencio.

**Lo que lo hace peligroso.** Lo que se pierde no es aleatorio: el detector
titubea justamente en las posturas raras, que son las que este proyecto quiere
medir. Un dataset al que se le caen los fotogramas difíciles no da un número con
más varianza, da un número optimista.

**El arreglo.** `extract_single_subject`, que no sigue a nadie: detecta por
fotograma y se queda con la caja de mayor área. Sobre los mismos vídeos pasa de
515 a **1474 de 1474** fotogramas en el sujeto 1 y a 1144 de 1147 en el 3, y
además es más rápido (11 ms por fotograma contra 77). El seguimiento vuelve a
hacer falta cuando haya varios operarios, y entonces a quién se mide será una
decisión de producto, no un detalle de implementación.

**La lección.** Una herramienta de más no es gratis. El seguidor estaba ahí por
previsión —«en una planta habrá varias personas»— y en el material de hoy solo
hacía daño. Y el segundo filo: el control de un experimento sirve para más de lo
que se escribió; este se puso para validar una comparación y lo que cazó fue un
fallo del instrumento.

---

## 3. La mitad del turno salía «en riesgo», y era un ángulo invertido (2026-09-29)

**Qué se creía.** Que el REBA geométrico funcionaba: 12 pruebas en verde, las tres
tablas de la norma comprobadas celda a celda y monótonas en todos sus ejes, los
umbrales verificados en sus fronteras. El primer número real sobre el sujeto 1 fue
**86,6 segundos de riesgo sobre 173 medidos —el 50% del turno— y 29 eventos**.
Alto, pero la tarea consiste en manipular cajas a distintas alturas, así que
colaba.

**Qué pasaba.** La flexión de rodilla y la de codo estaban calculadas como
`180 - ángulo(segmento1, segmento2)`. Con los vectores que usa el módulo —cadera→
rodilla y rodilla→tobillo, los dos en el mismo sentido de recorrido— una pierna
**recta** da 0 grados entre ellos, no 180. Así que la fórmula devolvía 180 grados
de flexión para una pierna estirada, y las piernas puntuaban 3 sobre 4 **todo el
rato**. El `180 -` es lo que uno escribe pensando en el ángulo interior de la
articulación, que es una imagen mental correcta y una implementación equivocada.

Corregido: **28,5 segundos de riesgo (16%) y 13 eventos**, y la causa dominante
pasó de `legs` a `trunk`, que es lo que tiene sentido en una tarea de recoger cajas
del suelo.

**Por qué las pruebas no lo cazaron.** `test_neutral_posture_is_low_risk` ponía una
persona de pie perfectamente recta y comprobaba que **el REBA agregado** saliera
«despreciable o bajo». Con el bug, esa postura daba piernas 3 y antebrazo 2… y un
REBA total de **2**, que es «bajo». La prueba pasaba. **El agregado tolera un
componente roto: por eso hay que comprobar el desglose.** Ahora se exige tronco 1,
cuello 1, piernas 1 y brazo 1 uno por uno, más dos pruebas que doblan el codo y la
rodilla a 90 grados para comprobar el ángulo en el otro sentido — con el ángulo
invertido, esas dos y la neutra no pueden pasar a la vez.

**Cómo se destapó, y es lo que hay que recordar.** No lo encontró ninguna prueba:
lo encontró **dibujar el esqueleto**. En el primer fotograma renderizado el panel
decía «Rodilla 161 grados» junto a un monigote con las piernas rectas, y eso es
imposible de no ver. Doce pruebas deterministas miraron el número y no lo vieron;
una imagen lo delató en dos segundos.

La visualización se había planificado como el entregable de presentación —el vídeo
para enseñar el proyecto— y resultó ser **un instrumento de depuración**. Un
número equivocado es plausible mientras sea un número; dibujado sobre un cuerpo,
deja de serlo. Conviene construir la vista antes de fiarse de las cifras, no
después.

---

## 4. «REBA no distingue las dos técnicas de levantar»: generalicé de un caso (2026-09-30)

**Qué se creía.** Que REBA es incapaz de separar levantar con la espalda doblada
de levantar con las rodillas dobladas, y que por eso hacía falta añadir la ecuación
NIOSH. La evidencia parecía sólida: calculadas las dos posturas, REBA daba **7
contra 6**, un solo punto de diferencia, y había una explicación mecánica
convincente —al agacharse bien, la norma *sube* el puntaje de piernas casi tanto
como baja el del tronco—.

**Qué pasaba.** El par de posturas que comparé tenía las manos **en el mismo
sitio**, así que lo único que cambiaba era el reparto de la flexión entre tronco y
rodillas. Pero en un levantamiento real las dos cosas van juntas: doblar las
rodillas te permite **acercarte a la carga**, y doblar la espalda te obliga a
alejarla. Medido sobre ese caso, el de verdad, REBA sube de **3 a 6** al alejar las
manos, porque también se inclina el tronco y se eleva el brazo. REBA sí lo ve.

**Cómo se destapó.** Por una prueba escrita para justificar el módulo nuevo:
`test_reba_barely_moves_between_the_same_two_lifts`, que afirmaba que REBA no
separa las dos técnicas. Falló a la primera. **La prueba estaba bien y la
afirmación estaba mal**, que es el orden correcto en que conviene que pasen las
cosas.

**Lo que sobrevive, y es lo que justifica NIOSH de verdad:** un resultado en
kilogramos —cuánto debería pesar la carga para que ese levantamiento fuera
aceptable—, un umbral que incorpora el peso real en vez de un ajuste de 0 a 3, y el
modelado de la frecuencia y el recorrido. Todo eso es cierto y comprobado; lo que
no era cierto es el argumento con el que llegué.

**La lección.** El ejemplo con el que uno se convence casi nunca es el caso
general, y cuanto mejor sea la explicación mecánica que lo acompaña, menos ganas
dan de comprobarlo. Aquí la explicación —«la norma sube las piernas»— era
verdadera, y la conclusión que sostenía era falsa igualmente: era verdad **en ese
par de posturas**, y ese par no ocurre en una planta.

---

## Sustos que se comprobaron y NO eran falsos

No todo lo sospechoso está mal, y anotar las falsas alarmas evita desconfiar de
los números buenos.

- **Dos sujetos dieron exactamente +30,11° de separación** (2026-09-29), con
  distinto número de fotogramas y de etiquetas. Dos medidas independientes que
  coinciden a dos decimales huelen a variable reutilizada, así que se comprobó con
  más precisión: las medias son 36,22 / 6,12 en un sujeto y 39,16 / 9,05 en el
  otro, y las diferencias 30,108353 y 30,114947. Coincidían solo al redondear.
