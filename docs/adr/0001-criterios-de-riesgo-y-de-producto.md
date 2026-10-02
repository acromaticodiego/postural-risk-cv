# ADR 0001 — Los criterios de riesgo, decididos como un producto y no como un experimento

Fecha: 2026-09-29
Estado: aceptado, pendiente de veto de Juan Diego

## Contexto

REBA da los niveles de riesgo pero no dice cuándo un sistema automático debe
contar una exposición, cuánto tiene que durar una postura para no ser ruido, ni
qué hacer con los componentes que una cámara no puede medir. Son cinco decisiones
de criterio y ninguna la resuelve la norma.

Se le presentaron a Juan Diego como menú y respondió con un criterio en vez de con
cinco respuestas: **«vamos por lo que sea más escalable y cercano a un
comportamiento de la vida real, esto tiene que quedar como si fuera un producto
real»**. Las decisiones de abajo las tomé yo aplicando ese criterio, y quedan aquí
escritas para que pueda vetar cualquiera. No se le atribuyen a él.

## Decisión

### 1. La exposición cuenta desde REBA ≥ 4, y se reporta por nivel

El nivel medio de la norma (4–7) es donde REBA dice «acción necesaria». Por
debajo, el nivel bajo (2–3) lo alcanza cualquiera de pie con un brazo algo
elevado: contarlo llenaría el informe de exposiciones que ningún ingeniero de SST
va a ir a corregir, y un informe que nadie acciona se deja de leer.

Y no se publica un contador único, sino el tiempo en cada nivel —medio, alto, muy
alto—, porque un jefe de planta prioriza por gravedad: diez minutos en riesgo muy
alto y dos horas en riesgo medio son problemas distintos con soluciones distintas,
y sumarlos en una sola cifra borra justo la información que decide dónde
intervenir.

**Descartado:** contar desde el nivel bajo (2–3). Da números más grandes y más
vistosos, y es exactamente por eso que no sirve: inflar la métrica de riesgo es la
forma más rápida de que el cliente desconfíe del sistema entero.

### 2. Un EVENTO necesita un segundo sostenido; la EXPOSICIÓN acumulada no tiene mínimo

Son dos cosas distintas y tratarlas como una era el error de la pregunta original:

- **Evento**, lo que dispara una alerta y aparece en la bandeja: postura de riesgo
  sostenida **al menos 1 segundo**. Un levantamiento real dura entre uno y tres
  segundos, así que el umbral no se los pierde; y por debajo de un segundo, a 10
  Hz, lo que hay son diez fotogramas que bien pueden ser un titubeo del detector.
- **Exposición acumulada**, lo que va al informe: **todo fotograma de riesgo suma**
  al total del turno, sin mínimo. Aplicar el filtro de un segundo aquí sesgaría el
  informe a la baja y de forma no aleatoria, porque lo que se caería son las
  posturas breves y repetidas, que son precisamente el mecanismo de lesión por
  esfuerzo repetitivo.

**Descartado:** el minuto del ajuste de actividad de REBA como umbral de evento.
La norma usa «más de un minuto» para clasificar una postura como estática, que es
otra pregunta; con ese umbral no se contaría ni un levantamiento.

#### Enmienda del 2026-10-02: el segundo tiene que ser SOSTENIDO, no CONSECUTIVO

Aprobado por Juan Diego tras verlo fallar en una prueba en vivo. Con la regla tal
como estaba escrita, un vídeo de 13 segundos de alguien levantando cajas cerró
**cero eventos** — y no por no ver el riesgo: 65 de 128 fotogramas pasaban de REBA
4. El puntaje parpadea alrededor del umbral:

```
6 4 4 4 3 4 4 3 4 4 3 6 3 4 3 3 4 3 3 3 ...
```

Un solo fotograma en 3 reiniciaba la racha entera, así que la más larga duraba
**0,9 s** contra el 1,0 s exigido. Fallaba por un fotograma, y el fallo no era de la
persona ni de la norma: era del instrumento. Un fotograma suelto por debajo es
temblor del detector de pose, no que alguien se haya erguido y vuelto a agachar en
una décima de segundo.

Así que una racha admite **hasta 2 fotogramas (0,2 s) por debajo del umbral** sin
cerrarse. Medido sobre esa misma serie: con 0 de tolerancia salen 0 eventos, con 1
sale 1, y con 2 salen 2. Se para en 2 porque a partir de ahí se empiezan a fundir
levantamientos distintos en uno, y entonces la duración publicada sería falsa.

El tiempo tolerado **sí cuenta** en la duración del evento: la postura de riesgo no
se interrumpió, lo que falló fue la medida, y descontarlo publicaría 1,0 s donde
hubo 1,2.

**Lo que esto NO cambia:** la exposición acumulada del informe sigue sin mínimo y
sin tolerancia, porque ahí se cuenta fotograma a fotograma y no hay rachas que
romper. La enmienda toca solo la bandeja de alertas.

**La salvaguarda, comprobada:** hay dos pruebas y la tolerancia se rompió en las dos
direcciones. Con 0 cae la del parpadeo; con 12 cae además la que exige que una pausa
de verdad —un segundo entero sin riesgo— siga cerrando el evento, que es lo que
impide que dos levantamientos seguidos se publiquen como uno.

### 3. El sistema se inclina a NO dar falsas alarmas, y publica su tasa

Un sistema de seguridad que avisa mal se desconecta el primer día, y una vez
desconectado su recall es cero. Así que el punto de operación de las **alertas**
se elige por precisión, y la métrica que se publica al lado es **falsas alarmas
por hora de vídeo continuo**, que es la que decide si el sistema es instalable.

**La tensión, dicha en vez de escondida:** en seguridad y salud en el trabajo,
subestimar el riesgo es lo peligroso para el trabajador, y esta decisión va en esa
dirección — igual que la del punto 4. Lo que la hace defendible es que el informe
**declara** que es una cota inferior, así que nadie puede leer un puesto sin
alertas como un puesto sin riesgo. Un sistema que subestima y lo dice es
utilizable; uno que exagera y no lo dice se apaga.

Nótese que el **informe agregado** no hereda esa inclinación: ahí la exposición se
acumula sin filtrar (punto 2), porque para medir se quiere calibración y no
prudencia.

### 4. Lo que la cámara no ve se CONFIGURA POR PUESTO, no se adivina

Es la decisión que más cambia el diseño, y sale directamente del criterio de
producto real.

De los componentes de REBA, la cámara no puede medir el peso de la carga, la
calidad del agarre, ni la torsión del tronco y del cuello. La reacción de
laboratorio es asumir valores neutros y publicar una cota inferior. La reacción de
producto es otra: **en una planta de verdad esos datos EXISTEN y el cliente los
conoce.** El ingeniero de SST sabe que en la línea 3 las cajas pesan 12 kg y que
el agarre es de asa firme. Pedirle a una red neuronal que adivine por visión lo
que el cliente puede declarar en un formulario no es más ambicioso, es peor
ingeniería: introduce un error estimado donde había un dato exacto.

Así que el sistema tiene **configuración por puesto de trabajo**: peso típico de
la carga, tipo de agarre, y si la tarea obliga a torsión. Con esa configuración el
REBA de ese puesto deja de ser una cota inferior y pasa a ser completo. Sin ella
—puesto no configurado— se usan los neutros y el informe lo marca como estimación
mínima.

Eso convierte el «puesto de trabajo» en una entidad de primera clase del sistema,
con su cámara, su configuración y su histórico, que es además lo que hace la
arquitectura escalable: añadir una línea de producción es añadir un puesto, no
tocar el código.

**Descartado:** estimar el peso de la carga por visión. Es un problema abierto,
daría un número sin intervalo de confianza, y sustituiría un dato que el cliente
tiene por una predicción que nadie puede auditar.

### 5. Partición por sujeto: 4 pliegues de validación cruzada más 4 reservados

Lo que un cliente compra es que funcione con **sus** operarios, que el modelo no
ha visto nunca. Eso es exactamente lo que mide una partición por sujeto, y la
validación cruzada evita que la conclusión dependa de qué cuatro personas cayeron
en el conjunto de prueba —con 20 sujetos, una partición fija hace que un solo
participante valga el 25% del resultado—.

El reservado (sujetos 11, 14, 19 y 20) no entra en ningún pliegue y se mide una
sola vez, con la barrera en el código y no en la memoria de nadie.

## Consecuencias

- El informe del sistema tiene dos caras que no se mezclan: **alertas**,
  optimizadas contra el falso positivo, y **exposición acumulada**, sin filtrar.
- Todo puntaje publicado va con la configuración del puesto que se usó, o con la
  marca de «estimación mínima» si no había.
- Hace falta un modelo de datos con puesto de trabajo, cámara, configuración y
  eventos. Es lo que la interfaz mostrará y lo que hace creíble la escalabilidad.
- El corte de REBA ≥ 4 y el segundo de sostenimiento son **parámetros con valor
  por defecto**, no constantes escondidas: un cliente con otro criterio los
  cambia, y cualquier número publicado dice con qué valores se obtuvo.
