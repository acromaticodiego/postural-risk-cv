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

**Lo que hay que verificar al descargarlo, y no dar por hecho:** varios trabajos
que usan este dataset afirman que trae anotaciones de riesgo **REBA a nivel de
fotograma**. La ficha oficial solo confirma las 17 etiquetas de acción. Si las
anotaciones REBA no vienen, el puntaje se calcula de los ángulos del esqueleto
—que es el plan de todas formas— y las etiquetas de acción sirven para la parte
que entrena el modelo.

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
