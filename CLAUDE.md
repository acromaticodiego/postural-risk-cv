# Proyecto: medición continua de riesgo ergonómico en planta

Directorio: `C:\Users\ASUS\Desktop\pose_stimation` (Windows, PowerShell).

**Lee `GUIA.md` antes que nada.** Es la fuente de verdad de las fases y dice qué
le toca a cada uno. Este fichero es el contexto; la guía es el plan.

## Quién soy

Juan Diego Ossa, Ingeniero Mecatrónico, ~2,5 años como Backend & AI Engineer.
Python (FastAPI), Node/NestJS, PostgreSQL, Redis, Docker, AWS, PyTorch, YOLOv8,
OpenCV, pgvector.

Tengo cinco proyectos de portafolio: tres de visión por computador (tráfico,
control de acceso facial, clasificación fitosanitaria de aguacates), un agente de
verificación documental con OCR, y un agente de voz telefónico. Este es el
cuarto de visión.

**El reparto en este proyecto: yo entreno los modelos, Claude construye el arnés
de datos, la evaluación y la aplicación.** Y Claude me va guiando: cada tanda
termina diciéndome qué me toca hacer, concreto y acotado.

## Qué tiene que demostrar, y qué no

**Lo que demuestra y los otros no:** entender el **tiempo** en el vídeo. Los tres
proyectos de visión anteriores detectan y siguen objetos fotograma a fotograma;
ninguno clasifica una *secuencia*. Aquí el objeto de estudio es una acción que
dura, y eso es otra familia de arquitecturas y otra forma de medir.

**Lo que tiene que demostrar sobre todo lo demás: que resuelve un problema que
alguien paga.** No es un proyecto de universidad. El rigor de medición va como
respaldo de una afirmación de producto, nunca como el asunto del proyecto: los
titulares son el problema, su coste, y la decisión que el sistema toma; los
nombres de los datasets y de las arquitecturas van debajo.

**Lo que NO debe demostrar:** microservicios (ya lo hice con 8), detección de
objetos fotograma a fotograma (ya lo hice tres veces), ni OCR.

## El problema, en una frase

Los desórdenes musculoesqueléticos son la primera causa de enfermedad laboral en
Colombia y hoy se miden observando a un operario media hora con un portapapeles.
Este sistema mide la exposición postural de todos, todo el turno, contra la norma
REBA — y **sin guardar una sola imagen**, porque la objeción de privacidad del
trabajador es lo que frena la adopción de estos sistemas.

Las cifras con su procedencia están en `GUIA.md`.

---

## ESTADO

**El estado real vive en `GUIA.md`, en la sección «DÓNDE ESTAMOS». Este resumen se
queda viejo; aquel se actualiza cada sesión** — de hecho este decía «Fase 0,
esperando el dataset» con el proyecto entero funcionando.

Al 2026-10-02: el sistema va de punta a punta —pose, REBA, NIOSH, el TCN de tareas,
el informe por tareas y el panel con modo en vivo—, **126 pruebas en verde**, y el
README existe. Lo único bloqueado es el afinado del detector de carga, que espera a
que Juan Diego grabe vídeo con la cámara en crudo (`scripts/record_loads.py`).

**Levantar el panel:**
```powershell
Set-Location C:\Users\ASUS\Desktop\pose_stimation
.\.venv\Scripts\python.exe -m uvicorn src.api.server:app --port 8010
# el 8000 lo ocupa el agente de voz
```

**Las pruebas:**
```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

---

## REGLAS DE TRABAJO (no negociables)

1. NUNCA `git push` sin preguntar. Una autorización no se extiende a la
   siguiente. Commitear en local sin preguntar sí está bien.
2. Rama de feature, nunca commits a `main`. Cada tanda nueva, su rama: yo abro
   el PR y lo fusiono.
3. NINGUNA atribución a Claude en los commits.
4. Commits y documentación EN ESPAÑOL. Ramas y código en inglés.
5. Estoy en PowerShell: nada de `&&`, `rm -rf`, `export VAR=x`, `2>/dev/null`.
6. VERIFICA CONTRA EL STACK LEVANTADO, no solo que compile.
7. Commitea ANTES de mutar código.

## MÉTODO

- **Todo número va con su tamaño de muestra, su procedencia y la fecha.**
- **Partición por sujeto.** Si una persona cruza la frontera entre entrenamiento
  y prueba, el número no vale.
- **La métrica no es *accuracy***, son precisión y recall por nivel de riesgo más
  falsas alarmas por hora de vídeo continuo.
- **Mira de qué está hecho el tiempo antes de optimizarlo.** Ya hay un aviso
  medido en la fase 5 de la guía.
- **Rompe tus propios tests a propósito.**
- **Coherente no es correcto.**
- **No ajustes la vara al resultado.** Las decisiones de criterio se toman antes
  de ver los números, y las toma Juan Diego.
- **Sé fiel al informar.** Si algo quedó sin comprobar, dilo. Si una
  recomendación resultó equivocada, dilo también.
- **Documenta el porqué, no el qué.**

## Quién decide qué

**Mías (Juan Diego):** lo que fija un método de medición, lo que compromete algo
que no se puede deshacer (qué sujetos son reservado), los criterios de producto
(a partir de qué puntaje hay riesgo), el alcance, y todo lo que salga de la
máquina. `git push` siempre se pregunta.

**De Claude:** la ejecución dentro de una tanda acordada —estructura de módulos,
formato de los datos intermedios, dónde vive una comprobación, cómo se escribe
una prueba—. Esas se deciden y se cuentan, sin menú.

## TRAMPAS DEL ENTORNO

- **Los heredocs de bash se comen los escapes.** `\n` se convierte en salto de
  línea real y `\b` en un byte 0x08. Para cualquier parche con escapes, usar la
  herramienta de escritura de ficheros, no un heredoc. En el proyecto del agente
  de voz esto rompió código cinco veces, una en silencio.
- **`predict` sobre una URL descarga la imagen en cada llamada.** Medir latencia
  así da un número inventado: la primera medida de este proyecto salió en 110 ms
  por eso, y con la imagen en memoria son 33–39 ms.
- **`faster-whisper` y compañía necesitan las DLL de CUDA antepuestas al PATH**
  del proceso en esta máquina. Aquí todavía no aplica, pero está avisado.
- La RTX 3050 tiene **6 GB**. Los modelos de pose caben de sobra; entrenar sobre
  keypoints también, porque los keypoints pesan nada.
