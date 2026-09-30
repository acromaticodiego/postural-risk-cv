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
