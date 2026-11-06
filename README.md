# Señales UPC · Captura de dibujos

Sitio web estático para capturar dibujos a mano de **5 señales de tránsito** y armar el dataset de la PC03.

| ID | Clase | Qué dibujar |
|----|-------|-------------|
| 0 | `PARE` | Octágono con la palabra STOP |
| 1 | `FLECHA` | Rombo con una flecha hacia la derecha |
| 2 | `PROHIBIDO` | Círculo con una E y una línea diagonal |
| 3 | `PEATON` | Rombo con una persona caminando |
| 4 | `SEMAFORO` | Rectángulo vertical con tres círculos |

## Uso

1. Elige la cantidad de casos y el modo: equilibrado, aleatorio o manual (con el selector de clase).
2. Dibuja la señal en el lienzo de 200 × 200 px y pulsa **Guardar y continuar**.
3. Cada dibujo se descarga como PNG con la etiqueta en el nombre: `CLASE_sesion_###.png`.
4. Al final puedes descargar:
   - `senales_<sesion>.zip`: una carpeta por clase más `labels.csv`.
   - `X.npy` `(N, 200, 200) uint8` e `y.npy` `(N,) <U9`.

Todo se ejecuta en el navegador; no se envían dibujos a ningún servidor.

## Desarrollo local

No necesita instalación. Abre `index.html` en el navegador o sirve la carpeta:

```bash
npx serve .
```

## Despliegue

Es un sitio estático: importa el repositorio en Vercel y despliega sin configuración adicional.

Adaptado de [letras_upc](https://github.com/PeterTXS09/letras_upc).
