# Security Policy

## Versiones soportadas

| Versión | Soportada |
|---------|-----------|
| 1.0.x   | Sí        |

## Reportar una vulnerabilidad

Este proyecto es de **uso académico e investigador**. No gestiona datos
personales ni sistemas en producción crítica, pero cualquier vulnerabilidad
debe reportarse de forma responsable.

**Por favor, NO abras un issue público para vulnerabilidades de seguridad.**
Envía un correo a los mantenedores (ver perfil de GitHub) con:

- Descripción de la vulnerabilidad.
- Pasos para reproducirla.
- Impacto potencial.
- Sugerencia de mitigación (si la tienes).

Los reportes se revisarán en un plazo máximo de 7 días. Una vez confirmada
la vulnerabilidad, se publicará un aviso y un parche en la siguiente versión.

## Buenas prácticas del proyecto

- **Sin secretos en el repositorio**: las claves y tokens van en variables
  de entorno (ver `.env.example`); `.env` está en `.gitignore`.
- **Validación de entradas**: la API rechaza valores NaN/Inf, claves no
  esperadas y fechas malformadas (HTTP 422).
- **Dependencias fijadas**: `requirements.txt` y `pyproject.toml` definen
  versiones mínimas verificadas.
- **Sin datos personales**: el dataset contiene series de mercado públicas.
- **Uso previsto**: este modelo NO debe utilizarse para trading automático
  sin supervisión humana (ver `docs/model_card.md`).
