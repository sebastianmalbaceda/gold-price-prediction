# Security Policy

## Versiones soportadas

| Versión | Soportada |
|---------|-----------|
| 1.1.x   | Sí        |
| 1.0.x   | No        |

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
  esperadas, rangos absurdos y fechas malformadas (HTTP 422). `/health` es
  liveness y `/ready` confirma que los artefactos existen y son compatibles.
- **Dependencias declaradas**: `requirements.txt` y `pyproject.toml` definen
  límites mínimos y superiores donde procede; un despliegue productivo debe
  generar además un lockfile con hashes y ejecutar un escáner de vulnerabilidades.
- **Sin datos personales**: el dataset contiene series de mercado públicas.
- **Uso previsto**: este modelo NO debe utilizarse para trading automático
  sin supervisión humana. La API no incluye autenticación, rate limiting ni
  TLS; no debe exponerse directamente a Internet (ver `docs/model_card.md`).
