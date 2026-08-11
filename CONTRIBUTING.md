# Contributing

¡Gracias por tu interés en contribuir a **Gold Price Prediction**!

Este proyecto sigue una metodología rigurosa de 23 fases (ver
[`docs/methodology-guide.md`](docs/methodology-guide.md)). Antes de contribuir,
por favor lee esta guía.

## Cómo contribuir

1. **Abre un issue** para discutir el cambio propuesto (bug, mejora, nueva
   funcionalidad) antes de escribir código.
2. **Haz un fork** del repositorio y crea una rama descriptiva:
   `git checkout -b feature/nombre-descriptivo`
3. **Desarrolla** siguiendo las convenciones del proyecto.
4. **Añade tests** para cualquier código nuevo (pytest).
5. **Ejecuta las verificaciones** locales (ver más abajo).
6. **Abre un Pull Request** describiendo el cambio, su motivación y los
   resultados de las verificaciones.

## Verificaciones locales

```bash
# Tests unitarios
pytest tests/ -q

# Estilo (flake8)
flake8 src scripts tests --max-line-length=100 --extend-ignore=E203,W503

# Sintaxis
python -m compileall -q src scripts tests
```

El CI (GitHub Actions) ejecuta estas mismas verificaciones en cada PR.

## Convenciones

- **Código**: Python 3.11+, tipado con type hints, docstrings descriptivos.
- **Notebooks**: son el corazón del proyecto. Cada fase del
  `docs/methodology-guide.md` debe tener su notebook con celdas markdown
  explicativas antes de cada bloque de código.
- **Datos**: `data/raw/` es inmutable. Los derivados van a `data/interim/`
  y `data/processed/` (regenerables con `make data`).
- **Resultados**: cualquier cambio en métricas debe actualizar los reports
  (`reports/`) y la documentación asociada (README, model card, informe).

## Reportar bugs

Usa la plantilla de issue de GitHub e incluye:

- Descripción del problema y contexto.
- Pasos para reproducirlo.
- Comportamiento esperado vs. observado.
- Versión de Python y dependencias (o el contenido de `requirements.txt`).

## Licencia

Al contribuir aceptas que tu código se distribuya bajo la licencia MIT del
proyecto (ver `LICENSE`).
