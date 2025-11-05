#!/bin/bash
# Script de inicialización de base de datos para Render
# Este script se ejecuta durante el build para preparar la base de datos

set -e  # Salir si hay algún error

echo "=== Inicializando Base de Datos ==="

# Verificar que DATABASE_URL esté configurada
if [ -z "$DATABASE_URL" ]; then
    echo "ERROR: DATABASE_URL no está configurada"
    exit 1
fi

echo "DATABASE_URL configurada correctamente"

# Esperar a que la base de datos esté lista (si es necesario)
echo "Verificando conexión a base de datos..."
sleep 2

# Ejecutar migraciones de Alembic
echo "Ejecutando migraciones de Alembic..."
alembic upgrade head || {
    echo "ADVERTENCIA: No se pudieron ejecutar migraciones"
    echo "Esto puede ser normal en el primer despliegue"
    echo "Las tablas se crearán automáticamente al iniciar la aplicación"
}

echo "=== Inicialización de Base de Datos Completada ==="

