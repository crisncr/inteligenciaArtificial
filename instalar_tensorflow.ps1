# Script PowerShell para instalar TensorFlow en Windows
# Ejecutar como Administrador

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "INSTALACIÓN DE TENSORFLOW EN WINDOWS" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Verificar si se ejecuta como administrador
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "⚠️  ADVERTENCIA: No se está ejecutando como Administrador" -ForegroundColor Yellow
    Write-Host "   Para habilitar rutas largas, ejecuta este script como Administrador" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "   Clic derecho en PowerShell -> Ejecutar como administrador" -ForegroundColor Yellow
    Write-Host ""
    $continue = Read-Host "¿Continuar de todos modos? (S/N)"
    if ($continue -ne "S" -and $continue -ne "s") {
        exit
    }
}

# Paso 1: Habilitar rutas largas (solo si es administrador)
if ($isAdmin) {
    Write-Host "🔧 Paso 1: Habilitando soporte de rutas largas..." -ForegroundColor Green
    try {
        $regPath = "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem"
        $regName = "LongPathsEnabled"
        $regValue = 1
        
        $currentValue = Get-ItemProperty -Path $regPath -Name $regName -ErrorAction SilentlyContinue
        
        if ($currentValue.LongPathsEnabled -eq 1) {
            Write-Host "   ✅ Rutas largas ya están habilitadas" -ForegroundColor Green
        } else {
            New-ItemProperty -Path $regPath -Name $regName -Value $regValue -PropertyType DWORD -Force | Out-Null
            Write-Host "   ✅ Rutas largas habilitadas" -ForegroundColor Green
            Write-Host "   ⚠️  IMPORTANTE: Necesitas reiniciar la computadora para que surta efecto" -ForegroundColor Yellow
            Write-Host ""
            $restart = Read-Host "   ¿Reiniciar ahora? (S/N)"
            if ($restart -eq "S" -or $restart -eq "s") {
                Restart-Computer -Force
                exit
            }
        }
    } catch {
        Write-Host "   ❌ Error al habilitar rutas largas: $_" -ForegroundColor Red
    }
} else {
    Write-Host "⚠️  Paso 1 omitido: Se necesita ejecutar como Administrador" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "🔧 Paso 2: Desinstalando TensorFlow anterior (si existe)..." -ForegroundColor Green
pip uninstall tensorflow tensorflow-cpu tensorflow-intel -y 2>$null
Write-Host "   ✅ Limpieza completada" -ForegroundColor Green

Write-Host ""
Write-Host "🔧 Paso 3: Instalando TensorFlow 2.16.1..." -ForegroundColor Green
Write-Host "   ⏳ Esto puede tardar varios minutos..." -ForegroundColor Yellow
Write-Host ""

# Intentar instalar TensorFlow
$installSuccess = $false
pip install tensorflow==2.16.1 --no-cache-dir
if ($LASTEXITCODE -eq 0) {
    $installSuccess = $true
}

if (-not $installSuccess) {
    Write-Host ""
    Write-Host "⚠️  Instalación falló. Intentando con tensorflow-cpu..." -ForegroundColor Yellow
    pip install tensorflow-cpu==2.16.1 --no-cache-dir
    if ($LASTEXITCODE -eq 0) {
        $installSuccess = $true
    }
}

Write-Host ""
if ($installSuccess) {
    Write-Host "✅ TensorFlow instalado correctamente" -ForegroundColor Green
    Write-Host ""
    Write-Host "🔍 Verificando instalación..." -ForegroundColor Cyan
    python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__} instalado correctamente')"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Green
        Write-Host "✅ INSTALACIÓN EXITOSA" -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Green
    } else {
        Write-Host ""
        Write-Host "⚠️  TensorFlow instalado pero hay problemas al importarlo" -ForegroundColor Yellow
        Write-Host "   Puede ser necesario reiniciar la computadora" -ForegroundColor Yellow
    }
} else {
    Write-Host ""
    Write-Host "❌ La instalación falló" -ForegroundColor Red
    Write-Host ""
    Write-Host "SOLUCIONES:" -ForegroundColor Yellow
    Write-Host "1. Ejecuta este script como Administrador" -ForegroundColor Yellow
    Write-Host "2. Reinicia la computadora después de habilitar rutas largas" -ForegroundColor Yellow
    Write-Host "3. Usa un entorno virtual de Python con una ruta más corta" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Presiona Enter para salir..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
