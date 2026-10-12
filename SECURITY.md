# Seguridad y alcance

Demostración educativa local con datos sintéticos, sin exposición pública.

## Invariantes que revisar

- No publicar claves, tokens, credenciales Google ni datos personales.
- API exclusivamente en 127.0.0.1, con destino local validado y CORS solo
  para https://isbetter.ai.
- Limitar cuerpos, contexto, salida, conexiones e inferencias pendientes.
- Clientes HTTP con lectura streaming, máximo 1 MiB descomprimido (8 MiB solo
  para el catálogo público de modelos de OpenRouter) y cierre
  de respuesta, sin redirecciones automáticas; validación de modelo hasta
  16.384 caracteres, profundidad JSON 32,
  listas de partes 64 y catálogo local 256 modelos.
- Los mensajes HTTP solo eligen las dos condiciones anunciadas; no archivos,
  adaptadores, proyectos cloud ni herramientas.
- No ejecutar salida del modelo: JSON se interpreta con validación estricta,
  sin eval, reparación ni reintentos correctores.
- CSV neutraliza prefijos de fórmulas y controles; JSON/JSONL conserva el
  texto original para evaluar, incluidos errores.
- Aislar el estado del adaptador entre consultas concurrentes.
- Qwen con revisión fijada, Safetensors y trust_remote_code desactivado.
- Conservar artefactos anteriores; las nuevas ejecuciones usan rutas nuevas.
- Llamadas de pago solo con configuración y ejecución explícitas.

## Supuestos de confianza

El operador controla el repo, sus rutas, adaptador y variables de entorno.
Los argumentos CLI no son entradas de usuarios remotos. El acceso de un
atacante con control de ese mismo usuario del sistema no se atribuye a la API.

isbetter.ai es el origen web autorizado. CORS no corrige un sitio autorizado
o una dependencia comprometidos. Instala dependencias en un entorno separado.

## Uso seguro

No exponer el puerto por LAN, proxy inverso, túnel ni servicio público.
Usa únicamente datos sintéticos. Los costes de hardware/energía no se miden.
Los proveedores remotos (Anthropic, OpenAI, OpenRouter, APIs compatibles,
Gemini, Google Cloud) y la publicación HF son rutas CLI opcionales del
operador, nunca acciones de la API ni efectos de importar módulos. Las claves
se leen del entorno; los registros guardan parámetros y metadatos, no claves.
`COMPAT_BASE_URL` remota exige https: las claves solo viajan sin cifrar hacia
un servidor del propio equipo.

## Publicación y comunicación de problemas

Git excluye .env, pesos, resultados, carpetas de trabajo local y metadatos de editores.
No incluyas secretos o datos privados en issues públicos. Una clave filtrada
requiere revocación/rotación con su proveedor; borrarla no revoca su acceso.
Las revisiones se refieren a su versión y alcance, no certifican despliegues.
Los locks fijan versiones, hashes y marcadores por plataforma. El estado de
avisos conocidos y las verificaciones se detallan en docs/publicacion.md.
