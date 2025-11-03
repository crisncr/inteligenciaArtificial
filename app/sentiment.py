import re
import string
from typing import Dict


# Diccionario expandido de palabras positivas en español
POSITIVE_WORDS = {
    # Calificativos generales positivos
    "excelente", "bueno", "buenisimo", "buenísimo", "genial", "fantastico", "fantástico",
    "maravilloso", "increible", "increíble", "agradable", "feliz", "perfecto", "ideal",
    "súper", "super", "extraordinario", "espectacular", "impresionante", "sorprendente",
    "increíble", "asombroso", "brillante", "genial", "estupendo", "fabuloso", "fenomenal",
    "formidable", "magnífico", "sensacional", "sobresaliente", "notable", "destacado",
    
    # Sentimientos y emociones positivas
    "encanta", "encantado", "encantada", "encantador", "encantadora", "amor", "adoro",
    "adorable", "feliz", "alegre", "contento", "contenta", "satisfecho", "satisfecha",
    "satisfacción", "satisfactorio", "satisfactoria", "emocionado", "emocionada",
    "emocionante", "entusiasmado", "entusiasmada", "orgulloso", "orgullosa", "agradecido",
    "agradecida", "agradecimiento", "esperanzado", "esperanzada", "optimista",
    
    # Servicio y atención
    "atencion", "atención", "amable", "amabilidad", "cortés", "educado", "educada",
    "respetuoso", "respetuosa", "atento", "atenta", "servicial", "disponible",
    "eficiente", "eficaz", "profesional", "competente", "experto", "experta",
    "cualificado", "cualificada", "preparado", "preparada", "experimentado", "experimentada",
    
    # Velocidad y eficiencia
    "rapido", "rápido", "rapida", "rápida", "veloz", "inmediato", "inmediata",
    "instantaneo", "instantáneo", "instantanea", "instantánea", "pronto", "puntual",
    "eficiente", "eficaz", "productivo", "productiva", "ágil", "expeditivo", "expeditiva",
    
    # Calidad y resultados
    "calidad", "premium", "superior", "excepcional", "único", "unico", "única", "unica",
    "especial", "premium", "premium", "refinado", "refinada", "pulido", "pulida",
    "cuidado", "cuidadoso", "cuidadosa", "detallado", "detallada", "completo", "completa",
    
    # Recomendación y confianza
    "recomendado", "recomendada", "recomiendo", "recomendamos", "recomendable",
    "confiable", "confiabilidad", "seguro", "segura", "garantizado", "garantizada",
    "verificado", "verificada", "probado", "probada", "testado", "testada",
    
    # Economía y valor
    "barato", "barata", "económico", "economico", "económica", "economica", "accesible",
    "asequible", "rentable", "valor", "vale", "vale_la_pena", "conveniente", "beneficioso",
    "beneficiosa", "ventajoso", "ventajosa", "oportuno", "oportuna",
    
    # Comodidad y facilidad
    "fácil", "facil", "sencillo", "sencilla", "simple", "cómodo", "comodo", "cómoda",
    "comoda", "práctico", "practico", "práctica", "practica", "útil", "util", "conveniente",
    "accesible", "intuitivo", "intuitiva", "user-friendly", "amigable",
    
    # Innovación y modernidad
    "moderno", "moderna", "innovador", "innovadora", "actualizado", "actualizada",
    "nuevo", "nueva", "fresco", "fresca", "original", "creativo", "creativa",
    "vanguardista", "avanzado", "avanzada", "tecnológico", "tecnologico", "tecnológica",
    
    # Limpieza y orden
    "limpio", "limpia", "ordenado", "ordenada", "organizado", "organizada", "pulcro",
    "pulcra", "inmaculado", "inmaculada", "higiénico", "higienico", "higiénica",
    
    # Éxito y logro
    "éxito", "exito", "exitoso", "exitosa", "triunfador", "triunfadora", "ganador",
    "ganadora", "victorioso", "victoriosa", "logrado", "lograda", "conseguido", "conseguida",
    
    # Otros términos positivos
    "mejor", "mejora", "mejorado", "mejorada", "mejorando", "progreso", "avance",
    "evolución", "crecimiento", "desarrollo", "potencial", "oportunidad", "ventaja",
    "beneficio", "beneficios", "ventaja", "ventajas", "pros", "positivo", "positiva",
}

# Diccionario expandido de palabras negativas en español
NEGATIVE_WORDS = {
    # Calificativos generales negativos
    "pesimo", "pésimo", "pesima", "pésima", "malo", "mala", "malisimo", "malísimo",
    "malisima", "malísima", "terrible", "horrible", "fatal", "desagradable", "pobre",
    "asqueroso", "asquerosa", "nefasta", "nefasto", "nefastos", "nefastas",
    "desastroso", "desastrosa", "desastre", "desastres", "catastrófico", "catastrofico",
    "catastrófica", "catastrofica", "ruinoso", "ruinosa", "ruin", "ruines",
    
    # Sentimientos y emociones negativas
    "triste", "tristeza", "tristemente", "odio", "odiar", "odioso", "odiosa",
    "decepcion", "decepción", "decepcionado", "decepcionada", "decepcionante",
    "desilusion", "desilusión", "desilusionado", "desilusionada", "desilusionante",
    "frustrado", "frustrada", "frustración", "frustracion", "frustrante",
    "enojado", "enojada", "enojoso", "enojosa", "molesto", "molesta", "molestia",
    "molestias", "irritado", "irritada", "irritante", "irritación", "irritacion",
    "furioso", "furiosa", "furor", "rabia", "rabioso", "rabiosa", "angustiado",
    "angustiada", "angustia", "angustioso", "angustiosa", "deprimido", "deprimida",
    "depresión", "depresion", "depresivo", "depresiva", "tristeza", "melancolía",
    "melancolia", "melancólico", "melancolico", "melancólica", "melancolica",
    
    # Velocidad y lentitud
    "lento", "lenta", "lentamente", "lentitud", "tardio", "tardío", "tardia",
    "tardía", "tarde", "tardanza", "tardanzas", "retrasado", "retrasada",
    "retraso", "retrasos", "demorado", "demorada", "demora", "demoras",
    "lentamente", "despacio", "despaciosamente", "pausado", "pausada",
    
    # Precio y economía
    "caro", "cara", "carísimo", "carisimo", "carísima", "carisima", "costoso",
    "costosa", "coste", "costos", "costes", "carísimo", "carisimo", "sobreprecio",
    "sobre-precio", "sobreprecios", "sobre-precios", "sobrecargado", "sobrecargada",
    "sobrecarga", "sobrecargas", "excesivo", "excesiva", "exceso", "excesos",
    "inflado", "inflada", "inflación", "inflacion", "inflacionario", "inflacionaria",
    
    # Calidad y deficiencia
    "deficiente", "deficiencia", "deficiencias", "defectuoso", "defectuosa",
    "defecto", "defectos", "fallo", "fallos", "falla", "fallas", "error",
    "errores", "mal", "malo", "mala", "malos", "malas", "mal funcionamiento",
    "mal-funcionamiento", "disfuncional", "disfunción", "disfuncion",
    "ineficiente", "ineficacia", "ineficiencia", "ineficaz", "inadecuado",
    "inadecuada", "inadecuados", "inadecuadas", "inapropiado", "inapropiada",
    "inapropiados", "inapropiadas", "incompleto", "incompleta", "incompletos",
    "incompletas", "inferior", "inferiores", "subestándar", "subestandar",
    "baja_calidad", "baja-calidad", "baja calidad", "mediocre", "mediocridad",
    
    # Servicio y atención negativa
    "atencion_pesima", "atención_pésima", "atencion-pesima", "atención-pésima",
    "atencion pesima", "atención pésima", "mal servicio", "mal-servicio",
    "mal_servicio", "servicio deficiente", "servicio-deficiente", "servicio_deficiente",
    "mala atención", "mala-atencion", "mala_atencion", "desatento", "desatenta",
    "desatención", "desatencion", "desatendido", "desatendida", "desatender",
    "desatendiendo", "ignorado", "ignorada", "ignorar", "ignorando", "desprecio",
    "despreciar", "despreciando", "despreciado", "despreciada", "despreciativo",
    "despreciativa",
    
    # Problemas y quejas
    "problema", "problemas", "queja", "quejas", "reclamo", "reclamos", "reclamar",
    "reclamando", "reclamado", "reclamada", "reclamación", "reclamacion",
    "reclamaciones", "reclamaciones", "denuncia", "denuncias", "denunciar",
    "denunciando", "denunciado", "denunciada", "denuncia", "denuncias",
    "incidencia", "incidencias", "incidente", "incidentes", "conflicto",
    "conflictos", "disputa", "disputas", "disputar", "disputando", "disputado",
    "disputada", "pelea", "peleas", "pelear", "peleando", "peleado", "peleada",
    "discusión", "discusion", "discusiones", "discusiones", "discutir",
    "discutiendo", "discutido", "discutida",
    
    # Otros términos negativos
    "nunca", "jamás", "jamas", "nadie", "ninguno", "ninguna", "ningunos",
    "ningunas", "nada", "nunca más", "nunca-mas", "nunca_mas", "nunca más",
    "peor", "peorar", "peorando", "peorado", "peorada", "empeorar", "empeorando",
    "empeorado", "empeorada", "deterioro", "deterioros", "deteriorar",
    "deteriorando", "deteriorado", "deteriorada", "deterioración", "deterioracion",
    "deterioraciones", "deterioraciones", "regresión", "regresion", "regresiones",
    "regresiones", "retroceso", "retrocesos", "retroceder", "retrocediendo",
    "retrocedido", "retrocedida", "retroceso", "retrocesos", "fracaso", "fracasos",
    "fracasar", "fracasando", "fracasado", "fracasada", "fallido", "fallida",
    "fallidos", "fallidas", "fallar", "fallando", "fallado", "fallada",
    
    # Insatisfacción
    "insatisfecho", "insatisfecha", "insatisfacción", "insatisfaccion",
    "insatisfacciones", "insatisfacciones", "insatisfactorio", "insatisfactoria",
    "insatisfactorios", "insatisfactorias", "disgusto", "disgustos", "disgustar",
    "disgustando", "disgustado", "disgustada", "disgustoso", "disgustosa",
    "disgustante", "repugnante", "repugnancia", "repugnancias", "repugnar",
    "repugnando", "repugnado", "repugnada", "repulsivo", "repulsiva",
    "repulsivos", "repulsivas", "asqueroso", "asquerosa", "asquerosos",
    "asquerosas", "asquerosamente", "asquerosidad", "asquerosidades",
    
    # Peligro y riesgo
    "peligroso", "peligrosa", "peligrosos", "peligrosas", "peligro", "peligros",
    "riesgoso", "riesgosa", "riesgosos", "riesgosas", "riesgo", "riesgos",
    "arriesgado", "arriesgada", "arriesgados", "arriesgadas", "arriesgar",
    "arriesgando", "arriesgado", "arriesgada", "inseguro", "insegura", "inseguros",
    "inseguras", "inseguridad", "inseguridades", "inestable", "inestables",
    "inestabilidad", "inestabilidades", "precario", "precaria", "precarios",
    "precarias", "precariedad", "precariedades",
    
    # Suciedad y desorden
    "sucio", "sucia", "sucios", "sucias", "suciedad", "suciedades", "desorden",
    "desordenes", "desordenes", "desordenado", "desordenada", "desordenados",
    "desordenadas", "desorganizado", "desorganizada", "desorganizados",
    "desorganizadas", "desorganización", "desorganizacion", "desorganizaciones",
    "desorganizaciones", "caos", "caótico", "caotico", "caótica", "caotica",
    "caóticos", "caoticos", "caóticas", "caoticas", "caóticamente", "caoticamente",
    
    # Otros términos negativos adicionales
    "negativo", "negativa", "negativos", "negativas", "negativamente", "negatividad",
    "negatividades", "desventaja", "desventajas", "contra", "contras", "desventaja",
    "desventajas", "inconveniente", "inconvenientes", "desventaja", "desventajas",
    "problema", "problemas", "dificultad", "dificultades", "dificultoso",
    "dificultosa", "dificultosos", "dificultosas", "complicado", "complicada",
    "complicados", "complicadas", "complicación", "complicacion", "complicaciones",
    "complicaciones", "complejo", "compleja", "complejos", "complejas", "complejidad",
    "complejidades",
}

NEGATIONS = {
    "no", "nunca", "jamás", "jamas", "nadie", "ninguno", "ninguna", "ningunos",
    "ningunas", "nada", "ni", "ni siquiera", "ni-siquiera", "ni_siquiera",
    "tampoco", "sin", "sin embargo", "sin-embargo", "sin_embargo",
}

INTENSIFIERS = {
    "muy": 1.5,
    "super": 1.5,
    "súper": 1.5,
    "re": 1.3,
    "tan": 1.2,
    "bastante": 1.2,
    "extremadamente": 1.8,
    "extremamente": 1.8,
    "sumamente": 1.6,
    "completamente": 1.5,
    "totalmente": 1.5,
    "absolutamente": 1.7,
    "realmente": 1.3,
    "verdaderamente": 1.4,
    "verdaderamente": 1.4,
    "increíblemente": 1.6,
    "increiblemente": 1.6,
    "extraordinariamente": 1.7,
    "extraordinariamente": 1.7,
    "especialmente": 1.3,
    "particularmente": 1.3,
    "particularmente": 1.3,
    "demasiado": 1.4,
    "demasiada": 1.4,
    "demasiados": 1.4,
    "demasiadas": 1.4,
}

DEINTENSIFIERS = {
    "poco": 0.5,
    "poca": 0.5,
    "pocos": 0.5,
    "pocas": 0.5,
    "algo": 0.8,
    "un poco": 0.6,
    "un-poco": 0.6,
    "un_poco": 0.6,
    "ligeramente": 0.7,
    "levemente": 0.7,
    "moderadamente": 0.8,
    "parcialmente": 0.7,
    "parcialmente": 0.7,
    "relativamente": 0.8,
    "relativamente": 0.8,
}


PUNCT_EMPHASIS = {"!": 0.2, "?": 0.05}


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower().strip()
    # Reemplazos simples para tratar compuestos frecuentes
    text = text.replace("pésima atención", "atencion_pesima")
    text = text.replace("pesima atencion", "atencion_pesima")
    return text


def tokenize(text: str):
    # Mantener signos de exclamación/interrogación para énfasis
    tokens = re.findall(r"[\wáéíóúñ]+|[!?]", text.lower(), flags=re.UNICODE)
    return tokens


def compute_punct_emphasis(text: str) -> float:
    bonus = 0.0
    for ch, weight in PUNCT_EMPHASIS.items():
        count = text.count(ch)
        if count:
            bonus += min(3, count) * weight
    return bonus


def analyze_sentiment(text: str) -> Dict[str, object]:
    text_norm = normalize_text(text)
    tokens = tokenize(text_norm)

    score = 0.0
    i = 0
    while i < len(tokens):
        token = tokens[i]

        if token in {"!", "?"}:
            i += 1
            continue

        # Intensificadores/deintensificadores miran a la siguiente palabra
        multiplier = 1.0
        if i + 1 < len(tokens):
            next_token = tokens[i + 1]
            if next_token in INTENSIFIERS:
                multiplier *= INTENSIFIERS[next_token]
                i += 1  # saltar intensificador
            elif next_token in DEINTENSIFIERS:
                multiplier *= DEINTENSIFIERS[next_token]
                i += 1

        token_score = 0.0
        if token in POSITIVE_WORDS:
            token_score = 1.0 * multiplier
        elif token in NEGATIVE_WORDS:
            token_score = -1.0 * multiplier

        # Negación en ventana corta hacia atrás (hasta 2 palabras)
        if token_score != 0:
            window_start = max(0, i - 2)
            window = tokens[window_start:i]
            if any(w in NEGATIONS for w in window):
                token_score *= -0.8  # invertir y atenuar

        score += token_score
        i += 1

    # Énfasis por signos de puntuación finales
    score += compute_punct_emphasis(text_norm)

    # Normalización simple
    if score > 0.5:
        label = "positivo"
        emoji = "🟢"
    elif score < -0.5:
        label = "negativo"
        emoji = "🔴"
    else:
        label = "moderado/neutral"
        emoji = "🟡"

    return {
        "text": text,
        "score": round(score, 3),
        "sentiment": label,
        "emoji": emoji,
    }


