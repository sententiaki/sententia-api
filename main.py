import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import requests
from bs4 import BeautifulSoup
from deep_translator import GoogleTranslator
from openai import OpenAI
import tiktoken
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID  = os.getenv("GOOGLE_CSE_ID")

if not (OPENAI_API_KEY and GOOGLE_API_KEY and GOOGLE_CSE_ID):
    raise ValueError("Chiavi API mancanti nel file .env.")

client = OpenAI(api_key=OPENAI_API_KEY)
MODEL  = "gpt-4o-mini"

app = Flask(__name__)
CORS(app)

limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://",
)

@app.errorhandler(429)
def troppe_richieste(e):
    return jsonify({"errore": "Limite di richieste superato. Riprova tra qualche minuto."}), 429


# ─── Utilità ────────────────────────────────────────────────────────────────

def traduci_parole_chiave(parole_chiave):
    def _traduci(target):
        return GoogleTranslator(source="it", target=target).translate(parole_chiave)

    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_de = ex.submit(_traduci, "de")
        fut_fr = ex.submit(_traduci, "fr")
        de = fut_de.result()
        fr = fut_fr.result()

    return {"it": parole_chiave, "de": de, "fr": fr}


def _cse_query(query):
    url = (
        f"https://www.googleapis.com/customsearch/v1"
        f"?q={query}+site:bger.ch"
        f"&key={GOOGLE_API_KEY}&cx={GOOGLE_CSE_ID}"
    )
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        risultati = []
        for item in resp.json().get("items", [])[:5]:
            titolo = item.get("title", "")
            link   = item.get("link", "")
            m = re.search(r"(\d+[A-Z]_\d+/\d+|\d+\s+[IVXLCDM]+\s+\d+)", titolo)
            if m:
                risultati.append({"codice": m.group(1), "link": link})
        return risultati
    except Exception:
        return []


def cerca_sentenze_google(parole_chiave):
    traduzioni = traduci_parole_chiave(parole_chiave)
    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = [ex.submit(_cse_query, q) for q in traduzioni.values()]
        risultati_finali = []
        for fut in as_completed(futures):
            risultati_finali.extend(fut.result())
    return risultati_finali[:5]


def costruisci_url_bgerli(codice):
    codice = codice.strip().replace(" ", "-").replace("/", "-")
    return f"https://bger.li/{codice}"


def estrai_testo_sentenza(url):
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        content = soup.find("div", id="content")
        return content.get_text(separator="\n").strip() if content else ""
    except Exception as e:
        return f"ERRORE:{e}"


def split_in_chunks(text, max_tokens=12000):
    enc    = tiktoken.encoding_for_model("gpt-4o")
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return [text]
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunks.append(enc.decode(tokens[i : i + max_tokens]))
    return chunks


def chiama_openai(system: str, user: str, max_tokens: int = 1200) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        max_tokens=max_tokens,
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def riassumi_con_chunking(testo: str, fn_call) -> str:
    chunks = split_in_chunks(testo)
    if len(chunks) == 1:
        return fn_call(testo)
    parziali = [fn_call(c) for c in chunks]
    return fn_call("\n\n".join(parziali))


# ─── Smart Search: sintesi compatta (~10 righe) ──────────────────────────────

SYSTEM_SEARCH = {
    "it": (
        "Sei un giurista svizzero esperto in diritto federale. "
        "Produci sintesi concise, precise e rigorosamente strutturate di sentenze del Tribunale federale."
    ),
    "de": (
        "Du bist ein erfahrener Schweizer Jurist im Bundesrecht. "
        "Erstelle präzise, strukturierte Zusammenfassungen von Bundesgerichtsurteilen."
    ),
    "fr": (
        "Vous êtes un juriste suisse expert en droit fédéral. "
        "Produisez des synthèses concises et rigoureusement structurées des arrêts du Tribunal fédéral."
    ),
}

PROMPT_SEARCH = {
    "it": (
        "Analizza la seguente sentenza del Tribunale federale svizzero e fornisci una sintesi strutturata:\n\n"
        "**Questione giuridica**: (1-2 frasi sulla questione centrale)\n"
        "**Fatti rilevanti**: (3-4 frasi sui fatti determinanti)\n"
        "**Decisione**: (2-3 frasi sull'esito e sulla motivazione principale)\n"
        "**Articoli applicati**: (lista puntata con i riferimenti normativi citati)\n\n"
        "Usa linguaggio giuridico preciso. Sii diretto ed essenziale.\n\n"
        "Testo della sentenza:\n{testo}"
    ),
    "de": (
        "Analysiere das folgende Urteil des Schweizer Bundesgerichts und erstelle eine strukturierte Zusammenfassung:\n\n"
        "**Rechtsfrage**: (1-2 Sätze zur zentralen Frage)\n"
        "**Relevanter Sachverhalt**: (3-4 Sätze zu den massgebenden Fakten)\n"
        "**Entscheid**: (2-3 Sätze zum Ergebnis und zur Hauptbegründung)\n"
        "**Angewendete Artikel**: (Stichpunkte mit den zitierten Normen)\n\n"
        "Verwende präzise juristische Sprache.\n\n"
        "Urteilstext:\n{testo}"
    ),
    "fr": (
        "Analysez l'arrêt du Tribunal fédéral suisse ci-dessous et fournissez une synthèse structurée:\n\n"
        "**Question juridique**: (1-2 phrases sur la question centrale)\n"
        "**Faits pertinents**: (3-4 phrases sur les faits déterminants)\n"
        "**Décision**: (2-3 phrases sur le résultat et la motivation principale)\n"
        "**Articles appliqués**: (liste à puces des références normatives citées)\n\n"
        "Utilisez un langage juridique précis.\n\n"
        "Texte de l'arrêt:\n{testo}"
    ),
}


def sintetizza_sentenza_10_righe(testo: str, lang: str = "it") -> str:
    l = lang if lang in PROMPT_SEARCH else "it"
    enc = tiktoken.encoding_for_model("gpt-4o")
    tokens = enc.encode(testo)
    if len(tokens) > 6000:
        testo = enc.decode(tokens[:6000])

    def call(t):
        return chiama_openai(
            system=SYSTEM_SEARCH[l],
            user=PROMPT_SEARCH[l].format(testo=t),
            max_tokens=550,
        )

    return call(testo)


# ─── Legal Summarization: analisi completa (4 punti) ────────────────────────

SYSTEM_SUMM = {
    "it": (
        "Sei un avvocato svizzero con profonda competenza in diritto federale. "
        "Produci analisi giuridiche professionali, strutturate e complete di sentenze del Tribunale federale. "
        "Il tuo linguaggio è tecnico, preciso e adatto a professionisti del diritto."
    ),
    "de": (
        "Du bist ein Schweizer Rechtsanwalt mit fundierter Expertise im Bundesrecht. "
        "Du erstellst professionelle, strukturierte und vollständige rechtliche Analysen von Bundesgerichtsurteilen. "
        "Deine Sprache ist technisch, präzise und für Rechtsfachleute geeignet."
    ),
    "fr": (
        "Vous êtes un avocat suisse avec une profonde expertise en droit fédéral. "
        "Vous produisez des analyses juridiques professionnelles, structurées et complètes des arrêts du Tribunal fédéral. "
        "Votre langage est technique, précis et adapté aux professionnels du droit."
    ),
}

PROMPT_SUMM = {
    "it": (
        "Analizza in modo completo e professionale la seguente sentenza del Tribunale federale svizzero. "
        "Struttura la tua analisi esattamente come segue:\n\n"
        "**1. Fattispecie**\n"
        "In 2-3 frasi: parti coinvolte, questione giuridica centrale e iter procedurale.\n\n"
        "**2. Articoli principali applicati**\n"
        "Elenca in modo puntuale tutti gli articoli di legge citati o applicati (indicando codice e numero, es. art. 41 CO, art. 146 CP).\n\n"
        "**3. Considerazioni del Tribunale**\n"
        "Esponi il ragionamento giuridico adottato dalla corte: interpretazione normativa, bilanciamento degli interessi, "
        "giurisprudenza richiamata e argomenti decisivi.\n\n"
        "**4. Dispositivo e implicazioni**\n"
        "In 2 frasi: esito del giudizio (accoglimento/rigetto/rinvio) e principale implicazione pratica.\n\n"
        "Testo della sentenza:\n{testo}"
    ),
    "de": (
        "Analysiere das folgende Urteil des Schweizer Bundesgerichts vollständig und professionell. "
        "Strukturiere deine Analyse genau wie folgt:\n\n"
        "**1. Sachverhalt**\n"
        "In 2-3 Sätzen: beteiligte Parteien, zentrale Rechtsfrage und Verfahrensgang.\n\n"
        "**2. Massgebende Rechtsartikel**\n"
        "Liste alle zitierten oder angewendeten Gesetzesartikel auf (mit Angabe des Gesetzes und Nummer, z.B. Art. 41 OR).\n\n"
        "**3. Erwägungen des Gerichts**\n"
        "Stelle die rechtliche Argumentation des Gerichts dar: Normeninterpretation, Interessenabwägung, "
        "herangezogene Rechtsprechung und entscheidende Argumente.\n\n"
        "**4. Dispositiv und Implikationen**\n"
        "In 2 Sätzen: Urteilsergebnis (Gutheissung/Abweisung/Rückweisung) und wichtigste praktische Implikation.\n\n"
        "Urteilstext:\n{testo}"
    ),
    "fr": (
        "Analysez de manière complète et professionnelle l'arrêt du Tribunal fédéral suisse ci-dessous. "
        "Structurez votre analyse exactement comme suit:\n\n"
        "**1. Faits et procédure**\n"
        "En 2-3 phrases: parties impliquées, question juridique centrale et déroulement de la procédure.\n\n"
        "**2. Articles principaux appliqués**\n"
        "Listez tous les articles de loi cités ou appliqués (avec indication du code et du numéro, ex. art. 41 CO).\n\n"
        "**3. Considérants du Tribunal**\n"
        "Exposez le raisonnement juridique: interprétation normative, pesée des intérêts, "
        "jurisprudence citée et arguments décisifs.\n\n"
        "**4. Dispositif et implications**\n"
        "En 2 phrases: issue du jugement (admission/rejet/renvoi) et principale implication pratique.\n\n"
        "Texte de l'arrêt:\n{testo}"
    ),
}


def sintetizza_testo_sentenza_4_punti(testo: str, lang: str = "it") -> str:
    l = lang if lang in PROMPT_SUMM else "it"

    def call(t):
        return chiama_openai(
            system=SYSTEM_SUMM[l],
            user=PROMPT_SUMM[l].format(testo=t),
            max_tokens=950,
        )

    return riassumi_con_chunking(testo, call)


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.route("/ricerca_sentenze", methods=["GET"])
@limiter.limit("5 per minute; 30 per day")
def ricerca_sentenze():
    query = request.args.get("query", "").strip()
    lang  = request.args.get("lang", "it")
    if not query:
        return jsonify({"errore": "Parametro 'query' mancante"}), 400

    sentenze = cerca_sentenze_google(query)

    def processa(s):
        url   = costruisci_url_bgerli(s["codice"])
        testo = estrai_testo_sentenza(url)
        if not testo or testo.startswith("ERRORE") or len(testo) < 100:
            sintesi = "Impossibile recuperare il testo della sentenza."
        else:
            sintesi = sintetizza_sentenza_10_righe(testo, lang)
        return {"titolo": s["codice"], "riassunto": sintesi, "link": url}

    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = {ex.submit(processa, s): i for i, s in enumerate(sentenze)}
        risultati = [None] * len(sentenze)
        for fut in as_completed(futures):
            risultati[futures[fut]] = fut.result()

    return jsonify([r for r in risultati if r is not None])


# ─── Law text retrieval ──────────────────────────────────────────────────────

LANG_CODE = {"it": "ITA", "de": "DEU", "fr": "FRA"}

def _sparql_find_law(sr: str, lang: str):
    """Query fedlex SPARQL for the HTML file URL of a law by SR number."""
    lc = LANG_CODE.get(lang, "ITA")
    query = f"""
PREFIX jolux: <http://data.legilux.public.lu/resource/ontology/jolux#>
SELECT DISTINCT ?act ?url WHERE {{
  ?act jolux:classifiedByTaxonomyEntry ?entry .
  ?entry ?p "{sr}" .
  OPTIONAL {{
    ?act jolux:isRealizedBy ?expr .
    ?expr jolux:language <http://publications.europa.eu/resource/authority/language/{lc}> .
    ?expr jolux:isEmbodiedBy ?manif .
    ?manif jolux:isExemplifiedBy ?url .
  }}
}}
LIMIT 10"""
    r = requests.get(
        "https://fedlex.data.admin.ch/sparql",
        params={"query": query, "format": "json"},
        headers={"Accept": "application/sparql-results+json"},
        timeout=10,
    )
    r.raise_for_status()
    bindings = r.json().get("results", {}).get("bindings", [])
    act_uri, html_url = None, None
    for b in bindings:
        act_uri = act_uri or b.get("act", {}).get("value")
        u = b.get("url", {}).get("value", "")
        if u and "html" in u.lower() and f"/{lang}/" in u:
            html_url = u
            break
    return act_uri, html_url


def _fetch_html_law(url: str):
    """Fetch HTML, trying Googlebot UA first (triggers SSR on some servers)."""
    user_agents = [
        "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
    ]
    for ua in user_agents:
        try:
            r = requests.get(url, headers={"User-Agent": ua, "Accept": "text/html"}, timeout=15)
            if r.ok and len(r.text) > 2000:
                soup = BeautifulSoup(r.text, "html.parser")
                # Reject Angular shell (empty app-root with no real text)
                app_root = soup.find("app-root")
                if app_root is None or app_root.get_text(strip=True):
                    return r.text, soup
        except Exception:
            pass
    return None, None


def _parse_law_soup(soup, sr: str):
    """Extract structured article list from parsed law HTML."""
    for tag in soup.find_all(["script", "style", "nav", "header", "footer", "iframe"]):
        tag.decompose()
    root = soup.find("app-root") or soup.find("main") or soup.body
    title_el = root.find("h1") if root else None
    title = title_el.get_text(strip=True) if title_el else f"SR {sr}"
    articoli = []
    if root:
        for el in root.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
            text = el.get_text(separator=" ", strip=True)
            if text and len(text) > 3:
                articoli.append({"tag": el.name, "text": text})
    return title, articoli[:600]


@app.route("/legge", methods=["GET"])
@limiter.limit("20 per minute; 100 per day")
def get_legge():
    sr   = request.args.get("sr", "").strip()
    lang = request.args.get("lang", "it").strip().lower()
    if not sr:
        return jsonify({"errore": "Parametro 'sr' mancante"}), 400

    act_uri, html_url = None, None

    # Step 1 – SPARQL lookup
    try:
        act_uri, html_url = _sparql_find_law(sr, lang)
    except Exception:
        pass

    # Step 2 – derive URL from ELI URI if SPARQL gave one but no html_url
    if act_uri and not html_url:
        eli_path = act_uri.replace("https://fedlex.data.admin.ch", "")
        html_url = f"https://fedlex.data.admin.ch/filestore/fedlex.data.admin.ch{eli_path}/{lang}/html/fedlex-data-admin-ch{eli_path.replace('/', '-')}-{lang}-html-1.html"

    fedlex_page = f"https://www.fedlex.admin.ch{act_uri.replace('https://fedlex.data.admin.ch','')}/{lang}" if act_uri else None

    if not html_url and not fedlex_page:
        return jsonify({"errore": f"Legge SR {sr} non trovata"}), 404

    # Step 3 – fetch and parse
    fetch_url = html_url or fedlex_page
    html_text, soup = _fetch_html_law(fetch_url)

    if not html_text or not soup:
        # Return metadata so frontend can show fallback
        return jsonify({
            "sr": sr,
            "url": fedlex_page or fetch_url,
            "solo_link": True,
        }), 206  # partial

    title, articoli = _parse_law_soup(soup, sr)
    return jsonify({
        "sr": sr,
        "titolo": title,
        "url": fedlex_page or fetch_url,
        "articoli": articoli,
    })


@app.route("/sintesi", methods=["GET"])
@limiter.limit("10 per minute; 50 per day")
def get_summary():
    codice = request.args.get("codice", "").strip()
    lang   = request.args.get("lang", "it")
    if not codice:
        return jsonify({"errore": "Parametro 'codice' mancante"}), 400

    url   = costruisci_url_bgerli(codice)
    testo = estrai_testo_sentenza(url)
    if not testo or testo.startswith("ERRORE") or len(testo) < 100:
        return jsonify({"errore": "Impossibile recuperare il testo della sentenza."}), 404

    sintesi = sintetizza_testo_sentenza_4_punti(testo, lang)
    return jsonify({"sintesi": sintesi})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
