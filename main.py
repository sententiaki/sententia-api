import os
import re
import time
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


# ‚îÄ‚îÄ‚îÄ Utilit√† ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ

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


# ‚îÄ‚îÄ‚îÄ Smart Search: sintesi compatta (~10 righe) ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ

SYSTEM_SEARCH = {
    "it": (
        "Sei un giurista svizzero esperto in diritto federale. "
        "Produci sintesi concise, precise e rigorosamente strutturate di sentenze del Tribunale federale."
    ),
    "de": (
        "Du bist ein erfahrener Schweizer Jurist im Bundesrecht. "
        "Erstelle pr√§zise, strukturierte Zusammenfassungen von Bundesgerichtsurteilen."
    ),
    "fr": (
        "Vous √™tes un juriste suisse expert en droit f√©d√©ral. "
        "Produisez des synth√®ses concises et rigoureusement structur√©es des arr√™ts du Tribunal f√©d√©ral."
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
        "**Rechtsfrage**: (1-2 S√§tze zur zentralen Frage)\n"
        "**Relevanter Sachverhalt**: (3-4 S√§tze zu den massgebenden Fakten)\n"
        "**Entscheid**: (2-3 S√§tze zum Ergebnis und zur Hauptbegr√ºndung)\n"
        "**Angewendete Artikel**: (Stichpunkte mit den zitierten Normen)\n\n"
        "Verwende pr√§zise juristische Sprache.\n\n"
        "Urteilstext:\n{testo}"
    ),
    "fr": (
        "Analysez l'arr√™t du Tribunal f√©d√©ral suisse ci-dessous et fournissez une synth√®se structur√©e:\n\n"
        "**Question juridique**: (1-2 phrases sur la question centrale)\n"
        "**Faits pertinents**: (3-4 phrases sur les faits d√©terminants)\n"
        "**D√©cision**: (2-3 phrases sur le r√©sultat et la motivation principale)\n"
        "**Articles appliqu√©s**: (liste √† puces des r√©f√©rences normatives cit√©es)\n\n"
        "Utilisez un langage juridique pr√©cis.\n\n"
        "Texte de l'arr√™t:\n{testo}"
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


# ‚îÄ‚îÄ‚îÄ Legal Summarization: analisi completa (4 punti) ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ

SYSTEM_SUMM = {
    "it": (
        "Sei un avvocato svizzero con profonda competenza in diritto federale. "
        "Produci analisi giuridiche professionali, strutturate e complete di sentenze del Tribunale federale. "
        "Il tuo linguaggio √® tecnico, preciso e adatto a professionisti del diritto."
    ),
    "de": (
        "Du bist ein Schweizer Rechtsanwalt mit fundierter Expertise im Bundesrecht. "
        "Du erstellst professionelle, strukturierte und vollst√§ndige rechtliche Analysen von Bundesgerichtsurteilen. "
        "Deine Sprache ist technisch, pr√§zise und f√ºr Rechtsfachleute geeignet."
    ),
    "fr": (
        "Vous √™tes un avocat suisse avec une profonde expertise en droit f√©d√©ral. "
        "Vous produisez des analyses juridiques professionnelles, structur√©es et compl√®tes des arr√™ts du Tribunal f√©d√©ral. "
        "Votre langage est technique, pr√©cis et adapt√© aux professionnels du droit."
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
        "Analysiere das folgende Urteil des Schweizer Bundesgerichts vollst√§ndig und professionell. "
        "Strukturiere deine Analyse genau wie folgt:\n\n"
        "**1. Sachverhalt**\n"
        "In 2-3 S√§tzen: beteiligte Parteien, zentrale Rechtsfrage und Verfahrensgang.\n\n"
        "**2. Massgebende Rechtsartikel**\n"
        "Liste alle zitierten oder angewendeten Gesetzesartikel auf (mit Angabe des Gesetzes und Nummer, z.B. Art. 41 OR).\n\n"
        "**3. Erw√§gungen des Gerichts**\n"
        "Stelle die rechtliche Argumentation des Gerichts dar: Normeninterpretation, Interessenabw√§gung, "
        "herangezogene Rechtsprechung und entscheidende Argumente.\n\n"
        "**4. Dispositiv und Implikationen**\n"
        "In 2 S√§tzen: Urteilsergebnis (Gutheissung/Abweisung/R√ºckweisung) und wichtigste praktische Implikation.\n\n"
        "Urteilstext:\n{testo}"
    ),
    "fr": (
        "Analysez de mani√®re compl√®te et professionnelle l'arr√™t du Tribunal f√©d√©ral suisse ci-dessous. "
        "Structurez votre analyse exactement comme suit:\n\n"
        "**1. Faits et proc√©dure**\n"
        "En 2-3 phrases: parties impliqu√©es, question juridique centrale et d√©roulement de la proc√©dure.\n\n"
        "**2. Articles principaux appliqu√©s**\n"
        "Listez tous les articles de loi cit√©s ou appliqu√©s (avec indication du code et du num√©ro, ex. art. 41 CO).\n\n"
        "**3. Consid√©rants du Tribunal**\n"
        "Exposez le raisonnement juridique: interpr√©tation normative, pes√©e des int√©r√™ts, "
        "jurisprudence cit√©e et arguments d√©cisifs.\n\n"
        "**4. Dispositif et implications**\n"
        "En 2 phrases: issue du jugement (admission/rejet/renvoi) et principale implication pratique.\n\n"
        "Texte de l'arr√™t:\n{testo}"
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


# ‚îÄ‚îÄ‚îÄ Endpoints ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ

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


# ‚îÄ‚îÄ‚îÄ Law text retrieval via Fedlex SPARQL + public filestore ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ

SPARQL_ENDPOINT  = "https://fedlex.data.admin.ch/sparqlendpoint"
PRIVATE_FILESTORE = "https://intranet.fedlex.admin.ch/casematesbo/"
PUBLIC_FILESTORE  = "https://fedlex.data.admin.ch/"

# In-memory cache: {(sr, lang): (timestamp, (title, elements))}
_law_cache: dict = {}
LAW_CACHE_TTL = 3600  # 1 hour


def _sparql_html_url(sr: str, lang: str) -> str | None:
    """Get the most recent public HTML file URL for a law via Fedlex SPARQL."""
    query = f"""PREFIX jolux: <http://data.legilux.public.lu/resource/ontology/jolux#>
SELECT ?htmlUrl WHERE {{
  ?act jolux:historicalLegalId "{sr}" .
  ?dateAct jolux:isMemberOf ?act .
  ?dateAct jolux:isRealizedBy ?expr .
  FILTER(STRENDS(STR(?expr), "/{lang}"))
  ?expr jolux:isEmbodiedBy ?htmlManif .
  FILTER(STRENDS(STR(?htmlManif), "/html"))
  ?htmlManif jolux:isExemplifiedByPrivate ?htmlUrl .
}}
ORDER BY DESC(STR(?dateAct)) LIMIT 1"""
    r = requests.post(
        SPARQL_ENDPOINT, data={"query": query},
        headers={"Accept": "application/sparql-results+json"}, timeout=12
    )
    r.raise_for_status()
    bindings = r.json().get("results", {}).get("bindings", [])
    if not bindings:
        return None
    private_url = bindings[0]["htmlUrl"]["value"]
    return private_url.replace(PRIVATE_FILESTORE, PUBLIC_FILESTORE)


def _parse_fedlex_html(html: str) -> tuple:
    """Parse fedlex law HTML into (title, elements) using fast regex (no BS4 needed)."""
    # Title
    t = re.search(r'class="erlasstitel[^"]*"[^>]*>(.*?)</h\d>', html, re.DOTALL)
    s = re.search(r'class="erlasskurztitel[^"]*"[^>]*>(.*?)</h\d>', html, re.DOTALL)
    title = re.sub(r'<[^>]+>', '', t.group(1)) if t else ""
    if s:
        title += " " + re.sub(r'<[^>]+>', '', s.group(1))
    title = re.sub(r'\s+', ' ', title).strip()

    # Main content area
    main_m = re.search(r'<main[^>]*id="maintext"[^>]*>(.*)', html, re.DOTALL)
    if not main_m:
        main_m = re.search(r'<div[^>]*id="lawcontent"[^>]*>(.*)', html, re.DOTALL)
    content = main_m.group(1) if main_m else html

    events = []  # (position, element_dict)

    # Section headings: <h1-5 class="heading"...><a href="...">TEXT</a>
    for m in re.finditer(
        r'<(h[1-5])\s[^>]*class="heading[^"]*"[^>]*>.*?<a\s[^>]*href="[^"]*"[^>]*>(.*?)</a>',
        content, re.DOTALL
    ):
        text = re.sub(r'<[^>]+>', '', m.group(2))
        text = re.sub(r'\s+', ' ', text).strip()
        if text:
            events.append((m.start(), {"type": m.group(1), "text": text}))

    # Articles: <article id="art_N">...</article>
    for m in re.finditer(
        r'<article\s+id="(art_[^"]+)"[^>]*>(.*?)</article>',
        content, re.DOTALL
    ):
        art_id = m.group(1)
        body   = m.group(2)

        # Article heading: built from art_id (reliable), optionally enriched by <b> title
        raw_num = art_id[4:]  # strip "art_"
        art_num_str = re.sub(r'_([a-z])', r'\1', raw_num)  # "653_a"->"653a"
        heading = f"Art. {art_num_str}"
        # Check for title text in heading (some articles have "Art. N ‚Äì Title")
        hm = re.search(r'<b>(Art\.[^<]*)</b>(?:<i>([^<]{1,10})</i>)?', body)
        if hm:
            title_extra = re.sub(r'<[^>]+>', '', hm.group(1))
            title_extra = title_extra.replace('\u00a0', ' ').strip()
            # Remove the number part, keep any title text after it
            title_extra = re.sub(r'^Art\.?\s*[\d\w]*\s*', '', title_extra).strip()
            if title_extra:
                heading += f" ‚Äì {title_extra}"

        def _clean_para(inner, is_cpv):
            # Strip footnote spans before removing tags
            txt = re.sub(r'<span[^>]*class="[^"]*(?:fn|footnote|fn-mark|fussnote)[^"]*"[^>]*>.*?</span>', '', inner, flags=re.DOTALL)
            txt = re.sub(r'<sup>[^<]*</sup>', '', txt)
            txt = re.sub(r'<[^>]+>', ' ', txt)
            txt = txt.replace('&nbsp;', '\u00a0').replace('&#160;', '\u00a0')
            txt = re.sub(r'\s+', ' ', txt).strip()
            # Strip trailing footnote reference numbers (e.g. " 9", "\u00a028", ". 28")
            txt = re.sub(r'[\s\u00a0]+\d{1,4}\s*$', '', txt)
            return txt

        # Paragraphs: capoversi (absatz) + cifre/lettere (dl>dt+dd)
        paras = []
        for em in re.finditer(
            r'(<p\s[^>]*class="(?:absatz|ingress|man-template)[^"]*"[^>]*>.*?</p>'
            r'|<dl[^>]*>.*?</dl>)',
            body, re.DOTALL
        ):
            tag = em.group(0)
            if tag.startswith('<dl'):
                # Numbered/lettered sub-items: <dt>1.</dt><dd>text</dd>
                for dt_raw, dd_raw in re.findall(
                    r'<dt[^>]*>(.*?)</dt>\s*<dd[^>]*>(.*?)</dd>',
                    tag, re.DOTALL
                ):
                    num = _clean_para(dt_raw, False).strip()
                    txt = _clean_para(dd_raw, False)
                    if txt and len(txt) > 1:
                        ptype = 'ziff' if re.match(r'^\d', num) else 'litera'
                        paras.append({"n": num, "text": txt, "type": ptype})
            else:
                # Regular paragraph (absatz)
                inner = re.sub(r'^<p[^>]*>', '', tag)
                inner = re.sub(r'</p>$', '', inner)
                sup_m = re.match(r'\s*<sup>([^<]{1,6})</sup>', inner)
                n = sup_m.group(1).strip() if sup_m else None
                txt = _clean_para(inner, True)
                if txt and len(txt) > 1:
                    paras.append({"n": n, "text": txt, "type": "absatz"})

        events.append((m.start(), {
            "type": "article",
            "id": art_id,
            "heading": heading,
            "paras": paras,
        }))

    events.sort(key=lambda x: x[0])
    return title, [e[1] for e in events]


@app.route("/legge", methods=["GET"])
@limiter.limit("10 per minute; 60 per day")
def get_legge():
    sr         = request.args.get("sr",          "").strip()
    lang       = request.args.get("lang",        "it").strip().lower()
    fedlex_url = request.args.get("fedlex_url",  "").strip()

    if not sr:
        return jsonify({"errore": "Parametro 'sr' mancante"}), 400

    # Check cache
    cache_key = (sr, lang)
    cached = _law_cache.get(cache_key)
    if cached and (time.time() - cached[0]) < LAW_CACHE_TTL:
        title, elements = cached[1]
        return jsonify({"sr": sr, "lang": lang, "titolo": title,
                        "url": fedlex_url, "articoli": elements})

    # Find HTML URL via SPARQL
    html_url = None
    try:
        html_url = _sparql_html_url(sr, lang)
    except Exception:
        pass

    if not html_url:
        return jsonify({"sr": sr, "url": fedlex_url, "solo_link": True}), 206

    # Fetch HTML from public filestore
    try:
        r = requests.get(
            html_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=25
        )
        html_text = r.content.decode("utf-8", errors="replace")
        if not r.ok or len(html_text) < 5000:
            return jsonify({"sr": sr, "url": fedlex_url, "solo_link": True}), 206
    except Exception:
        return jsonify({"sr": sr, "url": fedlex_url, "solo_link": True}), 206

    # Parse
    title, elements = _parse_fedlex_html(html_text)

    # Cache
    _law_cache[cache_key] = (time.time(), (title, elements))

    return jsonify({
        "sr": sr,
        "lang": lang,
        "titolo": title,
        "url": fedlex_url or html_url,
        "articoli": elements,
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


@app.route("/html_federale", methods=["GET"])
@limiter.limit("30 per minute")
def get_html_federale():
    url = request.args.get("url", "").strip()
    if not url:
        return jsonify({"errore": "Parametro 'url' mancante"}), 400
    allowed = ("https://bger.ch", "https://www.bger.ch", "https://bger.li")
    if not any(url.startswith(p) for p in allowed):
        return jsonify({"errore": "URL non consentito"}), 400
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "it,de,fr,en;q=0.5",
        }
        resp = requests.get(url, timeout=20, headers=headers, allow_redirects=True)
        resp.raise_for_status()
        return jsonify({"html": resp.text, "url": resp.url})
    except Exception as e:
        return jsonify({"errore": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
