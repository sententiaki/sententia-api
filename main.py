import os
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
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


# ─── Utilità ────────────────────────────────────────────────────────────────

def traduci_parole_chiave(parole_chiave):
    return {
        "it": parole_chiave,
        "de": GoogleTranslator(source="it", target="de").translate(parole_chiave),
        "fr": GoogleTranslator(source="it", target="fr").translate(parole_chiave),
    }


def cerca_sentenze_google(parole_chiave):
    risultati_finali = []
    traduzioni = traduci_parole_chiave(parole_chiave)
    for lang, query in traduzioni.items():
        url = (
            f"https://www.googleapis.com/customsearch/v1"
            f"?q={query}+site:bger.ch"
            f"&key={GOOGLE_API_KEY}&cx={GOOGLE_CSE_ID}"
        )
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
        except Exception:
            continue
        for item in resp.json().get("items", [])[:5]:
            titolo = item.get("title", "")
            link   = item.get("link", "")
            m = re.search(r"(\d+[A-Z]_\d+/\d+|\d+\s+[IVXLCDM]+\s+\d+)", titolo)
            if m:
                risultati_finali.append({"codice": m.group(1), "link": link})
    return risultati_finali[:5]


def costruisci_url_bgerli(codice):
    codice = codice.strip().replace(" ", "-").replace("/", "-")
    return f"https://bger.li/{codice}"


def estrai_testo_sentenza(url):
    try:
        resp = requests.get(url, timeout=120)
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

    def call(t):
        return chiama_openai(
            system=SYSTEM_SEARCH[l],
            user=PROMPT_SEARCH[l].format(testo=t),
            max_tokens=800,
        )

    return riassumi_con_chunking(testo, call)


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
        "Descrivi i fatti rilevanti, le parti coinvolte, il procedimento seguito e le questioni giuridiche sottoposte al tribunale.\n\n"
        "**2. Articoli principali applicati**\n"
        "Elenca in modo puntuale tutti gli articoli di legge citati o applicati (indicando codice e numero, es. art. 41 CO, art. 146 CP).\n\n"
        "**3. Considerazioni del Tribunale**\n"
        "Esponi il ragionamento giuridico adottato dalla corte: interpretazione normativa, bilanciamento degli interessi, "
        "giurisprudenza richiamata e argomenti decisivi.\n\n"
        "**4. Dispositivo e implicazioni**\n"
        "Indica l'esito del giudizio (accoglimento/rigetto/rinvio), il dispositivo, le spese processuali e le implicazioni "
        "giuridiche rilevanti per la prassi.\n\n"
        "Testo della sentenza:\n{testo}"
    ),
    "de": (
        "Analysiere das folgende Urteil des Schweizer Bundesgerichts vollständig und professionell. "
        "Strukturiere deine Analyse genau wie folgt:\n\n"
        "**1. Sachverhalt**\n"
        "Beschreibe die relevanten Fakten, die beteiligten Parteien, das Verfahren und die dem Gericht vorgelegten Rechtsfragen.\n\n"
        "**2. Massgebende Rechtsartikel**\n"
        "Liste alle zitierten oder angewendeten Gesetzesartikel auf (mit Angabe des Gesetzes und Nummer, z.B. Art. 41 OR).\n\n"
        "**3. Erwägungen des Gerichts**\n"
        "Stelle die rechtliche Argumentation des Gerichts dar: Normeninterpretation, Interessenabwägung, "
        "herangezogene Rechtsprechung und entscheidende Argumente.\n\n"
        "**4. Dispositiv und Implikationen**\n"
        "Gib das Urteilsergebnis an (Gutheissung/Abweisung/Rückweisung), das Dispositiv, die Prozesskosten und "
        "die relevanten rechtlichen Implikationen für die Praxis.\n\n"
        "Urteilstext:\n{testo}"
    ),
    "fr": (
        "Analysez de manière complète et professionnelle l'arrêt du Tribunal fédéral suisse ci-dessous. "
        "Structurez votre analyse exactement comme suit:\n\n"
        "**1. Faits et procédure**\n"
        "Décrivez les faits pertinents, les parties impliquées, la procédure suivie et les questions juridiques soumises.\n\n"
        "**2. Articles principaux appliqués**\n"
        "Listez tous les articles de loi cités ou appliqués (avec indication du code et du numéro, ex. art. 41 CO).\n\n"
        "**3. Considérants du Tribunal**\n"
        "Exposez le raisonnement juridique: interprétation normative, pesée des intérêts, "
        "jurisprudence citée et arguments décisifs.\n\n"
        "**4. Dispositif et implications**\n"
        "Indiquez l'issue du jugement (admission/rejet/renvoi), le dispositif, les frais judiciaires et "
        "les implications juridiques pertinentes pour la pratique.\n\n"
        "Texte de l'arrêt:\n{testo}"
    ),
}


def sintetizza_testo_sentenza_4_punti(testo: str, lang: str = "it") -> str:
    l = lang if lang in PROMPT_SUMM else "it"

    def call(t):
        return chiama_openai(
            system=SYSTEM_SUMM[l],
            user=PROMPT_SUMM[l].format(testo=t),
            max_tokens=1800,
        )

    return riassumi_con_chunking(testo, call)


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.route("/ricerca_sentenze", methods=["GET"])
def ricerca_sentenze():
    query = request.args.get("query", "").strip()
    lang  = request.args.get("lang", "it")
    if not query:
        return jsonify({"errore": "Parametro 'query' mancante"}), 400

    sentenze  = cerca_sentenze_google(query)
    risultati = []
    for s in sentenze:
        url   = costruisci_url_bgerli(s["codice"])
        testo = estrai_testo_sentenza(url)
        if not testo or testo.startswith("ERRORE") or len(testo) < 100:
            sintesi = "Impossibile recuperare il testo della sentenza."
        else:
            sintesi = sintetizza_sentenza_10_righe(testo, lang)
        risultati.append({"titolo": s["codice"], "riassunto": sintesi, "link": url})

    return jsonify(risultati)


@app.route("/sintesi", methods=["GET"])
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
