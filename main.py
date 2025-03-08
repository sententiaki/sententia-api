import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import requests
from bs4 import BeautifulSoup
from deep_translator import GoogleTranslator
import re
from dotenv import load_dotenv
import tiktoken

# Carica le variabili d'ambiente dal file .env
load_dotenv()

# Recupera le chiavi API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

if not (OPENAI_API_KEY and GOOGLE_API_KEY and GOOGLE_CSE_ID):
    raise ValueError("Assicurati che le chiavi API siano correttamente impostate nel file .env.")

openai.api_key = OPENAI_API_KEY

app = Flask(__name__)
CORS(app)

def traduci_parole_chiave(parole_chiave):
    traduzioni = {
        "it": parole_chiave,
        "de": GoogleTranslator(source='it', target='de').translate(parole_chiave),
        "fr": GoogleTranslator(source='it', target='fr').translate(parole_chiave)
    }
    return traduzioni

def cerca_sentenze_google(parole_chiave):
    risultati_finali = []
    traduzioni = traduci_parole_chiave(parole_chiave)
    for lang, query in traduzioni.items():
        url = f"https://www.googleapis.com/customsearch/v1?q={query}+site:bger.ch&key={GOOGLE_API_KEY}&cx={GOOGLE_CSE_ID}"
        response = requests.get(url)
        response.raise_for_status()
        risultati = response.json().get('items', [])
        for item in risultati[:5]:
            titolo = item.get('title', '')
            link = item.get('link', '')
            codice_match = re.search(r'(\d+[A-Z]_\d+/\d+|\d+\s+[IVXLCDM]+\s+\d+)', titolo)
            if codice_match:
                codice_sentenza = codice_match.group(1)
                risultati_finali.append({"codice": codice_sentenza, "link": link})
    return risultati_finali[:5]

def costruisci_url_bgerli(codice_sentenza):
    codice_sentenza = codice_sentenza.strip().replace(" ", "-").replace("/", "-")
    return f"https://bger.li/{codice_sentenza}"

def estrai_testo_sentenze(url):
    try:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        content = soup.find("div", id="content")
        if content:
            return content.get_text(separator="\n").strip()
        else:
            return "Testo della sentenza non trovato."
    except Exception as e:
        return f"Errore nell'estrazione del testo della sentenza: {e}"

# Funzioni per il chunking e il riassunto iterativo
def split_text_into_chunks(text, max_tokens=15000, model="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return [text]
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
        start = end
    return chunks

def summarize_with_chunking(text, summary_function, max_tokens=15000, model="gpt-3.5-turbo"):
    chunks = split_text_into_chunks(text, max_tokens, model)
    if len(chunks) == 1:
        return summary_function(text)
    else:
        chunk_summaries = []
        for chunk in chunks:
            summary_chunk = summary_function(chunk)
            chunk_summaries.append(summary_chunk)
        combined_summary = "\n".join(chunk_summaries)
        final_summary = summary_function(combined_summary)
        return final_summary

# Funzione per sintetizzare ogni sentenza (modalità search, ~10 righe) in base alla lingua
def sintetizza_sentenza_10_righe(testo_sentenza, lang="it"):
    def call_api(text):
        if lang == "it":
            prompt = f"""
Sei un assistente giuridico altamente specializzato. Analizza il seguente testo di sentenza e fornisci una sintesi estremamente precisa suddivisa in due sezioni:
1. Riassunto della sentenza: Riassumi in circa 10 righe il tema principale, evidenziando i fatti cruciali e le questioni legali chiave.
2. Articoli principali rilevanti: Elenca, in una lista puntata, i principali riferimenti normativi (codici e articoli) citati o discussi.
Ecco il testo della sentenza:
{text}
            """
        elif lang == "de":
            prompt = f"""
Du bist ein hochqualifizierter juristischer Assistent. Analysiere den folgenden Urteilstext und erstelle eine äußerst präzise Zusammenfassung, die in zwei Abschnitte unterteilt ist:
1. Urteilzusammenfassung: Fasse in etwa 10 Zeilen das Hauptthema zusammen, indem du die entscheidenden Fakten und wichtigsten rechtlichen Fragen hervorhebst.
2. Relevante Hauptartikel: Liste in Stichpunkten die wichtigsten gesetzlichen Verweise (Codes und Artikel) auf, die im Urteil zitiert oder diskutiert werden.
Hier ist der Urteilstext:
{text}
            """
        elif lang == "fr":
            prompt = f"""
Vous êtes un assistant juridique hautement spécialisé. Analysez le texte de la décision ci-dessous et fournissez un résumé extrêmement précis, divisé en deux sections :
1. Résumé de la décision : Résumez en environ 10 lignes le thème principal en mettant en évidence les faits cruciaux et les questions juridiques clés.
2. Articles principaux pertinents : Énumérez, sous forme de liste à puces, les principales références juridiques (codes et articles) citées ou discutées.
Voici le texte de la décision :
{text}
            """
        else:
            prompt = f"""
Sei un assistente giuridico altamente specializzato. Analizza il seguente testo di sentenza e fornisci una sintesi estremamente precisa suddivisa in due sezioni:
1. Riassunto della sentenza: Riassumi in circa 10 righe il tema principale, evidenziando i fatti cruciali e le questioni legali chiave.
2. Articoli principali rilevanti: Elenca, in una lista puntata, i principali riferimenti normativi (codici e articoli) citati o discussi.
Ecco il testo della sentenza:
{text}
            """
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Sei un assistente giuridico altamente specializzato."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.3
            )
            return response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"Errore durante la sintesi della sentenza: {e}"
    return summarize_with_chunking(testo_sentenza, call_api)

# Funzione per sintetizzare la sentenza in 4 punti (legal summarization) in base alla lingua
def sintetizza_testo_sentenza_4_punti(testo_sentenza, lang="it"):
    def call_api(text):
        if lang == "it":
            prompt = f"""
Sei un assistente giuridico esperto. Analizza attentamente il seguente testo di sentenza e fornisci una sintesi strutturata in quattro parti fondamentali, utilizzando un linguaggio giuridico. La tua risposta dovrà essere suddivisa come segue:

1. Riassunto della fattispecie: Fornisci una descrizione chiara e sintetica dei fatti rilevanti e delle questioni legali principali alla base della sentenza.
2. Articoli principali rilevanti: Elenca i principali articoli che sono stati citati o applicati, senza commenti aggiuntivi.
3. Considerazioni principali del tribunale: Riassumi le motivazioni essenziali e il ragionamento giuridico adottato dalla corte, evidenziando gli argomenti decisionali critici.
4. Conclusioni: Indica in modo sintetico l'esito finale della sentenza e le sue implicazioni giuridiche.

Ecco il testo della sentenza:
{text}
            """
        elif lang == "de":
            prompt = f"""
Du bist ein erfahrener juristischer Assistent. Analysiere den folgenden Urteilstext sorgfältig und erstelle eine strukturierte Zusammenfassung in vier wesentlichen Abschnitten, unter Verwendung einer juristischen Sprache. Deine Antwort sollte wie folgt gegliedert sein:

1. Sachverhalt Zusammenfassung: Gib eine klare und prägnante Beschreibung der relevanten Fakten und der wesentlichen rechtlichen Fragen, die dem Urteil zugrunde liegen.
2. Relevante Hauptartikel: Liste die wichtigsten gsetzliche artikeln (z.B Art. 146 Stgb) auf, die zitiert oder angewendet wurden, ohne zusätzliche Kommentare.
3. Erwägungen des Gerichts: Fasse die zentralen Motive und die juristische Argumentation des Gerichts zusammen und hebe die kritischen Entscheidungsargumente hervor.
4. Schlussfolgerungen: Gib eine kurze Zusammenfassung des endgültigen Urteils und seiner rechtlichen Implikationen.

Hier ist der Urteilstext:
{text}
            """
        elif lang == "fr":
            prompt = f"""
Vous êtes un assistant juridique extrêmement expérimenté. Analysez le texte de la décision ci-dessous et fournissez un résumé structuré en quatre parties, en utilisant un langage formel et technique :
1. Résumé des faits: Décrivez clairement les faits essentiels et les questions juridiques qui ont conduit à la décision.
2. Articles principaux pertinents: Énumérez, sous forme de liste à puces, les principales références juridiques (codes et articles) citées ou appliquées.
3. Principales considérations du tribunal: Résumez les motifs clés et le raisonnement juridique.
4. Conclusions: Indiquez brièvement le résultat final de la décision et ses implications juridiques.
Voici le texte de la décision :
{text}
            """
        else:
            prompt = f"""
Sei un assistente giuridico estremamente esperto. Analizza il seguente testo di sentenza e fornisci una sintesi strutturata in quattro parti, usando un linguaggio formale e tecnico:
1. Riassunto della fattispecie: Descrivi in modo chiaro i fatti essenziali e le questioni legali che hanno portato alla decisione.
2. Articoli principali rilevanti: Elenca in elenco puntato i principali riferimenti normativi (codici e articoli) citati o applicati.
3. Considerazioni principali del tribunale: Riassumi le motivazioni chiave e il ragionamento giuridico.
4. Conclusioni: Indica sinteticamente l'esito finale della sentenza e le sue implicazioni.
Ecco il testo della sentenza:
{text}
            """
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Sei un assistente giuridico estremamente esperto."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.3
            )
            return response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"Errore durante la sintesi della sentenza: {e}"
    return summarize_with_chunking(testo_sentenza, call_api)

# Endpoint per la ricerca delle sentenze (modalità search) – supporta il parametro 'lang'
@app.route('/ricerca_sentenze', methods=['GET'])
def ricerca_sentenze():
    query = request.args.get('query')
    lang = request.args.get('lang', 'it')
    if not query:
        return jsonify({"errore": "Parole chiave mancanti per la ricerca"}), 400

    sentenze_trovate = cerca_sentenze_google(query)
    risultati_sintetizzati = []
    for sentenza in sentenze_trovate:
        codice = sentenza["codice"]
        url_bgerli = costruisci_url_bgerli(codice)
        testo_scaricato = estrai_testo_sentenze(url_bgerli)
        if "Errore" in testo_scaricato or "Testo della sentenza non trovato" in testo_scaricato:
            sintesi = "Impossibile scaricare il contenuto della sentenza."
        else:
            sintesi = sintetizza_sentenza_10_righe(testo_scaricato, lang)
        risultati_sintetizzati.append({
            "titolo": codice,
            "riassunto": sintesi,
            "link": url_bgerli
        })
    return jsonify(risultati_sintetizzati)

# Endpoint per la sintesi di una singola sentenza (4 punti) – supporta il parametro 'lang'
@app.route('/sintesi', methods=['GET'])
def get_summary():
    codice_sentenza = request.args.get('codice')
    lang = request.args.get('lang', 'it')
    if not codice_sentenza:
        return jsonify({"errore": "Codice sentenza mancante"}), 400

    url_bgerli = costruisci_url_bgerli(codice_sentenza)
    testo_scaricato = estrai_testo_sentenze(url_bgerli)
    if "Errore" in testo_scaricato or "Testo della sentenza non trovato" in testo_scaricato:
        return jsonify({"errore": testo_scaricato}), 404

    sintesi = sintetizza_testo_sentenza_4_punti(testo_scaricato, lang)
    return jsonify({"sintesi": sintesi})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)


