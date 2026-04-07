#!/usr/bin/env python
"""Generate comprehensive training prompts for Zena_007.

Skills targeted:
  1. Translation between EN, RO, HU, HE, FR, ES, DE, ID, PT
  2. OCR error correction & understanding
  3. Multilingual chatbot (all 9 languages)

Output: data/zena007_prompts.jsonl  (one {"id", "prompt"} per line)
"""

import json
import random
from pathlib import Path

random.seed(42)

LANGS = {
    "en": "English",
    "ro": "Romanian",
    "hu": "Hungarian",
    "he": "Hebrew",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "id": "Indonesian",
    "pt": "Portuguese",
}

OUT = Path(__file__).resolve().parent.parent / "data" / "zena007_prompts.jsonl"

prompts: list[dict] = []


def add(pid: str, prompt: str):
    prompts.append({"id": pid, "prompt": prompt.strip()})


# ═══════════════════════════════════════════════════════════════════
# 1. TRANSLATION — systematic coverage of all language pairs
# ═══════════════════════════════════════════════════════════════════

# Sentences to translate — diverse topics, varying complexity
TRANSLATION_SENTENCES = [
    # Everyday
    "The weather is beautiful today, let's go for a walk in the park.",
    "Could you please help me find the nearest pharmacy?",
    "I would like to order two coffees and one tea, please.",
    "My flight has been delayed by three hours due to bad weather.",
    "Happy birthday! I hope all your wishes come true this year.",
    # Technical
    "The server encountered an internal error while processing your request.",
    "Please update your password to meet the new security requirements.",
    "The neural network achieved 95% accuracy on the validation dataset.",
    "Click the blue button to confirm your email address and activate your account.",
    "The database migration completed successfully with zero data loss.",
    # Cultural / Idiomatic
    "It's raining cats and dogs outside, you should take an umbrella.",
    "Rome wasn't built in a day, so be patient with your progress.",
    "The early bird catches the worm, but the second mouse gets the cheese.",
    "Actions speak louder than words when it comes to building trust.",
    "Every cloud has a silver lining if you look hard enough.",
    # Formal / Business
    "We are pleased to inform you that your application has been approved.",
    "The quarterly financial report shows a 12% increase in revenue.",
    "Please find attached the signed contract for your review and records.",
    "The board of directors has scheduled an emergency meeting for tomorrow.",
    "We regret to inform you that the position has been filled by another candidate.",
    # Medical
    "The patient presents with acute chest pain radiating to the left arm.",
    "Please take this medication twice daily with food for seven days.",
    "Your blood test results indicate slightly elevated cholesterol levels.",
    "The surgical procedure was successful and the patient is recovering well.",
    # Legal
    "The defendant pleaded not guilty to all charges brought against them.",
    "This agreement shall be governed by the laws of the respective jurisdiction.",
    # Literature / Creative
    "The autumn leaves danced in the wind like golden butterflies.",
    "She whispered softly, knowing that some secrets are better left untold.",
    # Numbers / Dates
    "The meeting is scheduled for March 15th, 2026 at 3:30 PM.",
    "The total cost including tax and shipping comes to $1,247.89.",
]

# All directed pairs through English as hub + select direct pairs
lang_codes = list(LANGS.keys())
idx = 0

# FROM English TO every other language
for sent in TRANSLATION_SENTENCES:
    for tgt in lang_codes:
        if tgt == "en":
            continue
        idx += 1
        add(f"tr-en2{tgt}-{idx:04d}",
            f"Translate the following English text to {LANGS[tgt]}:\n\n\"{sent}\"")

# FROM every other language TO English (ask teacher to first translate then back-translate)
for sent in TRANSLATION_SENTENCES[:15]:  # subset for reverse direction
    for src in lang_codes:
        if src == "en":
            continue
        idx += 1
        add(f"tr-{src}2en-{idx:04d}",
            f"First translate this English text to {LANGS[src]}, then translate the {LANGS[src]} result back to English. "
            f"Show both the {LANGS[src]} translation and the back-translation.\n\n\"{sent}\"")

# Direct pairs between non-English languages (selected pairs)
DIRECT_PAIRS = [
    ("ro", "hu"), ("ro", "fr"), ("ro", "de"), ("ro", "he"),
    ("fr", "es"), ("fr", "pt"), ("fr", "de"), ("fr", "ro"),
    ("es", "pt"), ("es", "de"), ("es", "id"),
    ("de", "hu"), ("de", "he"),
    ("hu", "he"), ("hu", "ro"),
    ("id", "pt"), ("id", "es"),
    ("he", "fr"), ("he", "de"),
    ("pt", "ro"), ("pt", "fr"),
]

for src, tgt in DIRECT_PAIRS:
    for sent in TRANSLATION_SENTENCES[:5]:
        idx += 1
        add(f"tr-{src}2{tgt}-{idx:04d}",
            f"Translate the following text from {LANGS[src]} to {LANGS[tgt]}. "
            f"First produce the {LANGS[src]} version from this English original, then translate it to {LANGS[tgt]}.\n\n"
            f"English original: \"{sent}\"")

# Multi-language translation (translate to ALL languages at once)
for sent in TRANSLATION_SENTENCES[:10]:
    idx += 1
    lang_list = ", ".join(LANGS[c] for c in lang_codes if c != "en")
    add(f"tr-multi-{idx:04d}",
        f"Translate this English text into ALL of the following languages: {lang_list}.\n"
        f"Format each translation as: [Language]: translation\n\n\"{sent}\"")

# Language detection
DETECTION_SAMPLES = [
    ("Astăzi este o zi frumoasă.", "ro"),
    ("A mai napja szép napja.", "hu"),
    ("היום יום יפה.", "he"),
    ("Aujourd'hui est une belle journée.", "fr"),
    ("Hoy es un hermoso día.", "es"),
    ("Heute ist ein schöner Tag.", "de"),
    ("Hari ini adalah hari yang indah.", "id"),
    ("Hoje é um lindo dia.", "pt"),
    ("Bună ziua, cum vă numiți?", "ro"),
    ("Jó napot, hogy hívják?", "hu"),
    ("?מה שלומך", "he"),
    ("Comment allez-vous aujourd'hui?", "fr"),
    ("¿Cómo estás hoy?", "es"),
    ("Wie geht es Ihnen heute?", "de"),
    ("Bagaimana kabar Anda hari ini?", "id"),
    ("Como você está hoje?", "pt"),
]

for text, lang in DETECTION_SAMPLES:
    idx += 1
    add(f"tr-detect-{idx:04d}",
        f"Identify the language of this text and translate it to English:\n\n\"{text}\"")

# ═══════════════════════════════════════════════════════════════════
# 2. OCR — error correction, noisy text understanding, document parsing
# ═══════════════════════════════════════════════════════════════════

OCR_NOISY_SAMPLES = [
    # Character substitution errors
    ("Th3 qu1ck br0wn f0x jumps 0ver th3 1azy d0g.",
     "Fix OCR errors in this text. The original is clean English."),
    ("lnvoice #2847 - Total Arnount: $1,247.89 - Due Oate: March 15, 2O26",
     "This invoice text has OCR scanning errors. Fix all character recognition mistakes."),
    ("Dr. Sarah Mirchell, M.D.\nCardioIogy Department\nSt. Mary's HospitaI\nPatient: John Srni1h, DOB: O3/15/198O",
     "Fix the OCR errors in this medical document header."),
    ("RENTAL AGREEMFNT\nThis agr33ment is made b3tween th3 LandIord and Tenant on this 5th day 0f ApriI, 2O26.",
     "Correct all OCR errors in this legal document text."),
    ("R3cipe: Choco1ate Cake\nlngredients:\n- 2 cups f1our\n- 1 cup su9ar\n- 3/4 cup cocoa p0wder\n- 2 tsp bak1ng powder",
     "Fix the OCR recognition errors in this recipe."),
    # Mixed language OCR
    ("Bienvenu3 à Par1s! Nous esp3rons que votre séj0ur sera agréabIe.",
     "Fix OCR errors in this French text and provide the English translation."),
    ("Bunä ziua! Bine ati veni1 la Bucure5ti. VremeA este frum0asä astäzi.",
     "Fix OCR errors in this Romanian text and provide the English translation."),
    ("Wir hei8en Sie herzl1ch wi11kommen in Ber1in. Die Stadt ist wunderschön im Frühling.",
     "Fix OCR errors in this German text and provide the English translation."),
    ("¡Bienvenid0 a Madri6! Esper4mos que disfrut3 de su cstancia cn nuestra ciudad.",
     "Fix OCR errors in this Spanish text and provide the English translation."),
    ("שלום! ברוכיW הבאיW לירושליW. אנו מקוויW שתהנו מהביקור.",
     "Fix OCR errors in this Hebrew text and provide the English translation."),
    # Table/structured OCR
    ("| Name    | Score | Grad3 |\n| A1ice   |  92   |   A   |\n| B0b     |  85   |   B   |\n| CharIie |  78   |   C+  |",
     "Fix OCR errors in this table and format it correctly."),
    ("Date: 2O26-O4-O5\nFrom: john.doe@exarnple.corn\nTo: jane.srnith@cornpany.corn\nSubject: Q1 FinanciaI Rep0rt",
     "Fix OCR errors in this email header."),
    # Handwriting OCR simulation
    ("I woold lke to schedoule a meetting for nextt Tuersday at 3 PM to discoss the projeckt.",
     "This text was produced by handwriting OCR. Fix all recognition errors."),
    ("Patiennt Name: Michaeel Johnsonn\nDiagnosis: Acutte bronchittis\nPrescripption: Amoxiciliin 500mg, 3x daiily for 7 dayss",
     "Fix the handwriting OCR errors in this medical prescription."),
    # Multi-script OCR
    ("The documeut contains text in multip1e Ianguages:\nEnglish: HeI1o WorId\nFrench: Bonj0ur le M0nde\nGerman: HaI1o WeIt",
     "Fix OCR errors in this multilingual document."),
]

for i, (noisy, instruction) in enumerate(OCR_NOISY_SAMPLES):
    idx += 1
    add(f"ocr-fix-{idx:04d}", f"{instruction}\n\n```\n{noisy}\n```")

# OCR post-processing tasks
OCR_TASKS = [
    "You are an OCR post-processor. Given the following noisy OCR output from a scanned Romanian identity card, fix all errors and extract: Full Name, ID Number, Date of Birth, Address.\n\n```\nCarte de ldentitate\nNume: P0PESCU I0N\nCNP: 18OO315O8O123\nData nasterii: 15.O3.198O\nAdresa: Str. Libertatii nr. 42, Bucure5ti\n```",  # xray: ignore[QUAL-013]
    "You are an OCR post-processor. This is OCR output from a scanned French restaurant menu. Fix errors and format properly.\n\n```\nMENU DU JOUR\nEntr3es:\n- Soupe à I'oignon gratin3e - 8,5O€\n- SaIade Nic0ise - 1O,OO€\nPIats:\n- B0euf bourguign0n - 18,5O€\n- Poulet r0ti aux herb3s - 15,OO€\nDess3rts:\n- Cr3me brûI3e - 7,5O€\n- Tarte aux pomrnes - 6,5O€\n```",  # xray: ignore[QUAL-013]
    "You are an OCR post-processor. Extract and correct the invoice data from this OCR scan:\n\n```\nlNV0lCE #2O26-OO42\nBiII To: Acme Corp0ration\n123 Business Blvd, Suite 4OO\nNew Y0rk, NY 1OOO1\n\nltem        Qty   Unit Price  T0tal\nWidget A     1O    $12.5O    $125.OO\nWidget B      5    $24.99    $124.95\nService C     1   $199.OO    $199.OO\n\nSubt0tal:                    $448.95\nTax (8.5%):                   $38.16\nT0TAL:                       $487.11\n```",  # xray: ignore[QUAL-013]
    "OCR output from a Hungarian government form. Fix all errors and translate to English:\n\n```\nMagyar K0ztarsasag\nSzem3lyi igazoIvany\nN3v: Nagy lstvan\nSzuIetesi d4tum: 1985. oktober 22.\nLakc1m: Budapest, Kossuth Lajos t3r 5.\nAIlampo1garsag: magyar\n```",  # xray: ignore[QUAL-013]
    "OCR output from a Portuguese newspaper article. Fix errors:\n\n```\nLISB0A - O govern0 anunciou h0je um n0vo pacote de medidas econ0micas para estimuIar o crescirnento. O Primeir0 Ministr0 afirmou que as rnedidas entrar4o em vig0r a partir de jurho.\n```",  # xray: ignore[QUAL-013]
    "OCR from a Hebrew document. Fix errors and translate to English:\n\n```\nמכתב רשמ1\nלכבוד: מר דוד כ0הן\nנושA: בקשה לחידוש רישי0ן\nתאר1ך: 5 באפר1ל 2026\nאנו מאשר1ם את בקשתך לחידוש הרישי0ן.\n```",
    "OCR from an Indonesian official letter. Fix errors:\n\n```\nSURAT RESM1\nKepada: Bapak Ahm4d Suryad1\nPerihaI: Permohonan lzin Usaha\nTanggaI: 5 Apr1I 2O26\nDengan h0rmat, kami memberi tahukan bahwa permohonan Anda teIah disetujui.\n```",  # xray: ignore[QUAL-013]
]

for i, task in enumerate(OCR_TASKS):
    idx += 1
    add(f"ocr-task-{idx:04d}", task)

# OCR capability questions
OCR_KNOWLEDGE = [
    "What are the most common OCR errors when scanning documents with mixed Latin and Hebrew scripts? How should a post-processor handle right-to-left text?",
    "Explain the typical OCR confusion patterns: which characters are most commonly misread (e.g., 0/O, 1/l/I, rn/m, 5/S)? Give examples in English, French, and German.",
    "How should an OCR post-processor handle tables and structured data from scanned invoices? What heuristics help reconstruct column alignment?",
    "What are best practices for OCR post-processing of handwritten text vs. printed text? How do error patterns differ?",
    "Describe a strategy for OCR post-processing of multilingual documents containing English, Romanian, Hungarian, and Hebrew text. How do you handle script switching?",
    "What techniques can improve OCR accuracy for degraded or low-quality document scans? Discuss pre-processing and post-processing approaches.",
]

for i, q in enumerate(OCR_KNOWLEDGE):
    idx += 1
    add(f"ocr-know-{idx:04d}", q)

# ═══════════════════════════════════════════════════════════════════
# 3. MULTILINGUAL CHAT — helpful assistant in all 9 languages
# ═══════════════════════════════════════════════════════════════════

CHAT_TEMPLATES = {
    "en": [
        "Explain quantum computing in simple terms that a 10-year-old could understand.",
        "I'm planning a trip to Japan. What are the top 5 things I should know before going?",
        "Write a professional email declining a job offer politely.",
        "What's the difference between machine learning and deep learning?",
        "Help me write a birthday message for my grandmother who turns 80.",
        "Explain the causes and consequences of inflation in a modern economy.",
        "What are the health benefits of Mediterranean diet? Give me a weekly meal plan.",
        "I need to prepare for a job interview as a software developer. What questions should I expect?",
        "Explain how blockchain technology works and its real-world applications beyond cryptocurrency.",
        "Write a short bedtime story about a brave little robot.",
    ],
    "ro": [
        "Explică-mi cum funcționează inteligența artificială pe înțelesul unui copil.",
        "Care sunt cele mai frumoase locuri de vizitat în România? Fă-mi un itinerar de 7 zile.",
        "Scrie o scrisoare oficială de demisie politicoasă în limba română.",
        "Care este diferența dintre un cont de economii și un depozit la termen?",
        "Ajută-mă să scriu un discurs pentru nunta prietenului meu.",
        "Explică procesul electoral din România pe scurt.",
        "Recomandă-mi 5 cărți clasice din literatura română și de ce merită citite.",
        "Cum pot să îmbunătățesc productivitatea la locul de muncă? Dă-mi 10 sfaturi practice.",
        "Scrie o rețetă tradițională românească de sarmale pas cu pas.",
        "Care sunt drepturile consumatorului în România conform legislației europene?",
    ],
    "hu": [
        "Magyarázd el egyszerűen, hogyan működik a mesterséges intelligencia.",
        "Melyek a legjobb látnivalók Budapesten? Készíts egy 3 napos útvonaltervet.",
        "Írj egy hivatalos levelet, amelyben munkát keresel.",
        "Mi a különbség az éghajlatváltozás és az időjárás között?",
        "Segíts írni egy köszönőbeszédet az érettségi ünnepségre.",
        "Mesélj a magyar konyha legfontosabb ételeiről és a receptjükről.",
        "Hogyan lehet egészségesebben élni? Adj 10 gyakorlati tanácsot.",
        "Magyarázd el a fotoszintézis folyamatát egyszerűen.",
        "Írj egy rövid mesét egy bátор kis sárkányról.",
        "Milyen készségekre van szükség egy sikeres programozónak?",
    ],
    "he": [
        "הסבר לי מהי בינה מלאכותית בצורה פשוטה.",
        "מהם 5 המקומות הכי שווים לביקור בירושלים?",
        "כתוב מכתב רשמי לבקשת חופשה מהעבודה.",
        "מה ההבדל בין חיסכון לבין השקעה?",
        "עזור לי לכתוב נאום לבר מצווה של הבן שלי.",
        "הסבר את מערכת הבריאות בישראל בקצרה.",
        "המלץ על 5 ספרים בעברית שכדאי לקרוא.",
        "איך אפשר לשפר את הזיכרון? תן 10 טיפים.",
        "כתוב מתכון לחומוס ביתי מושלם.",
        "מהם הזכויות הבסיסיות של עובד בישראל?",
    ],
    "fr": [
        "Explique-moi comment fonctionne l'intelligence artificielle simplement.",
        "Quels sont les 5 meilleurs musées à visiter à Paris?",
        "Écris une lettre de motivation pour un poste de développeur web.",
        "Quelle est la différence entre le réchauffement climatique et le changement climatique?",
        "Aide-moi à écrire un discours pour le mariage de ma sœur.",
        "Explique le système de santé français en quelques paragraphes.",
        "Recommande-moi 5 livres classiques de la littérature française.",
        "Comment améliorer son français écrit? Donne 10 conseils pratiques.",
        "Écris une recette traditionnelle de ratatouille étape par étape.",
        "Quels sont les droits fondamentaux des citoyens en France?",
    ],
    "es": [
        "Explícame cómo funciona la inteligencia artificial de forma sencilla.",
        "¿Cuáles son los 5 mejores lugares para visitar en España?",
        "Escribe una carta de presentación para solicitar un trabajo.",
        "¿Cuál es la diferencia entre ahorro e inversión?",
        "Ayúdame a escribir un discurso para la graduación de mi hija.",
        "Explica el sistema educativo en España brevemente.",
        "Recomienda 5 libros clásicos de la literatura en español.",
        "¿Cómo puedo mejorar mi español escrito? Dame 10 consejos.",
        "Escribe una receta de paella valenciana paso a paso.",
        "¿Cuáles son los derechos laborales básicos en España?",
    ],
    "de": [
        "Erkläre mir einfach, wie künstliche Intelligenz funktioniert.",
        "Was sind die 5 besten Sehenswürdigkeiten in Berlin?",
        "Schreibe ein Bewerbungsschreiben für eine Stelle als Ingenieur.",
        "Was ist der Unterschied zwischen Klima und Wetter?",
        "Hilf mir, eine Rede für die Hochzeit meines Bruders zu schreiben.",
        "Erkläre das deutsche Gesundheitssystem kurz.",
        "Empfiehl mir 5 Klassiker der deutschen Literatur.",
        "Wie kann ich mein Deutsch verbessern? Gib mir 10 Tipps.",
        "Schreibe ein Rezept für traditionelles Wiener Schnitzel.",
        "Was sind die Grundrechte der Bürger in Deutschland?",
    ],
    "id": [
        "Jelaskan bagaimana kecerdasan buatan bekerja dengan cara yang sederhana.",
        "Apa 5 tempat terbaik untuk dikunjungi di Indonesia?",
        "Tulis surat lamaran kerja untuk posisi pengembang perangkat lunak.",
        "Apa perbedaan antara tabungan dan investasi?",
        "Bantu saya menulis pidato untuk wisuda adik saya.",
        "Jelaskan sistem pendidikan di Indonesia secara singkat.",
        "Rekomendasikan 5 buku klasik sastra Indonesia.",
        "Bagaimana cara meningkatkan produktivitas kerja? Berikan 10 tips.",
        "Tulis resep nasi goreng tradisional langkah demi langkah.",
        "Apa hak-hak dasar pekerja di Indonesia?",
    ],
    "pt": [
        "Explica-me como funciona a inteligência artificial de forma simples.",
        "Quais são os 5 melhores lugares para visitar em Portugal?",
        "Escreve uma carta de motivação para uma vaga de programador.",
        "Qual é a diferença entre poupança e investimento?",
        "Ajuda-me a escrever um discurso para o casamento do meu amigo.",
        "Explica o sistema de saúde em Portugal resumidamente.",
        "Recomenda 5 clássicos da literatura portuguesa.",
        "Como posso melhorar o meu português escrito? Dá-me 10 dicas.",
        "Escreve uma receita de bacalhau à Brás passo a passo.",
        "Quais são os direitos fundamentais dos trabalhadores em Portugal?",
    ],
}

for lang_code, questions in CHAT_TEMPLATES.items():
    lang_name = LANGS[lang_code]
    for i, q in enumerate(questions):
        idx += 1
        if lang_code == "en":
            add(f"chat-{lang_code}-{idx:04d}", q)
        else:
            add(f"chat-{lang_code}-{idx:04d}",
                f"[Respond in {lang_name}]\n\n{q}")

# Cross-lingual chat (user asks in one language, answer expected in another)
CROSS_LINGUAL = [
    ("ro", "en", "Poți să-mi explici ce este machine learning? Răspunde în engleză."),
    ("en", "ro", "Explain the Romanian healthcare system. Respond in Romanian."),
    ("hu", "en", "Mi az a kvantumszámítógép? Válaszolj angolul."),
    ("en", "hu", "Describe the process of making Hungarian goulash. Respond in Hungarian."),
    ("he", "en", "מה זה שינוי אקלימי? ענה באנגלית."),
    ("en", "he", "Explain the history of Jerusalem in 3 paragraphs. Respond in Hebrew."),
    ("fr", "de", "Explique-moi la différence entre le français et l'allemand. Réponds en allemand."),
    ("es", "pt", "¿Cuáles son las diferencias entre el español y el portugués? Responde en portugués."),
    ("de", "fr", "Beschreibe die Europäische Union in einfachen Worten. Antworte auf Französisch."),
    ("id", "en", "Apa itu demokratisasi? Jawab dalam bahasa Inggris."),
    ("pt", "es", "Explica a história de Portugal. Responde em espanhol."),
    ("ro", "fr", "Prezintă pe scurt cultura română. Răspunde în franceză."),
    ("hu", "de", "Mesélj a magyar történelemről. Válaszolj németül."),
    ("fr", "ro", "Décris la cuisine française. Réponds en roumain."),
    ("es", "en", "Explica la importancia de las energías renovables. Responde en inglés."),
]

for i, (src, tgt, q) in enumerate(CROSS_LINGUAL):
    idx += 1
    add(f"chat-x-{src}2{tgt}-{idx:04d}", q)

# System-level language instructions
SYSTEM_TASKS = [
    "You are a multilingual assistant. A user writes in Romanian but wants the answer in French. Always respond in the language they request, not the language they write in.\n\nUser: Bună! Poți să-mi recomanzi un restaurant bun în Paris? Răspunde în franceză.",  # xray: ignore[QUAL-013]
    "You are a translation quality evaluator. Rate this translation from 1-10 and explain any errors:\n\nOriginal (English): 'The early bird catches the worm.'\nTranslation (Romanian): 'Pasărea matinală prinde viermele.'\n\nIs this translation natural? What would a native speaker say instead?",  # xray: ignore[QUAL-013]
    "You are a multilingual customer service agent. Handle this complaint in the same language the customer uses:\n\nCustomer: Ich habe vor zwei Wochen ein Produkt bestellt und es ist immer noch nicht angekommen. Die Sendungsverfolgung zeigt seit einer Woche keine Updates. Was ist passiert?",  # xray: ignore[QUAL-013]
    "You are a language tutor. A student learning Hungarian made these mistakes. Identify and correct each error, explaining the grammar rule:\n\n'Én vagyok menni a boltba holnap. A kutya az enyém van nagyon szép.'",  # xray: ignore[QUAL-013]
    "You are a professional translator. Translate this legal clause to ALL of these languages: Romanian, Hungarian, Hebrew, French, Spanish, German, Indonesian, Portuguese.\n\n'The parties agree that any disputes arising from this agreement shall be resolved through binding arbitration in the jurisdiction of the complainant.'",  # xray: ignore[QUAL-013]
]

for i, task in enumerate(SYSTEM_TASKS):
    idx += 1
    add(f"chat-sys-{idx:04d}", task)


# ═══════════════════════════════════════════════════════════════════
# WRITE OUTPUT
# ═══════════════════════════════════════════════════════════════════

random.shuffle(prompts)  # shuffle for better training

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", encoding="utf-8") as f:
    for p in prompts:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

print(f"Generated {len(prompts)} prompts → {OUT}")  # xray: ignore[PY-004]

# Stats
cats = {}
for p in prompts:
    cat = p["id"].split("-")[0]
    cats[cat] = cats.get(cat, 0) + 1
for cat, n in sorted(cats.items()):
    print(f"  {cat}: {n}")  # xray: ignore[PY-004]
