# -*- coding: utf-8 -*-
"""Red-flag augmentation for the BERTic NER training set.

Why this exists
---------------
The hand-labelled seed (`train.seed.txt`) has only ~27 sentences carrying a
`FAIL` span, and 13 of them are literally "ne radi". Mining the live OLX corpus
(6k+ descriptions) showed two things the seed never taught the model:

  1. Real defects are phrased dozens of ways it had never seen — "gasi se",
     "mijenjan displej", "mrtav piksel", "za dijelove", "napuhala baterija",
     "restartuje se", "ne radi Face ID", "displej ima linije", ...

  2. The market vocabulary is dominated by look-alikes that are NOT red flags:
       * "zamjena" almost always means a *trade* offer ("na pitanja o zamjeni
         ne odgovaram", "zamjene moguće samo lično") — not a replaced part.
       * tons of positives that share the flag's keywords: "Face ID radi",
         "True Tone radi", "displej nije puknut", "ništa nije mijenjano",
         "sve radi", "nova baterija" (a *new* battery is good, not a fault).
     Without contrastive (hard-negative) examples the model can't tell
     "zamijenjen displej" (FAIL) from "moguća zamjena" (O), or "displej ima
     linije" (FAIL) from "displej nije puknut" (O).

So this file adds a batch of examples — genuine defects across every family,
plus hard negatives — authored as readable text with span annotations, and
emits well-formed CoNLL. Labels stay in the existing scheme (single `FAIL`
type); the *categorisation* of a defect into a red-flag column happens in
`core/pipeline.py`, not in the label set.

Product decision baked in: a replaced/aftermarket **screen** ("mijenjan/
zamijenjen displej/ekran/LCD") is a FAIL; a replaced **battery** ("nova
baterija", "zamijenjena baterija") is NOT — it's neutral/positive.

Run
---
    python models/description_model/augment_redflags.py   # rebuilds train.txt

Rebuilds `dataset/train.txt = dataset/train.seed.txt + generated`. Deterministic:
always reads the seed and rewrites train.txt, so it is safe to run repeatedly.
Uses only the standard library (no torch) — running it does not train anything.
Lives outside `dataset/` on purpose: everything under `dataset/` is Git-LFS
tracked (see .gitattributes), and this is source code, not data.
"""

import re
from pathlib import Path

DATA = Path(__file__).resolve().parent / "dataset"
SEED = DATA / "train.seed.txt"
OUT = DATA / "train.txt"

# Tokeniser matching the seed's conventions: runs of word chars stay together
# ("128GB", "5x", "iPhone"), every punctuation mark is its own token ("%", ".",
# "(", ")"). The model's wordpiece tokeniser re-splits these anyway; this only
# needs to agree with how the seed was tokenised so the two mix cleanly.
_TOK = re.compile(r"[^\W_]+|[^\w\s]", re.UNICODE)


def tokenize(text: str):
    return _TOK.findall(text)


# Each example: (text, [(span_substring, LABEL), ...]).
# Spans must appear in the text in the given order; the builder BIO-tags them
# and labels every other token "O". A span that doesn't match raises — so a
# typo fails the build instead of silently mislabelling.
EXAMPLES = [
    # ── A. Face ID / Touch ID not working (the single most common real defect) ──
    ("iPhone 11 64 GB u dobrom stanju, ne radi Face ID, zdravlje baterije 83 posto",
     [("iPhone 11", "MOD"), ("64 GB", "MEM"), ("dobrom stanju", "COND"),
      ("ne radi Face ID", "FAIL"), ("83 posto", "BATT")]),
    ("Prodajem iPhone X, Face ID ne radi jer je mijenjan ekran, baterija 79 %",
     [("iPhone X", "MOD"), ("Face ID ne radi", "FAIL"), ("mijenjan ekran", "FAIL"),
      ("79 %", "BATT")]),
    ("iPhone 12 Pro 128 GB, 100 % baterija, Face ID ne radi, ostalo sve uredno",
     [("iPhone 12 Pro", "MOD"), ("128 GB", "MEM"), ("100 %", "BATT"),
      ("Face ID ne radi", "FAIL")]),
    ("iPhone 11 Pro potpuno ispravan, face id ne radi, zbog toga niža cijena",
     [("iPhone 11 Pro", "MOD"), ("face id ne radi", "FAIL")]),
    ("Telefon je u top stanju, samo mu ne radi Face ID, 100 % zdravlje baterije",
     [("top stanju", "COND"), ("ne radi Face ID", "FAIL"), ("100 %", "BATT")]),
    ("iPhone 13 128 GB, ne radi Face ID, True Tone radi, sitni tragovi korištenja",
     [("iPhone 13", "MOD"), ("128 GB", "MEM"), ("ne radi Face ID", "FAIL")]),
    ("iPhone SE 2020 ispravan telefon, ne radi home dugme",
     [("iPhone SE 2020", "MOD"), ("ne radi home dugme", "FAIL")]),
    ("iPhone X 64 GB, Face ID ne radi jer je mijenjan displej, 83 % baterija",
     [("iPhone X", "MOD"), ("64 GB", "MEM"), ("Face ID ne radi", "FAIL"),
      ("mijenjan displej", "FAIL"), ("83 %", "BATT")]),
    ("Ajfon 12 Pro, 100 % baterija, Face ID ne radi, telefon u odličnom stanju",
     [("Ajfon 12 Pro", "MOD"), ("100 %", "BATT"), ("Face ID ne radi", "FAIL"),
      ("odličnom stanju", "COND")]),
    ("iPhone 8 64 GB, Touch ID ne radi, baterija 88 posto",
     [("iPhone 8", "MOD"), ("64 GB", "MEM"), ("Touch ID ne radi", "FAIL"),
      ("88 posto", "BATT")]),
    ("iPhone 14 Pro Max, mijenjan displej, ne radi face recognition funkcija, ostalo ok",
     [("iPhone 14 Pro Max", "MOD"), ("mijenjan displej", "FAIL"),
      ("ne radi face recognition funkcija", "FAIL")]),
    ("iPhone 7 128 GB, ne radi otisak prsta, baterija 90 %",
     [("iPhone 7", "MOD"), ("128 GB", "MEM"), ("ne radi otisak prsta", "FAIL"),
      ("90 %", "BATT")]),
    ("Mobitel extra, sve radi osim Face ID, prednja kamera radi, displej originalan",
     [("osim Face ID", "FAIL")]),
    ("iPhone 11 Pro Max, ne radi Face ID, zbog toga niža cijena, 100 % zdravlje baterije",
     [("iPhone 11 Pro Max", "MOD"), ("ne radi Face ID", "FAIL"), ("100 %", "BATT")]),
    ("iPhone 12, Face ID ne radi, može provjera, baterija 89 %",
     [("iPhone 12", "MOD"), ("Face ID ne radi", "FAIL"), ("89 %", "BATT")]),
    ("iPhone XR, ne radi Face ID i mijenjan ekran, baterija 82 %",
     [("iPhone XR", "MOD"), ("ne radi Face ID", "FAIL"), ("mijenjan ekran", "FAIL"),
      ("82 %", "BATT")]),

    # ── B. Replaced / aftermarket screen (FAIL) — battery replacement stays O ──
    ("iPhone 13 128 GB, mijenjan ekran, stavljen originalni, baterija 95 %",
     [("iPhone 13", "MOD"), ("128 GB", "MEM"), ("mijenjan ekran", "FAIL"),
      ("95 %", "BATT")]),
    ("Na mobitelu zamijenjen displej, ugrađen originalni, iPhone 12 mini 64 GB, 75 % baterija",
     [("zamijenjen displej", "FAIL"), ("iPhone 12 mini", "MOD"), ("64 GB", "MEM"),
      ("75 %", "BATT")]),
    ("iPhone 15 Pro Max 256 GB, zamijenjen displej, ugrađen zamjenski display, baterija 81 %",
     [("iPhone 15 Pro Max", "MOD"), ("256 GB", "MEM"), ("zamijenjen displej", "FAIL"),
      ("zamjenski display", "FAIL"), ("81 %", "BATT")]),
    ("iPhone 11 83 % zdravlje, mijenjan displej pa ima tačkica na njemu, ne smeta pri radu",
     [("iPhone 11", "MOD"), ("83 %", "BATT"), ("mijenjan displej", "FAIL")]),
    ("APPLE iPhone 12 Pro Max 128 GB, zamijenjen displej, Face ID i True Tone rade, 79 % baterija",
     [("APPLE", "BRAND"), ("iPhone 12 Pro Max", "MOD"), ("128 GB", "MEM"),
      ("zamijenjen displej", "FAIL"), ("79 %", "BATT")]),
    ("iPhone 16 Pro Max 256 GB, na telefonu mijenjan displej stavljen original, baterija 93 %",
     [("iPhone 16 Pro Max", "MOD"), ("256 GB", "MEM"), ("mijenjan displej", "FAIL"),
      ("93 %", "BATT")]),
    ("iPhone X, zamjenjen display oled stavljen, ne radi Face ID, ima originalni proximity senzor",
     [("iPhone X", "MOD"), ("zamjenjen display", "FAIL"), ("ne radi Face ID", "FAIL")]),
    ("iPhone 11 128 GB, nije original displej, aftermarket panel, sve ostalo radi, 70 % baterija",
     [("iPhone 11", "MOD"), ("128 GB", "MEM"), ("nije original displej", "FAIL"),
      ("aftermarket panel", "FAIL"), ("70 %", "BATT")]),
    ("iPhone 14, mijenjan LCD, boja ekrana malo blijeđa, baterija 88 %",
     [("iPhone 14", "MOD"), ("mijenjan LCD", "FAIL"), ("88 %", "BATT")]),
    ("iPhone XR 64 GB, displej je mijenjan, True Tone ne radi nakon zamjene, 84 %",
     [("iPhone XR", "MOD"), ("64 GB", "MEM"), ("displej je mijenjan", "FAIL"),
      ("True Tone ne radi", "FAIL"), ("84 %", "BATT")]),
    ("iPhone 13 Pro Max, zamijenjen ekran, nije original panel, True Tone ne radi",
     [("iPhone 13 Pro Max", "MOD"), ("zamijenjen ekran", "FAIL"),
      ("nije original panel", "FAIL"), ("True Tone ne radi", "FAIL")]),
    ("iPhone 12, displej mijenjan servisiran, boja malo odudara, ostalo radi",
     [("iPhone 12", "MOD"), ("displej mijenjan", "FAIL")]),

    # ── C. Powers off / restarts / boot loop (FAIL) ──
    ("iPhone 14 6 128 GB, ima problem sa matičnom pločom jer se često gasi i pali",
     [("iPhone 14", "MOD"), ("128 GB", "MEM"), ("problem sa matičnom pločom", "FAIL"),
      ("gasi i pali", "FAIL")]),
    ("iPhone 5s ima problem, logo blinka 10-15 sekundi, ugasi pa upali dok je na punjaču",
     [("iPhone 5s", "MOD"), ("logo blinka", "FAIL"), ("ugasi pa upali", "FAIL")]),
    ("iPhone 12 mini, restartuje se sam, prednja kamera ne radi, baterija 92 %",
     [("iPhone 12 mini", "MOD"), ("restartuje se sam", "FAIL"),
      ("prednja kamera ne radi", "FAIL"), ("92 %", "BATT")]),
    ("iPhone 8 Plus, sam se gasi na 20 %, baterija napuhala, za dijelove ili popravku",
     [("iPhone 8 Plus", "MOD"), ("sam se gasi", "FAIL"), ("baterija napuhala", "FAIL"),
      ("za dijelove", "FAIL")]),
    ("iPhone 6s, ne pali se, vjerujem da je pokvaren, prodajem za dijelove",
     [("iPhone 6s", "MOD"), ("ne pali se", "FAIL"), ("pokvaren", "FAIL"),
      ("za dijelove", "FAIL")]),
    ("iPhone 11, zna se ugasiti iako je baterija puna, povremeno se restartuje",
     [("iPhone 11", "MOD"), ("zna se ugasiti", "FAIL"), ("restartuje", "FAIL")]),
    ("iPhone X, boot loop, vrti logo i gasi se, za dijelove",
     [("iPhone X", "MOD"), ("boot loop", "FAIL"), ("vrti logo i gasi se", "FAIL"),
      ("za dijelove", "FAIL")]),
    ("iPhone 13 Pro, telefon se nakon nekog vremena restartuje, ekran ima pukotinu",
     [("iPhone 13 Pro", "MOD"), ("restartuje", "FAIL"), ("ekran ima pukotinu", "FAIL")]),
    ("iPhone SE, neće da upali, ne reaguje na punjač, mislim da je matična",
     [("iPhone SE", "MOD"), ("neće da upali", "FAIL"), ("ne reaguje na punjač", "FAIL")]),
    ("iPhone 11, povremeno se sam gasi i restartuje, baterija 84 %",
     [("iPhone 11", "MOD"), ("sam gasi i restartuje", "FAIL"), ("84 %", "BATT")]),
    ("iPhone 8, pali i gasi se u krug, ne mogu da ga upalim, za dijelove",
     [("iPhone 8", "MOD"), ("pali i gasi se", "FAIL"), ("ne mogu da ga upalim", "FAIL"),
      ("za dijelove", "FAIL")]),

    # ── D. For parts / broken / fault (FAIL) ──
    ("iPhone 15 Pro, telefon za dijelove, ekran se odvojio ali stoji na fletovima",
     [("iPhone 15 Pro", "MOD"), ("za dijelove", "FAIL"), ("ekran se odvojio", "FAIL")]),
    ("iPhone 6, imao problem sa displejom, nekad ne registruje dodir, vjerujem da je pokvaren",
     [("iPhone 6", "MOD"), ("problem sa displejom", "FAIL"),
      ("ne registruje dodir", "FAIL"), ("pokvaren", "FAIL")]),
    ("iPhone 11 za dijelove, ne znam šta je s telefonom, ne pali",
     [("iPhone 11", "MOD"), ("za dijelove", "FAIL"), ("ne pali", "FAIL")]),
    ("iPhone X razbijen, pogodan za dijelove",
     [("iPhone X", "MOD"), ("razbijen", "FAIL"), ("za dijelove", "FAIL")]),
    ("iPhone 7 Plus, MDM bypass, radi ali za dijelove",
     [("iPhone 7 Plus", "MOD"), ("MDM bypass", "FAIL"), ("za dijelove", "FAIL")]),
    ("iPhone 12, neispravan, u kvaru, javlja grešku pri paljenju",
     [("iPhone 12", "MOD"), ("neispravan", "FAIL"), ("u kvaru", "FAIL"),
      ("javlja grešku", "FAIL")]),
    ("iPhone 12 mini, ima grešku na kameri, ne može da fokusira, slika bude zamagljena",
     [("iPhone 12 mini", "MOD"), ("ima grešku na kameri", "FAIL"),
      ("ne može da fokusira", "FAIL"), ("slika bude zamagljena", "FAIL")]),

    # ── E. Screen defects that aren't cracks (FAIL) ──
    ("iphone 15 Pro 128 GB 82 %, ima dva mrtva piksela na ekranu, ne utiče na rad",
     [("iphone 15 Pro", "MOD"), ("128 GB", "MEM"), ("82 %", "BATT"),
      ("dva mrtva piksela", "FAIL")]),
    ("iPhone 12 mini, displej ima linije, normalno se koristi, baterija 85 %",
     [("iPhone 12 mini", "MOD"), ("displej ima linije", "FAIL"), ("85 %", "BATT")]),
    ("iPhone X 64 GB, oštećeno zadnje staklo i ima mrtav piksel, zdravlje baterije 78 %",
     [("iPhone X", "MOD"), ("64 GB", "MEM"), ("oštećeno zadnje staklo", "FAIL"),
      ("mrtav piksel", "FAIL"), ("78 %", "BATT")]),
    ("iPhone 13, na ekranu se pojavila žuta mrlja, burn-in vidljiv na svijetloj pozadini",
     [("iPhone 13", "MOD"), ("žuta mrlja", "FAIL"), ("burn-in", "FAIL")]),
    ("iPhone 11 Pro, glavna kamera ima flekice, ušla je prašina, slika mutna",
     [("iPhone 11 Pro", "MOD"), ("kamera ima flekice", "FAIL"), ("ušla je prašina", "FAIL"),
      ("slika mutna", "FAIL")]),
    ("iPhone XS, displej ima zapečenu sliku, gorenje ekrana u statusnoj traci",
     [("iPhone XS", "MOD"), ("zapečenu sliku", "FAIL"), ("gorenje ekrana", "FAIL")]),
    ("iPhone 13 Pro, sitna linija na ekranu nakon pada, ostalo radi",
     [("iPhone 13 Pro", "MOD"), ("linija na ekranu", "FAIL")]),
    ("iPhone 12, mrlja na displeju u donjem dijelu, ne smeta previše",
     [("iPhone 12", "MOD"), ("mrlja na displeju", "FAIL")]),
    ("iPhone 12 Pro, mrtav piksel na sredini ekrana, ne smeta pri gledanju, baterija 91 %",
     [("iPhone 12 Pro", "MOD"), ("mrtav piksel", "FAIL"), ("91 %", "BATT")]),
    ("iPhone 11, ekran ima zelenu liniju sa strane nakon pada",
     [("iPhone 11", "MOD"), ("zelenu liniju", "FAIL")]),

    # ── F. Cracks, front vs back (FAIL) ──
    ("iPhone 11 Pro Max 256 GB, malo napuklo staklo, ne smeta u radu",
     [("iPhone 11 Pro Max", "MOD"), ("256 GB", "MEM"), ("napuklo staklo", "FAIL")]),
    ("iPhone XS, oštećen zadnji dio telefona, napuklo staklo pozadi, baterija 78 %",
     [("iPhone XS", "MOD"), ("napuklo staklo pozadi", "FAIL"), ("78 %", "BATT")]),
    ("iPhone 12 Pro razbijen naprijed i nazad, true tone radi, face id radi, imam kutiju",
     [("iPhone 12 Pro", "MOD"), ("razbijen naprijed i nazad", "FAIL"), ("kutiju", "BOX")]),
    ("iPhone 14, lijevi gornji ćošak napukao ekran, ne vidi se puno, ne smeta pri radu",
     [("iPhone 14", "MOD"), ("napukao ekran", "FAIL")]),
    ("iphone 15 Pro, puknut displej i zadnje staklo, ploča ispravna, baterija 85 %",
     [("iphone 15 Pro", "MOD"), ("puknut displej i zadnje staklo", "FAIL"), ("85 %", "BATT")]),
    ("iPhone 13, oštetilo se zadnje staklo, ne radi Face ID, 70 % baterija",
     [("iPhone 13", "MOD"), ("oštetilo se zadnje staklo", "FAIL"), ("ne radi Face ID", "FAIL"),
      ("70 %", "BATT")]),
    ("Samsung Galaxy S21 128 GB, napukao ekran u uglu, ostalo radi",
     [("Samsung", "BRAND"), ("Galaxy S21", "MOD"), ("128 GB", "MEM"), ("napukao ekran", "FAIL")]),
    ("iPhone 11, razbijeno zadnje staklo, prednji ekran čist, baterija 88 %",
     [("iPhone 11", "MOD"), ("razbijeno zadnje staklo", "FAIL"), ("88 %", "BATT")]),

    # ── G. Camera (FAIL) ──
    ("iPhone 13, ne radi 0.5x kamera, ostalo sve radi, zdravlje baterije 68 %",
     [("iPhone 13", "MOD"), ("ne radi 0.5x kamera", "FAIL"), ("68 %", "BATT")]),
    ("iPhone 12, prednja kamera prikazuje mutno, potrebno očistiti u servisu",
     [("iPhone 12", "MOD"), ("prednja kamera prikazuje mutno", "FAIL")]),
    ("iPhone 11, zadnja kamera ne radi, prednja ok, baterija 90 %",
     [("iPhone 11", "MOD"), ("zadnja kamera ne radi", "FAIL"), ("90 %", "BATT")]),
    ("iPhone XS, oštećeno staklo od kamere, kamera radi, baterija 79 %",
     [("iPhone XS", "MOD"), ("oštećeno staklo od kamere", "FAIL"), ("79 %", "BATT")]),

    # ── H. Battery degraded / drains (FAIL) — note: "nova baterija" is NOT here ──
    ("iPhone 13 Pro, napuhala se baterija, kratko traje, zna se ugasiti",
     [("iPhone 13 Pro", "MOD"), ("napuhala se baterija", "FAIL"), ("kratko traje", "FAIL"),
      ("zna se ugasiti", "FAIL")]),
    ("iPhone X, baterija se brzo prazni, ostalo radi",
     [("iPhone X", "MOD"), ("baterija se brzo prazni", "FAIL")]),
    ("iPhone 11, slaba baterija drži jako kratko, treba mijenjati",
     [("iPhone 11", "MOD"), ("slaba baterija", "FAIL"), ("drži jako kratko", "FAIL")]),

    # ── I. Speaker / mic / vibration / charging / touch / sensors / buttons (FAIL) ──
    ("iPhone 12, ne radi zvučnik za razgovor, u slušalicama se ne čuje",
     [("iPhone 12", "MOD"), ("ne radi zvučnik", "FAIL")]),
    ("iPhone 11, zvučnik za uho zamijenjen, ne radi kako treba, baterija 76 %",
     [("iPhone 11", "MOD"), ("zvučnik za uho zamijenjen", "FAIL"), ("ne radi", "FAIL"),
      ("76 %", "BATT")]),
    ("iPhone X, mikrofon ne radi tokom poziva, sagovornik me ne čuje",
     [("iPhone X", "MOD"), ("mikrofon ne radi", "FAIL")]),
    ("iPhone SE, ne radi vibracija, ostalo ispravno",
     [("iPhone SE", "MOD"), ("ne radi vibracija", "FAIL")]),
    ("iPhone 13, ne puni se, mijenjan konektor za punjenje ne pomaže",
     [("iPhone 13", "MOD"), ("ne puni se", "FAIL"), ("mijenjan konektor za punjenje", "FAIL")]),
    ("iPhone 12 Pro, touch ne reagira u gornjem dijelu ekrana",
     [("iPhone 12 Pro", "MOD"), ("touch ne reagira", "FAIL")]),
    ("iPhone 11, ne radi senzor blizine, ekran ostaje upaljen tokom poziva",
     [("iPhone 11", "MOD"), ("ne radi senzor blizine", "FAIL")]),
    ("iPhone 8, ne radi tipka za glasnoću, ostalo radi",
     [("iPhone 8", "MOD"), ("ne radi tipka za glasnoću", "FAIL")]),

    # ── J. HARD NEGATIVES — "zamjena" = trade offer, never a red flag ──
    ("iPhone 13 Pro 256 GB, cijena fiksna, na pitanja o zamjeni ne odgovaram, baterija 90 %",
     [("iPhone 13 Pro", "MOD"), ("256 GB", "MEM"), ("90 %", "BATT")]),
    ("iPhone 12 128 GB, moguća zamjena za noviji model uz doplatu, sve ispravno",
     [("iPhone 12", "MOD"), ("128 GB", "MEM")]),
    ("iPhone 14 Pro, zamjena isključivo za iPhone, u zamjeni skuplji, baterija 92 %",
     [("iPhone 14 Pro", "MOD"), ("92 %", "BATT")]),
    ("iPhone 15 Pro Max 256 GB, zamjene moguće samo lično, telefon bez ikakvih mana",
     [("iPhone 15 Pro Max", "MOD"), ("256 GB", "MEM")]),
    ("iPhone 11, prednost keš, zamjena samo za telefon, sve original",
     [("iPhone 11", "MOD")]),
    ("iPhone 16 Pro, prodaja otkup zamjena, na rate, kao nov",
     [("iPhone 16 Pro", "MOD"), ("kao nov", "COND")]),
    ("iPhone X, nudim na zamjenu uz moju nadoplatu, ispravan, baterija 84 %",
     [("iPhone X", "MOD"), ("84 %", "BATT")]),
    ("iPhone 13 mini, cijena fiksna, zamjene ne dolaze u obzir, baterija 88 %",
     [("iPhone 13 mini", "MOD"), ("88 %", "BATT")]),
    ("iPhone 12 Pro, isključivo prodaja, bez zamjene, cijena fiksna, baterija 90 %",
     [("iPhone 12 Pro", "MOD"), ("90 %", "BATT")]),
    ("iPhone 13, može zamjena za Samsung uz doplatu, telefon ispravan",
     [("iPhone 13", "MOD")]),

    # ── K. HARD NEGATIVES — positives that share the flag's keywords ──
    ("iPhone 11 64 GB, Face ID radi besprijekorno, True Tone radi, ništa nije mijenjano",
     [("iPhone 11", "MOD"), ("64 GB", "MEM")]),
    ("iPhone 12 Pro, displej originalan, nikad otvaran, sve funkcije rade bez greške",
     [("iPhone 12 Pro", "MOD")]),
    ("iPhone 13, displej nije puknut nigdje, bez ijedne packe, kao nov",
     [("iPhone 13", "MOD"), ("kao nov", "COND")]),
    ("iPhone XS 64 GB, kao novo, ubačena nova baterija, zvučnici extra, kamera bez greške",
     [("iPhone XS", "MOD"), ("64 GB", "MEM"), ("kao novo", "COND")]),
    ("iPhone 11, sve radi, Face ID ispravan, bez oštećenja, prvi vlasnik",
     [("iPhone 11", "MOD")]),
    ("iPhone 14 Pro Max, telefon nikad otvaran, ništa mijenjano niti servisirano, baterija 95 %",
     [("iPhone 14 Pro Max", "MOD"), ("95 %", "BATT")]),
    ("iPhone 12, kamere i sve ostalo 100 % ispravno, može na provjeru, bez skrivenih mana",
     [("iPhone 12", "MOD")]),
    ("iPhone X, baterija nova zamijenjena u Apple storu, sve radi bez problema",
     [("iPhone X", "MOD")]),
    ("iPhone 13 Pro, True Tone radi, Face ID radi, displej je originalan, bez ogrebotina",
     [("iPhone 13 Pro", "MOD")]),
    ("iPhone 11 Pro, malo napuklo zaštitno staklo, telefon ispod netaknut, sve radi",
     [("iPhone 11 Pro", "MOD")]),
    ("iPhone 12 mini, ne koči, ne baguje, sve funkcije rade, baterija 87 %",
     [("iPhone 12 mini", "MOD"), ("87 %", "BATT")]),
    ("iPhone 15 telefon kao nov bez oštećenja ili ogrebotina, full pakovanje originalno",
     [("iPhone 15", "MOD"), ("kao nov", "COND"), ("full pakovanje", "BOX")]),
    ("iPhone 13, na telefonu ništa nije mijenjano, displej originalan Apple, Face ID i True Tone rade",
     [("iPhone 13", "MOD")]),
    ("iPhone 12 Pro Max, telefon u perfektnom stanju bez mrlje, bez packe, prvi vlasnik",
     [("iPhone 12 Pro Max", "MOD"), ("perfektnom stanju", "COND")]),
    ("iPhone 11, displej nije mijenjan, original Apple panel, Face ID radi",
     [("iPhone 11", "MOD")]),
    ("iPhone X, ništa nije mijenjano, nije razbijen, sve ispravno, može provjera",
     [("iPhone X", "MOD")]),

    # ── L. Mixed multi-flag listings (mirroring real descriptions) ──
    ("Iphone 16 Pro 128 GB, mobitel ima problem, restartuje se i prednja kamera ne radi, zadnje staklo ima liniju, baterija 92 %, displej nije puknut",
     [("Iphone 16 Pro", "MOD"), ("128 GB", "MEM"), ("restartuje se", "FAIL"),
      ("prednja kamera ne radi", "FAIL"), ("zadnje staklo ima liniju", "FAIL"),
      ("92 %", "BATT")]),
    ("iPhone 12 mini 64 GB, zamijenjen displej stavljen originalni, Face ID ispravan, True Tone prisutan, zdravlje baterije 75 %",
     [("iPhone 12 mini", "MOD"), ("64 GB", "MEM"), ("zamijenjen displej", "FAIL"),
      ("75 %", "BATT")]),
    ("iPhone 11 128 GB, Face ID nije u funkciji, zamijenjen displej ugrađen originalni, ostalo uredno, baterija 70 %",
     [("iPhone 11", "MOD"), ("128 GB", "MEM"), ("Face ID nije u funkciji", "FAIL"),
      ("zamijenjen displej", "FAIL"), ("70 %", "BATT")]),
    ("iPhone 13 Pro 128 GB razbijen, ne radi 0.5x kamera, normalna kamera radi, Face ID ne radi, baterija 68 %",
     [("iPhone 13 Pro", "MOD"), ("128 GB", "MEM"), ("razbijen", "FAIL"),
      ("ne radi 0.5x kamera", "FAIL"), ("Face ID ne radi", "FAIL"), ("68 %", "BATT")]),
    ("iPhone 14 Plus 128 GB, malo oštećeno staklo od pozadi, baterija 76 posto, zvučnik za uho zamijenjen, ne radi Face ID",
     [("iPhone 14 Plus", "MOD"), ("128 GB", "MEM"), ("oštećeno staklo od pozadi", "FAIL"),
      ("76 posto", "BATT"), ("zvučnik za uho zamijenjen", "FAIL"), ("ne radi Face ID", "FAIL")]),
    ("iPhone X 256 GB, jedina mana što se malo više grije i ne radi Face ID, ostalo sve radi, baterija 100 %",
     [("iPhone X", "MOD"), ("256 GB", "MEM"), ("grije", "FAIL"), ("ne radi Face ID", "FAIL"),
      ("100 %", "BATT")]),
    ("iPhone 16 Pro Max 256 GB, na telefonu mijenjan displej stavljen original, sve radi uredno, baterija 93 %",
     [("iPhone 16 Pro Max", "MOD"), ("256 GB", "MEM"), ("mijenjan displej", "FAIL"),
      ("93 %", "BATT")]),
    ("iphone 15 Pro 128 GB, mijenjan displej ne baš najbolje, ne radi face recognition, za ove pare valja",
     [("iphone 15 Pro", "MOD"), ("128 GB", "MEM"), ("mijenjan displej", "FAIL"),
      ("ne radi face recognition", "FAIL")]),

    # ── M. Back-glass & display damage + dead pixels — real phrasings the model
    #    under-extracted in production: it tagged "staklo" but dropped the damage
    #    verb ("oštećeno"), missed "razbijen"/"napuklo" outright, and left a
    #    disclosed "jedan piksel" unrouted. Both diacritic and plain-ASCII
    #    spellings (as sellers actually type them) are included on purpose. ──
    ("iPhone 8 Plus u dobrom stanju, nazad razbijen, ekran kao nov, sve radi normalno",
     [("iPhone 8 Plus", "MOD"), ("dobrom stanju", "COND"), ("nazad razbijen", "FAIL")]),
    ("iPhone 8 64 GB, oštećeno zadnje staklo, ne smeta pri radu, zdravlje baterije 82 %",
     [("iPhone 8", "MOD"), ("64 GB", "MEM"), ("oštećeno zadnje staklo", "FAIL"), ("82 %", "BATT")]),
    ("iPhone 11, zadnje staklo razbijeno, prednji ekran čist, sve funkcije rade",
     [("iPhone 11", "MOD"), ("zadnje staklo razbijeno", "FAIL")]),
    ("iPhone 12 Pro, ima ostecenje na zadnjem staklu, ekran bez greske, baterija 88 %",
     [("iPhone 12 Pro", "MOD"), ("ostecenje na zadnjem staklu", "FAIL"), ("88 %", "BATT")]),
    ("iPhone XR, napuklo staklo na poleđini, prednja strana netaknuta",
     [("iPhone XR", "MOD"), ("napuklo staklo na poleđini", "FAIL")]),
    ("iPhone 13, zadnja strana napukla, displej ispravan, baterija 79 %",
     [("iPhone 13", "MOD"), ("zadnja strana napukla", "FAIL"), ("79 %", "BATT")]),
    ("iPhone 14 Pro, staklo pozadi ispucano, radi besprijekorno, baterija 90 %",
     [("iPhone 14 Pro", "MOD"), ("staklo pozadi ispucano", "FAIL"), ("90 %", "BATT")]),
    ("iPhone X, zadnje staklo puklo, ekran radi normalno, za svakodnevnu upotrebu",
     [("iPhone X", "MOD"), ("zadnje staklo puklo", "FAIL")]),
    ("iPhone 12 mini, poleđina napukla u uglu, prednja strana čista, baterija 85 %",
     [("iPhone 12 mini", "MOD"), ("poleđina napukla", "FAIL"), ("85 %", "BATT")]),
    ("iPhone 13 128 GB, lagano oštećenje u ćošku displeja, ostalo uredno, baterija 78 %",
     [("iPhone 13", "MOD"), ("128 GB", "MEM"), ("oštećenje u ćošku displeja", "FAIL"), ("78 %", "BATT")]),
    ("iPhone 12, ostecenje u cosku ekrana, ne smeta pri radu",
     [("iPhone 12", "MOD"), ("ostecenje u cosku ekrana", "FAIL")]),
    ("iPhone 11 Pro, displej oštećen u gornjem uglu, funkcije rade normalno",
     [("iPhone 11 Pro", "MOD"), ("displej oštećen u gornjem uglu", "FAIL")]),
    ("iPhone XS, staklo ispucano i ima napuknuće, zadnja kamera zna da vibrira pa stane",
     [("iPhone XS", "MOD"), ("staklo ispucano", "FAIL"), ("napuknuće", "FAIL"),
      ("kamera zna da vibrira", "FAIL")]),
    ("iPhone 14 Pro 128 GB, ima jedan piksel na ekranu koji ne smeta pri radu, baterija 77 %",
     [("iPhone 14 Pro", "MOD"), ("128 GB", "MEM"), ("jedan piksel na ekranu", "FAIL"), ("77 %", "BATT")]),
    ("iPhone 12, jedan piksel ne radi na displeju, jedva se primijeti",
     [("iPhone 12", "MOD"), ("jedan piksel ne radi", "FAIL")]),
    ("iPhone 11, zapeo piksel na ekranu, na tamnoj pozadini se vidi",
     [("iPhone 11", "MOD"), ("zapeo piksel", "FAIL")]),
    ("iPhone 13 Pro, ima jedan crveni piksel, ne smeta pri gledanju",
     [("iPhone 13 Pro", "MOD"), ("jedan crveni piksel", "FAIL")]),
    # hard negatives: "piksel"/"staklo" mentions that are NOT defects
    ("iPhone 13, kamera 12 megapiksela, ekran bez ijednog mrtvog piksela, kao nov",
     [("iPhone 13", "MOD"), ("kao nov", "COND")]),
    ("iPhone 12, zaštitno staklo zalijepljeno, zadnje staklo bez oštećenja, sve radi",
     [("iPhone 12", "MOD")]),
]


def _find_span(tokens, span_tokens, start):
    """Index of the first contiguous occurrence of span_tokens at or after start."""
    n, m = len(tokens), len(span_tokens)
    for i in range(start, n - m + 1):
        if tokens[i:i + m] == span_tokens:
            return i
    return -1


def build_block(text, spans):
    tokens = tokenize(text)
    tags = ["O"] * len(tokens)
    cursor = 0
    for span, label in spans:
        span_tokens = tokenize(span)
        idx = _find_span(tokens, span_tokens, cursor)
        if idx < 0:
            raise ValueError(
                f"span {span!r} ({span_tokens}) not found from token {cursor} in:\n"
                f"  {text!r}\n  tokens={tokens}"
            )
        tags[idx] = f"B-{label}"
        for j in range(idx + 1, idx + len(span_tokens)):
            tags[j] = f"I-{label}"
        cursor = idx + len(span_tokens)
    return "\n".join(f"{tok} {tag}" for tok, tag in zip(tokens, tags))


def main():
    blocks = [build_block(text, spans) for text, spans in EXAMPLES]

    seed_text = SEED.read_text(encoding="utf-8").rstrip("\n")
    generated = "\n\n".join(blocks)
    OUT.write_text(seed_text + "\n\n" + generated + "\n", encoding="utf-8")

    # Report added FAIL coverage so a bad edit is obvious.
    fail_spans = sum(1 for _, spans in EXAMPLES for _, lab in spans if lab == "FAIL")
    neg_sents = sum(1 for _, spans in EXAMPLES if not any(l == "FAIL" for _, l in spans))
    print(f"examples added:        {len(EXAMPLES)}")
    print(f"  with FAIL spans:     {len(EXAMPLES) - neg_sents}")
    print(f"  hard negatives:      {neg_sents}")
    print(f"FAIL spans added:      {fail_spans}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
