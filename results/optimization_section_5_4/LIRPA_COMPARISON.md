# Poređenje sa auto_LiRPA — baseline koji Pasado rad nema

## Zašto ovo poređenje

Pasado (OOPSLA 2023) u Section 5.4 poredi lokalne Lipšicove konstante samo
protiv **interval AD** i **zonotope AD** — a oba su raniji rad istih autora
(Laurel et al. 2022a/b). Rad **citira** Shi et al. 2022 ("Efficiently Computing
Local Lipschitz Constants of Neural Networks via Bound Propagation", NeurIPS),
ali isključivo u sekciji 5.5 o monotonosti, i to samo da se ogradi na nivou
postavke eksperimenta. Za Lipšicov zadatak — isti zadatak koji Shi et al. rešavaju
— poređenja nema.

To je bitno jer headline brojevi iz rada ("do 182× preciznije" na FFNNBig,
"do 2750×" na ConvBig) mere razliku prema **slabom** baseline-u. Ovaj dokument
meri razliku prema stvarnom state of the art-u.

## Šta se tačno poredi

Pasado 5.4 ograničava izvod izlaza po skalarnom haze parametru:
`x(t) = img + t*(1 - img)`, `t ∈ [0, eps]`, i izveštava

    lc = max_j max(|lb_j|, |ub_j|)   nad dF_j/dt

prosečeno preko **30 tačno klasifikovanih** MNIST test slika, za 16 epsilona
(`get_lipschitz.py`: `num_images_to_test = 30`, `break` na
`correct_images == 30`, `n_splits = 1`, float64, bez normalizacije ulaza).

**Redukcija koja čini poređenje mogućim.** `x(t)` je afino u `t`, pa umesto
ograničavanja punog 784-dimenzionog ulaznog Jakobijana izražavamo haze mapu
unutar modela i tražimo od auto_LiRPA Jakobijan **po skalaru `t`** — što jeste
`dF/dt`. Ista veličina, bez konverzionog faktora. Slika ulazi kao drugi,
neperturbovan ulaz (isti obrazac koji njihov sopstveni primer koristi za masku
klase).

Skripte: `experiments/colab_compare_pasado_lirpa.py` (GPU),
`lipschitz-comparison/compare_pasado_lirpa.py` (CPU, van repoa).

## Poštenost merenja

- auto_LiRPA radi u svom **podrazumevanom `optimize=True` (CROWN-Optimized)**
  režimu — najjačem standardnom, sa po-neuronski optimizovanim relaksacijama.
  **Branch-and-Bound nije uključen**; to je njihov mehanizam za dodatno stezanje
  i predstavlja glavnu otvorenu stavku (v. dole).
- float64 svuda, isto kao Pasado.
- Verzija: auto_LiRPA 0.7.2 (aktuelni master), torch 2.11.
- **Vremena nisu uporediva** i ne izveštavaju se kao rezultat: auto_LiRPA je
  projektovan za GPU, Pasado radi na CPU. Metrika je tesnoća.

## Validacija: da li su granice uopšte korektne

Provereno egzaktnim autograd izračunavanjem na slici 0 (`big` mreža,
`lipschitz-comparison/exact_dfdt_check.py`):

    tačno max_j |dF_j/dt| pri t=0     = 245.22
    auto_LiRPA za istu sliku, eps→0   = 245.59   (0.15 % iznad, sound i tesno)

Ovo je usput i prva nezavisna provera korektnosti bilo koje strane u ovom
projektu.

## Rezultati (30 slika, iste slike sa obe strane)

Odnos `lirpa / precise` — vrednost > 1 znači da je Pasado tešnji.

| eps | 3layer | 4layer | 5layer | big |
|---|---:|---:|---:|---:|
| 0.632 | 1.006 | 1.289 | 1.748 | 2.248 |
| 0.356 | 1.298 | 1.822 | 2.633 | **2.698** |
| **0.200** | **1.323** | **1.928** | **2.826** | 2.284 |
| 0.112 | 1.173 | 1.554 | 2.190 | 1.586 |
| 0.063 | 1.035 | 1.177 | 1.289 | 1.137 |
| 0.036 | 1.034 | 1.056 | 1.027 | 1.001 |
| 0.020 | 1.034 | 1.036 | 1.070 | 1.049 |
| 0.006 | 1.004 | 1.008 | 1.016 | 1.084 |
| 0.0002 | 1.000 | 1.001 | 0.9997 | 1.086 |
| 0.000112 | 1.000 | 1.002 | 0.9995 | 1.085 |

Arhitekture, jer je za tumačenje presudno razlikovati dubinu od širine:

| mreža | broj sigmoid slojeva | širina | vrhunac odnosa |
|---|---:|---:|---:|
| 3layer | 2 | 100 | 1.32 |
| 4layer | 3 | 100 | 1.93 |
| 5layer | 4 | 100 | **2.83** |
| big | 4 | 1024 | 2.70 |

### Nalazi

1. **Pasado je tešnji, ali za 1.3–2.8×, ne za redove veličine.** Za poređenje,
   na 4layer pri eps=0.2 Pasado je ~35× tešnji od zonotope AD-a (baseline iz
   rada) ali samo **1.93×** tešnji od auto_LiRPA. Prednost je stvarna; brojevi
   iz rada su naduvani slabim baseline-om.

2. **Prednost raste sa brojem nelinearnih slojeva, ne sa širinom.** Po broju
   sigmoid slojeva: 2 → 1.32×, 3 → 1.93×, 4 → 2.83×. Ali `big` ima **isti broj
   sigmoid slojeva kao 5layer** (4) uz 10× veću širinu, i daje 2.70× — neznatno
   *manje*, ne više. Mehanizam je dakle složeno gomilanje po-slojne relaksacione
   greške kroz dubinu; širina mu ne doprinosi.

   Ovo je ispravka ranijeg zaključka u ovom projektu: dok su postojali samo
   3/4/5layer rezultati, trend je opisan kao rast "sa dubinom/veličinom" i
   očekivalo se da `big` bude najviši. Nije — vrhunac je na 5layer.

3. **Prednost je koncentrisana u srednjem režimu** (eps ≈ 0.1–0.36). Na malim
   epsilonima obe metode konvergiraju: tangentna relaksacija je tu praktično
   egzaktna.

4. **Na 5layer, pri eps ≤ 0.002, Pasado je neznatno labaviji** i od auto_LiRPA
   i od običnog zonotope AD-a (~0.05 %). Sitno, ali sistematski — konzistentno
   sa akumulacijom fiksnog režijskog troška (fitovana ravan + član maksimalne
   greške + nedokumentovana `np.random.normal(0, 1e-4)` perturbacija koeficijenta
   A) kroz 4 sigmoid sloja.

5. **Na `big`, za razliku od uskih mreža, odnos ne pada na 1.0 u limesu** nego
   se zaravnjuje na ~1.085. Pošto pri eps → 0 obe metode teže istoj tačnoj
   vrednosti, to znači da auto_LiRPA na 1024-širokoj mreži zadržava ~8.5 %
   relaksacione labavosti koju na uskim mrežama nema. (Na pojedinačnoj slici 0
   je ipak bio tesan — 245.59 naspram tačnih 245.22 — pa taj prosečni jaz dolazi
   od drugih slika.)

5. **auto_LiRPA-in Jacobian put ne skalira na široke mreže.** `_expand_jacobian`
   (auto_LiRPA/jacobian.py) eksplicitno gasi *sve* sparse opcije čim se u grafu
   pojavi Jacobian čvor, pa se međugranice računaju nad gustim 1024×1024
   identity specovima (biblioteka sama upozorava). Jedan solve na `big` dostiže
   ~10.7 GB; mašina sa 15.7 GB ne može da izvede ni dva uzastopno. Pasado istu
   analizu radi za ~2.1 s po slici na CPU-u. Za `big` je zato korišćen A100
   (80 GB).

## Metodološka greška koja je nađena i ispravljena

Prvi `big` runovi su izveštavani sa 1 i 5 slika, a upoređivani sa Pasado
vrednostima koje su **prosek preko 30 slika**. To nisu iste veličine — slika 0
ima natprosečan gradijent (tačno 245.2) naspram 30-slikovnog proseka od 168.1,
pa je "Pasado ispod tačne vrednosti" izgledalo kao nekorektnost dok je zapravo
bilo poređenje jabuka i krušaka. Odnosi 2.96× i 3.67× objavljeni u toj fazi su
**povučeni**. Skripta sada odbija tiho da poredi i ispisuje upozorenje kad
`--num-images != 30`.

## Branch-and-Bound: krive se seku na ~7 podproblema

Gornja tabela poredi **jedan prolaz** obe metode. To nije kraj priče, jer se
domen može deliti: `t ∈ [0, eps]` se razlomi na k podintervala, svaki se ograniči
zasebno, uzme se maksimum. Deljenje uvek steže, pa je pitanje samo koliko
podproblema treba.

Presudno je da **obe strane imaju taj mehanizam**. Pasado ga ima ugrađenog
(`get_lipschitz.py`, `n_splits`) i pušta se sa `n_splits = 1`; auto_LiRPA ga ima
kao BaB. Prvi pokušaj ovog eksperimenta dao je auto_LiRPA-i deljenja a Pasado
ostavio nepodeljen, iz čega je izgledalo da BaB briše prednost — to je bila
greška u postavci, ne nalaz.

5layer, eps = 0.2, prosek preko istih 30 slika, isti budžet na obe strane
(`lipschitz-comparison/bab_experiment.py`, best-first bisekcija):

| podproblema | Pasado (`n_splits=k`) | auto_LiRPA + BaB | odnos |
|---:|---:|---:|---:|
| 1 | 1288.40 | 3640.73 | **2.83×** Pasado tešnji |
| 3 | 559.26 | 1533.44 | **2.74×** Pasado tešnji |
| 5 | 350.83 | 483.39 | **1.38×** Pasado tešnji |
| 7 | 249.14 | 250.69 | **1.01×** izjednačeno |
| 15 | 166.54 | 149.38 | **0.90×** auto_LiRPA tešnji |

**Nalaz: Pasado-ova prednost u tesnoći postoji samo pri malom budžetu.** Opada
monotono i nestaje oko sedam podproblema; iznad toga auto_LiRPA je tešnji.
Tvrdnja "Pasado je tešnji od state of the art-a" **ne stoji bezuslovno** — stoji
samo uz "pri jednakom i malom broju podproblema".

Kontrolna provera: BaB skripta na k=1 daje 3640.733, identično jednoprolaznom
poređenju iz gornje tabele — dva nezavisna puta kroz kod, isti broj.

### Pun sweep preseka (A100): presek na ~7 je pravilo — i na malim epsilonima se pomera naviše, ne naniže

GPU sweep (`experiments/colab_bab_sweep.py`, A100, 30 slika, isti protokol kao
gore) pokrio je 3/4/5layer × eps indekse 0–4 × K ∈ {1,3,5,7,9}; Pasado strana
je `pasado_splits.csv` (lokalni CPU sweep, `n_splits`, iste slike). Kontrola:
A100 brojevi za 5layer/eps=0.2 identični su CPU eksperimentu iznad
(3641 / 1533 / 483.4 / 250.7) — treći nezavisni put kroz kod, isti brojevi.

Odnos `lirpa / pasado` pri jednakom budžetu (>1 znači da je Pasado tešnji);
„presek" je linearno interpolisan K na kome odnos pada na 1:

| mreža | eps | K=1 | K=3 | K=5 | K=7 | K=9 | presek |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3layer | 0.632 | 1.006 | 1.473 | 1.352 | 1.011 | 0.785 | ~7.1 |
| 3layer | 0.356 | 1.298 | 1.447 | 1.204 | 0.953 | 0.904 | ~6.6 |
| 3layer | 0.200 | 1.323 | 1.340 | 1.126 | 1.014 | 0.957 | ~7.5 |
| 3layer | 0.112 | 1.172 | 1.227 | 1.084 | 1.021 | 1.008 | >9 |
| 3layer | 0.063 | 1.035 | 1.130 | 1.058 | 1.029 | 1.010 | >9 |
| 4layer | 0.632 | 1.288 | 2.310 | 2.018 | 1.206 | 0.776 | ~8.0 |
| 4layer | 0.356 | 1.822 | 2.281 | 1.538 | 1.004 | 0.807 | ~7.0 |
| 4layer | 0.200 | 1.928 | 1.797 | 1.253 | 0.990 | 0.918 | ~6.9 |
| 4layer | 0.112 | 1.554 | 1.421 | 1.148 | 1.020 | 0.995 | ~8.6 |
| 4layer | 0.063 | 1.177 | 1.239 | 1.073 | 1.039 | 1.011 | >9 |
| 5layer | 0.632 | 1.748 | 3.669 | 3.113 | 1.415 | 0.615 | ~8.0 |
| 5layer | 0.356 | 2.633 | 3.694 | 2.120 | 0.930 | 0.681 | ~6.9 |
| 5layer | 0.200 | 2.826 | 2.741 | 1.378 | 1.006 | 0.883 | ~7.1 |
| 5layer | 0.112 | 2.190 | 1.763 | 1.288 | 1.055 | 1.004 | >9 |
| 5layer | 0.063 | 1.289 | 1.420 | 1.168 | 1.052 | 1.039 | >9 |

Tri nalaza:

1. **Presek na ~7 podproblema je pravilo, ne izuzetak.** Svuda gde se krive
   uopšte seku, seku se u uskom pojasu K ≈ 6.6–8.6 — kroz tri dubine i ceo
   srednji režim epsilona. 5layer/eps=0.2 nije bila povlašćena tačka.
2. **Ranije očekivanje da je na manjim epsilonima presek na manjem budžetu
   bilo je pogrešno — pomera se naviše.** Za eps ≤ 0.11 presek izlazi iznad
   K=9: Pasado ostaje marginalno (1.00–1.05×) tešnji kroz ceo mereni budžet.
   U tom režimu obe metode konvergiraju ka tačnoj vrednosti pa deljenje sve
   manje pomaže obema — ali auto_LiRPA ne stiže da pretekne.
3. **Odnos nije monoton u K: vrh je na K=3, ne na K=1** (npr. 5layer/eps=0.63:
   1.75 → 3.67 → 3.11 → 1.42 → 0.61). Objašnjenje je u strategiji deljenja:
   Pasado deli interval uniformno na K delova, a auto_LiRPA BaB best-first
   bisekcijom — posle dva deljenja njegov najgori podinterval i dalje ima
   širinu eps/2 naspram Pasadovih eps/3, pa uniformna podela na malom budžetu
   steže više. Best-first prestiže tek kad budžet dovoljno poraste.

Podaci: `lipschitz-comparison/bab_sweep_gpu.csv` (auto_LiRPA strana),
`lipschitz-comparison/bab_crossover.csv` (spojena tabela),
`lipschitz-comparison/merge_bab_crossover.py` (spajanje).

**Važna kvalifikacija preseka:** ceo ovaj presek važi za jednak **broj
podproblema** — metriku koja favorizuje skupljeg. Pri jednakom **vremenu**
presek ne postoji: Pasado dominira u svih 15 ćelija, sa 14–50× manje vremena
za istu tesnoću (CPU protiv A100). V. `EQUAL_TIME_COMPARISON.md`.

### Gde Pasado i dalje ubedljivo vodi: cena po podproblemu (sada izmereno i na GPU)

Presek na K≈7 važi za **broj podproblema**, ne za utrošeno vreme. A100 sweep
daje ukupno vreme po slici po epsilonu za svih 25 solve-ova (K=1+3+5+7+9):
3layer ~14.3 s, 4layer ~24.2 s, 5layer ~37.1 s, tj. približno **0.57 / 0.97 /
1.49 s po podproblemu** (procena deljenjem ukupnog vremena; CROWN-Optimized
staje posle 13–19 iteracija pa cena po solve-u varira, ali ne za red veličine).

Na budžetu izjednačenja K=7, po slici po epsilonu (Pasado vremena su prosek
eps indeksa 0–4 iz `pasado_splits.csv`):

| mreža | Pasado, CPU, k=7 | auto_LiRPA, A100, 7 podpr. | odnos |
|---|---:|---:|---:|
| 3layer | 0.135 s | ~4.0 s | ~30× |
| 4layer | 0.196 s | ~6.8 s | ~35× |
| 5layer | 0.286 s | ~10.4 s | ~36× |

Dakle i protiv A100 — najpovoljnijeg realnog okruženja za auto_LiRPA — Pasado
na laptop CPU-u plaća **~30× manje vremena** da dođe do iste tesnoće. Ovo
zamenjuje raniju CPU-CPU procenu (~45×), koja je bila nepoštena prema GPU
alatu; sadašnja tvrdnja sme u tekst, uz ogradu da su Pasado vremena iz noćnog
nepinovanog sweep-a, pa su okvirna (red veličine je ipak nedvosmislen:
desetinke naspram sekundi).

## Otvoreno

- **Najmanji epsiloni (indeksi 5+, eps ≤ 0.036) nisu u sweep-u** (pokriveni su
  indeksi 0–4). Trend sa indeksa 3–4 sugeriše da Pasado i tamo ostaje
  marginalno iznad 1 kroz K=9, ali to nije mereno.
- **`big` nije u BaB sweep-u** — solve na A100 traje ~4.7 s, pa je to još ~2h
  mašinskog vremena; bez toga tvrdnje o preseku važe samo za uske mreže.
- **Dublje mreže**: pošto prednost prati broj sigmoid slojeva a ne širinu,
  prirodan sledeći eksperiment je mreža sa 6-8 sigmoid slojeva — da se vidi da
  li rast 1.32 → 1.93 → 2.83 nastavlja ili se zasićuje. Trenutni podaci imaju
  samo tri tačke po dubini i jednu po širini.
- Poređenje pokriva samo Section 5.4 (Lipšic na MNIST FCN sigmoid mrežama).
  Ne generalizovati na "Pasado je tešnji od CROWN-a" — ostale Pasado sekcije
  (ODE, finansije, monotonost) nisu merene ovim putem.
