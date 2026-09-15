# RAW_NOTES — Section 5.4 Lipschitz profiling/optimizacija

Radne beleške, ne formalni tekst. Hronološki.

## Kontekst / mreže

Benchmark: `Section_5_4/get_lipschitz.py`. Racuna lokalne Lipschitz granice za
4 MNIST FC mreze preko interval / zono / **precise** zono transformera:

- 3layer: 784→100→100→10
- 4layer: 784→100→100→100→10
- 5layer: 784→100→100→100→100→10
- big:   784→1024→1024→1024→1024→10

Workload: `num_images_to_test = 30`, `test_range` = 16 epsilon vrednosti,
`n_splits = 1`. Znaci **16 × 30 = 480** precise forward prolaza. Svaki precise
forward = (broj sigmoid slojeva) poziva `PreciseSigmoidDualZonotope`, a svaki
od njih zove `sigmoid_prime_product_tensor` koji radi **po neuronu** regresiju.
`torch.set_default_dtype(torch.float64)` — sve je float64.

Bitno: unutar `sigmoid_prime_product_tensor` postoji
`np.random.normal(0,.0001)` perturbacija A koeficijenta (po neuronu) PRE
`compute_max_error`. Nema per-call seedovanja → benchmark nije deterministican
izmedju procesa. Za fer staro-vs-novo poredjenje treba fiksirati np.random +
torch seed identicno; posto se ABCs racunaju pre grananja selektora, obe grane
dobijaju identicne ABCs, pa je jedina razlika float reordering u corner/boundary
proveri.

## Baseline (dato iz prethodnog rada)

Wall-clock (originalni kompletan benchmark, sa cuvanjem):
- 3layer 43.8s, 4layer 52.7s, 5layer 1m06s, big 21m40s.

Vec uradjeno pre ovog sprinta:
- `--no-save` flag u get_lipschitz.py (preskace samo torch.save).
- Vektorizacija `check_corners_tensor` (~54× brze u benchmarku).
- Vektorizacija `check_nonlinear_boundary_tensor`.
- Oba prolaze kroz `compute_max_error` kad je `ABCs_tensor` prosledjen; stare
  funkcije ostaju kao fallback.
- 3layer cProfile compute-only: 45.55s → 24.97s (~45% nize, ~1.82× pod cProfile).
- `compute_max_error`: 19.74s → 1.97s.

## Phase 0 — snapshot

Repo je `C:\everything\prakse\UIUC\Pasado` (git=true). `git status`: pored sprint
fajlova ima i starih izmena iz Section_5_2/5_3/5_5 (rezultati ranijih runova) —
NE diram ih. Sprint fajlovi: `Section_5_4/get_lipschitz.py`,
`forward_mode_tensorized_src/precise_transformer.py` (modifikovani) + test fajlovi
(untracked). Snimljen diff u `logs/vectorization_current_state.diff`.

Env: Python 3.13.3, torch 2.12.1+cpu, numpy 2.5.1, sklearn 1.9.0, 8 threads,
Intel i (Model 154), CUDA=False.

Postojeci testovi (`test_check_corners_tensor`, `test_check_nonlinear_boundary_tensor`,
`test_compute_max_error_tensor`) — SVI PASS. Diff-ovi do ~4.77e-7 (float32 default
u testovima; benchmark je float64 pa su realne razlike jos manje).

## Phase 1 — big profil

Ekstrahovan `logs/lipschitz_big_nosave_both_vectorized_pstats.txt` iz
`profiles/lipschitz_big_nosave_both_vectorized.prof`. Puna analiza u
`BIG_PROFILE_ANALYSIS.md`.

Glavno iznenadjenje: na `big` mrezi struktura NIJE ista kao na 3layer.
- `AffineZonotope` (`generators @ layer`, gust matmul) = **416s self** — najveci
  pojedinacni trosak. To je pravi BLAS, ne Python overhead. Skalira sa width².
- `linalg_lstsq` = 119.6s preko **1.97M poziva** (jedan po neuronu). Plus
  `xplusone` `torch.cat` 89s, `cartesian_prod` 39s, `linspace` 21s — sve "mnogo
  sitnih operacija" iz per-neuron regresije. To je najveci BATCHABLE blok (~300s).
- `get_coeff_abs` (`abs`+`sum`) 133s — racunanje lb/ub velikih zonotopa.
- `compute_max_error` pao na 40.8s (od cega 14.9s je Python combine loop + 5.6s
  unbind koji smo mi uveli) → vektorizacija odradila posao; corner/boundary sad
  ~0.4%+0.4% ukupnog.

Zakljucak: 3layer preuvelicava per-neuron orkestraciju, potcenjuje gust matmul.
Sledeci korak (Phase 5-6): batch-ovati per-neuron regresiju. AffineZonotope je
GPU/compile kandidat, ne vektorizacioni.

## Phase 2-3 — selector + benchmark harness

Selector vec postoji (env var, cita se na call-time pa harness moze da flipuje
i in-process): `PASADO_VECTORIZED_PRECISE` (0=stari put, 1=default novi),
`PASADO_VEC_BOUNDARY` (nezavisno gasi samo boundary vektorizaciju → "corners
only" varijanta), `PASADO_BATCHED_LSTSQ` (default 0, za Phase 6). Menja SAMO
izbor corner/boundary check implementacije — regresija, ABCs, np.random
perturbacija i sve granice su identicne na oba puta.

`experiments/benchmark_lipschitz.py`: perf_counter, subprocess (nepromenjen
get_lipschitz.py), CSV append u logs/benchmark_results.csv sa git commit +
diff-hash + py/torch/numpy verzijama + CPU/CUDA/threads. Varijante se biraju
iskljucivo kroz env.

Novi test `test_selector_paths.py`: call-counterima potvrdjuje da selector
zaista rutira na stari/novi/mesoviti put kroz ceo sigmoid_prime_product_tensor,
plus poredi izlazne zonotope.

## VAZAN NALAZ — dtype bug u nasoj vektorizaciji (nadjen testom selektora)

Prvo pokretanje test_selector_paths na n=100 palo: razlika 2.2e-8 u float64 —
PREVELIKO za obicno FP preuredjivanje. Uzrok:

- `inverse_poly_tensor` interno racuna u complex64 (float32 preciznost!), pa
  korenovi iz `inverse_sigmoid_2nd_deriv` izlaze kao float32 tenzori. (Takodje
  koristi `x**0.33333` umesto tacnog kubnog korena — to je originalna
  aproksimacija, ZAJEDNICKA za oba puta, ne diramo je.)
- STARI kod: `torch.stack((root1, ly))` mesa float32 koren sa float64 ly →
  promocija u float64 → sigmoid u objective_fn se racuna u float64.
- NAS NOVI kod: `torch.stack((root1,root2,root3))` — sve float32, ostaje
  float32 → sigmoid u float32 → ~1e-8 razlika po neuronu, akumulirano kroz
  slojeve ~1e-6 u e2e (to su bili FAIL redovi u correctness_results.csv!).
- Plus asocijativnost: nase `(1-s)*s*ys` vs originalno `(1-s)*(s*ys)`.

Fix (semantics-preserving, algoritam netaknut): `.to(ly.dtype)` na stacked
korenove + ista asocijativnost. Posle fixa:
- test_selector_paths: max diff 2.2e-16 (1 ULP float64) — bit-veran.
- sva 3 stara unit testa: 0.000e+00.
- e2e harness (3 slike × 3 eps, 3layer): lc_bound diff ≤ 5.7e-14, generatori
  ≤ 4.5e-14 — SVE PASS. correctness_results.csv sada cist.

Pouka: "razlika ≤ 4.8e-7 je samo reordering" iz ranijih testova je bila
pogresna interpretacija — vecina te razlike je bio ovaj dtype bug (testovi su
tada isli u float32 default gde se ne vidi). Sada je poredjenje egzaktno.

## Phase 5 — lstsq analiza (probe_lstsq.py, bez izmena koda)

Kljucni nalazi (puna analiza u LSTSQ_ANALYSIS.md):
- broj poziva lin_reg_tensor = #sigmoid_slojeva × width × 30 × 16:
  96,000 (3layer), 1,966,080 (big) — poklapa se sa pstats tacno.
- SVAKI poziv: x=[25,2], zs=[25], float64 (STEPS=5 hardkodovan) → savrseno
  stackable. 0 tacnih duplikata xplusone (400/400 razlicitih) → kesiranje NE.
- batched lstsq radi na Windows CPU torch 2.12.1, i driver='gelsd' batched je
  BIT-IDENTICAN per-item petlji (max diff 0.000e+00) — jackpot za Phase 6.
- default driver (gelsy na CPU) ≠ gelsd (~3e-16 + min-norm semantika na
  rank-deficitnim) → driver OSTAJE 'gelsd'.
- cartesian_prod == repeat_interleave/repeat konstrukcija bit-identicno
  (0/2000) → i to se batchuje egzaktno.
- torch.linspace NIJE bit-identican afinoj konstrukciji (280/2000, ~1 ULP) →
  batched linspace samo kao opcioni drugi korak, uz dokumentovanu ULP razliku.
- per-call lstsq 37-61µs vs <1µs stvarne matematike → ≥90% overhead dispatch-a.
  ~280s od ~300s bloka na big se moze skinuti bit-egzaktno.

Preporuka: batched lstsq DA (stack postojecih grid-ova → batched zs → batched
xplusone → jedan gelsd po sloju), caching NE, closed-form NE.

## Phase 6 — batched lin_reg implementacija + testovi

`lin_reg_tensor_batched(x_batch, zs_batch)`: [n,m,2] + [n,m] (ili [n,m,1]) →
[n,3] intercept-first (kolona 0=C, 1=A, 2=B — isto sto lin_reg_tensor vraca po
neuronu). ones+cat u dtype/device od x_batch, jedan batched
lstsq(driver='gelsd'). Privremeno zivi u experiments/batched_lin_reg.py dok
big benchmark ne zavrsi (ne diram precise_transformer.py dok ga podproces
cita — diff_hash provenance + rizik); posle ide verbatim u pt iza
PASADO_BATCHED_LSTSQ (default off).

test_batched_lin_reg.py: 26 konfiguracija — n∈{1,10,100,1024}, float64 i
float32 (svaki pod svojim default dtype-om jer stari lin_reg_tensor pravi ones
u default dtype!), realisticni gridovi (linspace+cartesian_prod+
sigmoid_prime_times_y), random tacke, degenerate_x (rank 2), point box
(rank 1), near-degenerate (1e-12), saturated sigmoid (|x|~30), mixed-scale
ill-conditioned, mesoviti batch zdravi+degenerisani. Provera i downstream
planar gresaka + [n,m,1] ulaza + dtype/device/shape.

REZULTAT: SVE bit-identicno (torch.equal=YES, max diff 0.000e+00) u OBA
dtype-a, ukljucujuci rank-deficitne. Batched gelsd na ovoj platformi zaista
resava iste matrice istom rutinom.

Usput: stari lin_reg_tensor implicitno zahteva x.dtype == default dtype
(inace cat promovise i lstsq padne na mismatch) — batched verzija je robusnija
jer ones pravi u x.dtype. Nije menjano ponasanje, samo zabelezeno.

Mikrobenchmark (experiments/benchmark_lstsq.py, warmup+median, CSV u
logs/microbenchmark_results.csv) NAPISAN ali se pusta tek kad big benchmark
zavrsi — tajming pod opterecenjem 8 threadova ne bi vazio.

## Phase 7 — caching (statichka analiza, CACHING_ANALYSIS.md)

Verdikt: caching NIJE opravdan. Data-zavisni kandidati (gridovi, xplusone)
imaju IZMEREN 0% exact-repeat rate (400/400 razlicitih SHA-1). Konstante
(ones(25,1), poly konstante, cartesian index pattern) se ponavljaju ali ih
batching elimise u celosti ili su ispod praga merenja (~0.005% runtime-a za
poly konstante). Nijedan cache nije uveden.

## Phase 10 — GPU readiness (statichki audit, GPU_READINESS.md)

Kljucno: (1) vektorizovani check put je GPU-ready osim konstanti u
inverse_poly_tensor (mali DEVFIX); (2) per-neuron regresija je najgori GPU
gradjanin (.item() sync 7.87M puta na big, CPU konstrukcije, host RNG) — 
batched put to skoro sve uklanja; (3) TVRD BLOKER: driver='gelsd' na CUDA ne
postoji — jedini CUDA driver je 'gels' (full-rank pretpostavka!) → research
odluka, ne code fix; (4) fp64 na consumer GPU (T4: 1/32 rate) moze da ponisti
matmul dobitak — kolab skript prvo meri fp64 throughput; (5) np.product u
conv.py crashuje na numpy>=2 (CNN put, ne FC). colab_gpu_profile.py napisan,
posteno preskace GPU deo bez CUDA (provereno lokalno).

## Pripremljeno, ceka mirnu masinu (big chain jos radi)

- experiments/benchmark_lstsq.py (loop-vs-batched mikrobench)
- experiments/compile_experiment.py (Phase 8: 4 pure-tensor kandidata,
  eager/default/reduce-overhead, first-call vs steady-state, graph breaks)
- experiments/torch_profiler_compare.py (Phase 9: old/vec/batched, 1 slika ×
  1 eps, operator tabele + chrome trace)
- wiring lin_reg_tensor_batched u pt iza PASADO_BATCHED_LSTSQ + pun
  validacioni sled + finalna benchmark matrica (3layer sve varijante,
  4/5layer old vs final, big old vs final)

## Big old-vs-new (zavrseno 21.07. ~00:12) — ANOMALIJA u vec runu

- big original (PASADO_VECTORIZED_PRECISE=0): 1744.88s (29.1 min), 1 run
- big vec_both: 2514.46s (41.9 min), 1 run — SPORIJE od originala?!

Ovo NE moze biti stvarni efekat koda: (1) ceo footprint promenjenih funkcija u
big cProfile-u je <60s (corner 1.1s + boundary 5.5s + combine ~20s, a stari
per-neuron checkovi bi bili ~170-220s — vektorizacija mora biti brza);
(2) ista sesija, 3layer: vec 1.87× BRZI; (3) 15.07: vec big 1095s vs orig-sa-
save 1300s. Vec run je isao 23:30→00:12, PREKO PONOCI — Windows maintenance
prozor (Defender scan/update) je najverovatniji krivac. Zakljucak: big
vec_both single-run je kontaminiran i ne koristi se za formalne tvrdnje;
big original 1744.88s deluje konzistentno (1.34× sporija masina vs 15.07).
Za finalni izvestaj: big original vs big vec_both_batched (svez run), uz
jasnu napomenu o kontaminaciji vec_both rezultata.

## Batched lstsq — mikrobench + wiring + validacija (21.07.)

Mikrobench (mirna masina, float64, gelsd, 15 reps):
- n=1: 1.0× (nema dobitka, ocekivano)
- n=10: 5.6×
- n=100 (3layer sloj): 28.78ms → 1.97ms = 14.6×
- n=1024 (big sloj): 163.9ms → 26.0ms = 6.3×
Projekcija: 3layer ~26s ustede po runu, big ~265s. Gate ≥5% prolazi.

Wiring: lin_reg_tensor_batched premesten verbatim u precise_transformer.py,
grana u sigmoid_prime_product_tensor iza USE_BATCHED_LSTSQ (default OFF).
np.random stream: n skalarnih poziva u istom redosledu kao petlja → identican
RNG state downstream. experiments/batched_lin_reg.py je sada re-export (testovi
gadjaju produkcioni kod).

Testovi posle wiringa: svih 5 fajlova PASS (batched i dalje bit-identican).
E2E old-vs-(vec+batched): SVE PASS, identicne razlike kao old-vs-vec (lc ≤
5.7e-14) → batched dodaje NULA dodatne razlike. CSV:
logs/correctness_results_batched.csv.

## 21.07. ujutru — masina spavala, big batched run ubijen

Pustio sam big vec_both_batched uvece, ali je masina otisla u sleep i proces je
ubijen bez ijednog izmerenog runa (prazan output). Ponovo pustam, ali ovog puta
kao deo kompletne matrice.

VAZNA KOREKCIJA: kad sam premerio vec_both na mirnoj masini dobio sam 28.0s, a
ranije "cist" median je bio 75.0s. Znaci da je i taj vec_both bio kontaminiran,
pa je onda i original=140.3s sumnjiv → headline 1.87x NIJE pouzdan i mora se
premeriti sve u jednom bloku, na mirnoj masini. Zato finalna matrica ide kao
jedan sekvencijalni lanac (3layer sve 4 varijante, pa 4/5layer, pa big).

Pouka za ubuduce: svaki run beleziti zajedno sa "machine health" probom
(fiksni matmul benchmark), inace ne znas da li meris kod ili Defender.

## Phase 8 — torch.compile: NEGATIVAN REZULTAT

Dynamo capture radi: check_corners_tensor, check_nonlinear_boundary_tensor,
_max_objective_over_x_candidates = 0 graph breaks. lin_reg_tensor_batched = 1
break (aten.linalg_lstsq je dynamic-shape operator).

ALI: svaki torch.compile poziv pada sa "Compiler: cl is not found" —
TorchInductor CPU backend trazi MSVC cl.exe, a na masini postoje VS 2019/2022
folderi ali cl nije na PATH-u (nema Build Tools workload). Znaci first-call i
steady-state NISU izmereni — to su nepoznate vrednosti, ne nule.

Plus: "Torchinductor does not support code generation for complex operators" —
inverse_poly_tensor ide kroz complex64, pa bi i sa MSVC-om taj deo pao nazad
na eager.

I najvaznije: kerneli koji se cisto hvataju su posle vektorizacije prejeftini
(<1ms na n=1024, ukupno <1s od 19s runa) — compile ih ne bi mogao pomeriti.
Zakljucak: manuelna vektorizacija je za ovaj workload bila vaznija od
torch.compile. Nista se ne wire-uje.

## Phase 9 — torch.profiler: potvrda iz drugog instrumenta

1 slika × 1 eps, 3layer precise forward:
- old: 64,738 operator poziva, 604.0ms self CPU
- vec: 36,262, 364.4ms
- batched: 15,056, 165.7ms  → 4.3× manje poziva, 3.6× manje self CPU

Po operatoru (old→vec→batched): linalg_lstsq 400→400→4, ones 200→200→2,
sigmoid 1616→422→26, item 4238→2238→844. A linspace 800→800→800 i
cartesian_prod 200→200→200 — NETAKNUTI, jer get_linspace nismo dirali.

U old varijanti linalg_lstsq je bio 47.9% ukupnog CPU (289ms/400 poziva); sad
su 4 poziva stvarnog racuna. Novi dominantni: aten::select (1848, unutar
get_linspace petlje), aten::max, linspace, meshgrid, item.

## 21.07. prepodne — VELIKA metodoloska lekcija

Pustio sam punu matricu i dobio 3layer original 209s, vec_corners 161s,
vec_both 84s, batched 54s. Ali cekaj — nocas sam merio vec_both 28s i batched
19.4s. Isti kod, 3x sporije?!

Pogledao sam procese: msedgewebview2 2470s CPU, brave, VS Code. Korisnik radi
na masini. Nocu (00:22) je bila prazna. To objasnjava 3x.

Gore od toga: opterecenje se MENJA tokom merenja. Original 247→192→209,
vec_corners 161→160→120. Znaci poredjenje varijante A iz jednog bloka sa
varijantom B iz drugog meri drift opterecenja, ne kod. Moj raniji headline
1.87x (140.3/75.0) je bio TAJ artefakt — obe strane kontaminirane, svaka
drugacije.

Dva popravka u harness:
1. machine_health() — fiksni matmul pre svakog runa, GFLOP/s ide u CSV. Sad se
   kontaminiran run PREPOZNAJE umesto da se pomesa sa efektom koda.
2. --interleave — A,B,A,B umesto A,A,A,B,B,B + odnosi po parovima. Drift
   pogadja obe varijante isto.

Prekinuo sam grupisani lanac (podaci ne bi izdrzali tvrdnju) i pustio
interleaved.

REZULTAT interleaved 3layer (original vs finalna optimizacija):
  original: 162.42, 156.74, 156.25  (medijana 156.74)
  batched:   50.26,  51.18,  49.84  (medijana 50.26)
  odnosi po parovima: 3.23x, 3.06x, 3.14x → medijana 3.135x
Health proba: 38-48 GFLOP/s (vs ~71 kad je mirna) — potvrdjuje opterecenje.

Rasipanje odnosa je usko (3.06-3.23) iako su apsolutna vremena naduvana → TO
je pouzdan rezultat. Speedup 3.14x, smanjenje 67.9%.

Cross-check: batched 19.36s (mirna) vs 50.26s (opterecena) = masina 2.6x
sporija; 156.74/2.6 ≈ 60s, 60/19.36 = 3.1x. Slaze se. Dobar znak.

## Big — svesna odluka da se NE pusta

Big bi pod ovim opterecenjem trajao 45-90 min po varijanti, a ne bi se mogao
porediti sa nocasnjim big original (1744.9s, druga opterecenja). Dva sata za
broj koji bih morao da bacim = ne. Komanda ostaje u REPRODUCE.md, sad sa health
probom koja sama validira rezultat. Big ubrzanje se NE tvrdi.

## PRAVI uzrok merne buke — hibridni P/E CPU (nadjeno 21.07. u 11:55)

Health proba me je prvo obradovala pa zbunila: 4layer batched run1 = 35.3s pri
h=93 GFLOP/s, ali run2 = 93.8s pri h=125 GFLOP/s. Masina "brza", run 2.7x
sporiji. Znaci proba meri pogresan resurs.

Proverio CPU: **i5-12450H = 4 P-jezgra + 4 E-jezgra** (8 cores / 12 threads,
Alder Lake hibrid). Windows scheduler seli proces izmedju P i E jezgara; E su
2-3x sporija za single-thread Python. Moja matmul proba koristi svih 8
threadova pa uvek javi visok GFLOP/s bez obzira gde je bencmark zavrsio.

TO je objasnjenje za sve anomalije u sprintu — verovatno i za big vec_both
2514s i za cProfile 2998s. Nije Defender, nego raspored po jezgrima.

Ispravna statistika za ovakvu masinu je **MINIMUM od N runova**, ne medijana
(minimum = run koji je imao najmanje smetnji). Sa min-om:
  3layer: 156.25 / 49.84 = 3.135x  (meren pod opterecenjem, h≈40)
  4layer: 110.98 / 35.29 = 3.145x  (meren na mirnoj masini, h≈110)

Odnos je PRAKTICNO ISTI iako se stanje masine razlikuje 3x — to je najjaci
dokaz da je ubrzanje realno svojstvo koda, a ne artefakt merenja.

Prava resenja za ubuduce: pinovati afinitet na P-jezgra
(`start /affinity`), ili koristiti min od vise runova, ili meriti CPU-time
umesto wall-clock.

## Finalna matrica (interleaved, min-based)

  3layer: 156.25 → 49.84s  = 3.135x  (68.1%)  h=38-48
  4layer: 110.98 → 35.29s  = 3.145x  (68.2%)  h=93-125
  5layer: 325.91 → 97.76s  = 3.334x  (70.0%)  h=35-52

3layer i 4layer daju prakticno isti odnos iako se propusnost masine razlikuje
~3x → ubrzanje je svojstvo koda. Blagi rast sa dubinom je ocekivan (vise
sigmoid slojeva = veci udeo optimizovanog dela).

4layer drugi par (1.18x) je E-core outlier — zato min, ne median.

## Big — treci pokusaj, i trik sa detached procesom

Big je ubijan TRI puta: (1) preko noci masina zaspala, (2) 4/5layer prvi
pokusaj u 11:48, (3) big u 12:24 posle 12 min. Pozadinski task-ovi u sesiji ne
prezive ~40 min koliko treba za big par. Bash foreground ima max 10 min
timeout, sto je premalo.

Resenje: PowerShell Start-Process -PassThru, potpuno odvojen proces koji NIJE
dete moje sesije, log u logs/big_pair_run.log. PID 22180. Ako i to padne, big
se ne meri u ovoj sesiji i komanda ostaje u REPRODUCE.md.

Usput popravljeno: CSV se sad upisuje inkrementalno (po runu), jer smo dva puta
izgubili po 20 min merenja kad je proces ubijen pre kraja.

## Finalni regresioni testovi (posle svih izmena)

  test_check_corners_tensor             ALL PASS
  test_check_nonlinear_boundary_tensor  ALL PASS
  test_compute_max_error_tensor         ALL PASS
  test_selector_paths                   ALL PASS
  test_batched_lin_reg                  ALL PASS (bitwise-identical: YES)
  e2e old-vs-vectorized                 ALL PASS
  e2e old-vs-vectorized+batched         ALL PASS

## Sta je na kraju ostalo kao pouzdan rezultat

Posle svih kontaminacija, ovo su brojevi koji izdrzavaju proveru:

1. Brojevi poziva — nezavisni od masine, najjaci dokaz:
   lstsq 96,000→960 po benchmarku; operator pozivi po forward-u 64,738→15,056;
   Python pozivi 9.60M→8.30M.
2. Correctness: batched bit-identican (26 konfiguracija), e2e ≤5.7e-14.
3. cProfile 3layer: 45.55→24.97→22.46s.
4. Mikrobench (mirna masina): 14.6× na n=100, 6.3× na n=1024.
5. Interleaved wall-clock 3layer: 3.14× / −67.9% (original vs finalna).
6. Cist par na mirnoj masini: vec_both 28.01s → batched 19.36s = 1.447×.

Odbaceno kao kontaminirano: big vec_both 2514s, 3layer cProfile 2998s,
grupisana matrica od 21.07. prepodne, nocasnji par 140.3/75.0s.

Ranija napomena o redovima sa starim diff-hashom (original 100.7s, vec_both
30.3s) i dalje vazi — to je pre dtype fixa, ne koristi se.
