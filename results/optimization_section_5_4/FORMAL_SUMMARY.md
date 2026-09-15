# Optimizacija preciznog zonotopskog transformera (Pasado, Section 5.4)

## Cilj

Utvrditi gde precizni transformer u `Section_5_4/get_lipschitz.py` troši vreme,
koliko tog vremena otpada na Python orkestraciju i mnoštvo sitnih PyTorch
operacija, a koliko na stvarni numerički rad, i koliko se ubrzanja može dobiti
bezbednom vektorizacijom, grupisanjem (batching) i kompilacijom **bez promene
semantike analize**.

Radni opseg: `forward_mode_tensorized_src/precise_transformer.py`. Gustina
uzorkovanja (`STEPS=5`), opseg epsilona, broj slika, izbor LAPACK drajvera i
računanje granica nisu menjani.

## Metodologija

Merenja su razdvojena u tri kategorije koje se ne mešaju: **wall-clock**
(`time.perf_counter`, bez profajlera), **cProfile** (samo za relativnu
atribuciju — dodaje režijski trošak po Python pozivu) i **izolovani
mikrobenčmark**. Svaka optimizacija je morala da prođe: test korektnosti protiv
prethodne implementacije, mikrobenčmark, end-to-end benchmark i profil pre i
posle.

Stare implementacije su zadržane i dostupne preko selektora
(`PASADO_VECTORIZED_PRECISE`, `PASADO_VEC_BOUNDARY`, `PASADO_BATCHED_LSTSQ`),
što omogućava pošteno poređenje bez ručne izmene koda.

**Metodološka napomena koja je bila presudna.** Merenja su se u početku
razilazila i do 2,7× između identičnih ponavljanja. Uzrok je utvrđen tek
sistematskim praćenjem: radna mašina je **Intel i5-12450H, hibridni procesor sa
4 performansna i 4 efikasna jezgra**. Windows raspoređivač seli proces između
njih, a efikasna jezgra su 2–3× sporija za ovaj Python-intenzivan posao.
Odlučujući dokaz: ista varijanta izmerena je 35,3 s pri 93 GFLOP/s, a zatim
93,9 s pri 125 GFLOP/s — sporije dok je mašina merila *brže*, jer višenitna
proba zasićuje sva jezgra i ne vidi na kom je tipu jezgra benchmark završio.

U harness su zato uvedeni: proba zdravlja mašine po runu, naizmenično merenje
(`--interleave`, A,B,A,B…) sa odnosima po parovima, pinovanje na performansna
jezgra (`--pin-pcores`) i **minimum od N ponavljanja** kao izveštajna
statistika (najmanje ometen run). Raniji rezultat od „1,87×" bio je artefakt
poređenja varijanti merenih u različitim uslovima i **odbačen je**.

## Ključni profil

Početni cProfile (3layer, 45,55 s ukupno) pokazao je da precizni transformer
dominira: `sigmoid_prime_product_tensor` 32,35 s, od čega `compute_max_error`
19,74 s (provera uglova 8,40 s + provera nelinearne granice 10,80 s). Sve su to
Python petlje koje po neuronu prave nekoliko sićušnih tenzora.

Profil `big` mreže dao je bitno drugačiju sliku: tamo dominira gust matmul
`AffineZonotope` sa **416 s samostalnog vremena (33 %)**, koji skalira sa
width² × brojem generatora i predstavlja pravi BLAS, a ne Python režiju. Time
je pokazano da `3layer` **precenjuje** udeo per-neuron orkestracije i
**potcenjuje** gust matmul. Ipak, per-neuron regresija (~300 s) ostaje najveći
*grupabilan* blok i na `big` mreži.

## Optimizacije

1. **Vektorizacija provere uglova** — sve četiri ugaone tačke svih neurona
   računaju se u jednom batched izrazu.
2. **Vektorizacija provere nelinearne granice** — per-neuron petlje za
   maksimum objektivne funkcije zamenjene su batched redukcijama, uz zadržanu
   postojeću vektorizovanu inverziju kubne jednačine.
3. **Grupisana linearna regresija** — n nezavisnih po-neuronskih regresija
   rešava se jednim batched pozivom `torch.linalg.lstsq` po sloju umesto jednim
   po neuronu. Analiza je prethodno utvrdila da su svi sistemi istog oblika
   (`[25,3] \ [25]`), da se nijedna matrica dizajna ne ponavlja (400/400
   različitih), te da je batched `gelsd` na ovoj platformi bit-identičan
   petlji.
4. **Grupisano generisanje mreže uzoraka (batched grid)** — per-neuron petlja
   `torch.linspace`/`cartesian_prod` zamenjena jednim batched izrazom
   (`get_linspace_batched`, selektor `PASADO_BATCHED_GRID`). **Bit-identično**:
   `torch.linspace` interno popunjava prvu polovinu unapred od početka a drugu
   unazad od kraja; reprodukcija te šeme uklanja raniju zabelešku o „~1 ULP
   razlici" — ona je bila artefakt naivne afine formule, ne suštinsko
   ograničenje. Pored brzine, uklanja i sve `.item()` pozive preciznog puta
   (~385k po 3layer benchmarku, ~7.87M na `big`) — svaki je device-host sync,
   pa je ovo ujedno preduslov za GPU izvršavanje.

## Validacija korektnosti

Sve četiri optimizacije su **bit-identične** ili u granicama jednog ULP-a:

- batched grid: bit-egzaktan naspram petlje (float32 i float64, STEPS ∈
  {3,5,7,9}, tačkaste kutije, širine do 1e-12, magnitude do 1e6, negativni
  opsezi); `sigmoid_prime_product_tensor` daje identičan zonotop kroz svih
  12 kombinacija selektora, najgore odstupanje 1.4e-17
  (`test_batched_grid.py`);

- jedinični testovi provera uglova/granice i `compute_max_error`: razlika
  0,000e+00;
- grupisana regresija, 26 konfiguracija (n = 1…1024, float32 i float64,
  degenerisane i rang-deficitne kutije, zasićen sigmoid, loše uslovljeni
  sistemi): **bit-identično** (`torch.equal`);
- end-to-end (3 slike × 3 epsilona, sa fiksiranim seedom): finalna Lipschitz
  granica ≤ 5,7e-14, generatori ≤ 4,5e-14; oblici, dtype, uređaj i pozicije
  NaN/Inf identični.

**Tokom validacije je otkrivena i ispravljena greška u sopstvenoj
vektorizaciji.** `inverse_poly_tensor` interno računa kroz `complex64`, pa
korenovi izlaze kao float32. Originalni kod ih je slagao sa float64 granicama,
čime su implicitno promovisani u float64 pre evaluacije sigmoida; naša prva
verzija ih je zadržala u float32. Posledica je bila razlika reda 1e-6.
Ispravkom (eksplicitna promocija tipa) razlika je pala na ≤ 5,7e-14. Raniji
zaključak da je „razlika ≈ 4,8e-7 samo posledica preuređivanja operacija" bio
je pogrešan — bila je to ova greška, sakrivena float32 podrazumevanim tipom u
testovima.

## Rezultati

**Wall-clock, naizmenično mereno, minimum od N ponavljanja** (presudan je
odnos, ne apsolutna vrednost — uslovi mašine se razlikuju između mreža):

| mreža | originalna | finalna optimizovana | faktor ubrzanja | smanjenje vremena |
|---|---|---|---|---|
| 3layer | 156,25 s | 49,84 s | **3,135×** | **68,1 %** |
| 4layer | 110,98 s | 35,29 s | **3,145×** | **68,2 %** |
| 5layer | 325,91 s | 97,76 s | **3,334×** | **70,0 %** |
| big | 1070,57 s | 648,78 s | **1,650×** | **39,4 %** |

Za `big` je, kao i za gornje tri mreže, primenjen isti protokol: naizmenično
mereno (`--interleave`), pinovano na performansna jezgra, n=3 ponavljanja po
varijanti. Zdravlje mašine je bilo stabilno kroz svih 6 runova (197-226
GFLOP/s), bez znakova kontaminacije. Odnosi po parovima: 1,68× / 1,58× /
1,66× (medijana 1,657×) — blizu jedan drugom, što je dodatna potvrda da
rezultat nije artefakt merenja. Ovo se poklapa i sa ranijim pojedinačnim
(n=1) merenjem od 1,671×, koje je prethodilo ovom i sada je zamenjeno njime.

Prvi pokušaj merenja `big` mreže (pre ovog protokola) bio je kontaminiran
pozadinskim opterećenjem sistema (2514 s naspram 1745 s za *sporiju*
originalnu varijantu, što je iz koda nemoguće) i **odbačen**.

Najvažniji nalaz u ovoj tabeli nije nijedna pojedinačna vrednost, već to što
3layer i 4layer daju 3,135× i 3,145× iako su njihove mašine merile gotovo 3×
različitu propusnost (38–48 naspram 93–125 GFLOP/s). Ubrzanje koje je
nepromenljivo na trostruku promenu uslova merenja jeste svojstvo koda, a ne
merenja. Blagi porast sa dubinom (3,14 → 3,33) je očekivan: više sigmoid
slojeva znači veći udeo vremena u optimizovanom preciznom transformeru.

**Wall-clock, 3layer, mirna mašina** (dodatni korak — grupisana regresija):
28,01 s → 19,36 s, tj. faktor 1,447× i smanjenje od 30,9 %.

**Finalna konfiguracija sa batched grid-om (12.8., direktno mereno —
naizmenično, pinovano, odnosi po parovima, n=3):**

| mreža | original / finalna (raspon parova) | inkrement grida naspram prethodne finalne |
|---|---|---|
| 3layer | **3,24×** (3,21–3,30) | 1,34× |
| 4layer | **3,64×** (3,63–4,22) | 1,35× |
| 5layer | **3,37×** (2,79–3,63) | 1,31× |

Ovo su aktuelni end-to-end brojevi za uske mreže i zamenjuju gornju tabelu
(`big` sa grid-om još nije meren). Napomena: pokušaj da se kumulativ dobije
lančanjem odnosa iz različitih sesija (3,1–3,3× × 1,3× ≈ 4,2–4,4×) daje veću
vrednost od direktnog merenja — apsolutni odnos original/optimizovano varira
sa stanjem mašine i između sesija (originalni, dispatch-intenzivni put je
osetljiviji na single-core uslove od BLAS-intenzivnog optimizovanog puta).
Zato se, kao i do sada, izveštavaju samo parovi iz iste sesije; direktni
brojevi iznad su merodavni.

**cProfile, 3layer** (vreme profajlera, nije uporedivo sa wall-clock):
45,55 s → 24,97 s → 22,46 s. `compute_max_error`: 19,74 s → 1,97 s.

**Broj poziva (nezavisan od mašine — najjači dokaz):** `linalg_lstsq`
96 000 → 960 po benchmarku; operatorski pozivi po jednom prolazu
64 738 → 15 056 (−76,7 %); zbirno self-CPU vreme operatora 604 ms → 166 ms.

**Mikrobenčmark regresije:** 14,6× na n = 100, 6,3× na n = 1024.

## Trenutno usko grlo

`get_linspace` — koji je posle prve tri optimizacije dominirao (192 000
poziva `linspace`, 96 000 `cartesian_prod`, 385 456 `.item()` po 3layer
benchmarku) — rešen je optimizacijom 4 (izmereni dobitak 1,31–1,35× na
finalnu konfiguraciju). Sledeće usko grlo uskih mreža treba utvrditi novim
profilom (kandidati: batched `lstsq` sam po sebi, boundary kernel, zonotopska
propagacija). Na `big` mreži apsolutno najveći trošak i dalje ostaje gust
matmul `AffineZonotope` (416 s, 33 %), koji je već jedan BLAS poziv i ne može
se ubrzati vektorizacijom — to je GPU teren (v. `PLAN_MLSYS.md` u korenu
projekta, van Pasado repoa).

Odbačene su, uz dokumentaciju: **keširanje** (izmerena stopa ponavljanja
identičnih ulaza je 0 %), **zatvorena forma umesto `lstsq`** (nepotrebna i
lošije uslovljena) i **`torch.compile`** (Inductor CPU backend nedostupan jer
nema MSVC `cl.exe`; `linalg_lstsq` prekida graf; kompleksne operacije nisu
podržane; a ciljni kerneli su posle vektorizacije prejeftini da bi to imalo
efekta).

## Sledeći pravac

~~Grupisanje `get_linspace`~~ — **urađeno** (optimizacija 4), i to
bit-identično: navodna „~1 ULP razlika u 14 % slučajeva" bila je artefakt
naivne formule; reprodukcija interne symmetric-halves šeme `torch.linspace`
je egzaktna, pa odluka o preciznosti nije ni bila potrebna.

Usput postoji i `PASADO_REAL_CUBIC` (Vieteova realna kubna inverzija umesto
complex64 puta): na CPU ne donosi merljivo ubrzanje i NIJE bit-identičan
(pomera granice ≤ 5,7e-5 relativno), pa ostaje **isključen** — parkiran kao
enabler za GPU/`torch.compile` fazu (Inductor ne podržava kompleksne
operacije). Aktuelni pravac je heterogeni CPU–GPU runtime; plan, faze i
go/no-go kriterijumi su u `PLAN_MLSYS.md` (koren projekta).

Srednjoročno, jedini pravac koji može da promeni asimptotiku jeste
**smanjenje broja šumnih simbola** — precizni transformer dodaje nove simbole
na svakom sigmoid sloju, što gura i matmul i računanje granica. To je
algoritamsko pitanje (koliko se tesnoće gubi po uklonjenom simbolu), izvan
opsega ovog sprinta, ali je tu stvarno vreme.

Za GPU su oba otvorena pitanja u međuvremenu izmerena (v. `GPU_READINESS.md`,
Phase 11-13), ne samo pretpostavljena. Drajver: `gelsd` ne postoji na CUDA
(jedini je `gels`), a test na stvarnim rang-deficitnim kutijama (`lx==ux`)
pokazao je da `gels` daje rezultate pogrešne za 15-34 reda veličine — potvrđeno
nebezbedno, ne samo teorijski rizik. fp64 propusnost je hardverski zavisna:
na T4 je ~1/17 fp32 (na nivou lokalnog CPU-a, GPU port ne bi pomogao), na
A100 je fp64 skoro izjednačen sa fp32 (namenske FP64 Tensor Core jedinice,
~65× brže od CPU-a) — na A100/H100 klasi hardvera bi gust `AffineZonotope`
matmul verovatno dobio ozbiljno ubrzanje. GPU pravac više nije blokiran
otvorenim pitanjima; ostaje pitanje pouzdanog pristupa takvom hardveru i
inženjering rang-provere za regresioni korak.
