# Optimizacija preciznog zonotopskog transformera (Pasado, Section 5.4)

## Cilj

Utvrditi gde precizni transformer u `Section_5_4/get_lipschitz.py` trosi vreme,
koliko tog vremena otpada na Python orkestraciju i mnostvo sitnih PyTorch
operacija, a koliko na stvarni numericki rad, i koliko se ubrzanja moze dobiti
bezbednom vektorizacijom, grupisanjem (batching) i kompilacijom **bez promene
semantike analize**.

Radni opseg: `forward_mode_tensorized_src/precise_transformer.py`. Gustina
uzorkovanja (`STEPS=5`), opseg epsilona, broj slika, izbor LAPACK drajvera i
racunanje granica nisu menjani.

## Metodologija

Merenja su razdvojena u tri kategorije koje se ne mesaju: **wall-clock**
(`time.perf_counter`, bez profajlera), **cProfile** (samo za relativnu
atribuciju - dodaje rezijski trosak po Python pozivu) i **izolovani
mikrobencmark**. Svaka optimizacija je morala da proe: test korektnosti protiv
prethodne implementacije, mikrobencmark, end-to-end benchmark i profil pre i
posle.

Stare implementacije su zadrzane i dostupne preko selektora
(`PASADO_VECTORIZED_PRECISE`, `PASADO_VEC_BOUNDARY`, `PASADO_BATCHED_LSTSQ`),
sto omogucava posteno poreenje bez rucne izmene koda.

**Metodoloska napomena koja je bila presudna.** Merenja su se u pocetku
razilazila i do 2,7x izmeu identicnih ponavljanja. Uzrok je utvren tek
sistematskim pracenjem: radna masina je **Intel i5-12450H, hibridni procesor sa
4 performansna i 4 efikasna jezgra**. Windows rasporeivac seli proces izmeu
njih, a efikasna jezgra su 2-3x sporija za ovaj Python-intenzivan posao.
Odlucujuci dokaz: ista varijanta izmerena je 35,3 s pri 93 GFLOP/s, a zatim
93,9 s pri 125 GFLOP/s - sporije dok je masina merila *brze*, jer visenitna
proba zasicuje sva jezgra i ne vidi na kom je tipu jezgra benchmark zavrsio.

U harness su zato uvedeni: proba zdravlja masine po runu, naizmenicno merenje
(`--interleave`, A,B,A,B...) sa odnosima po parovima, pinovanje na performansna
jezgra (`--pin-pcores`) i **minimum od N ponavljanja** kao izvestajna
statistika (najmanje ometen run). Raniji rezultat od "1,87x" bio je artefakt
poreenja varijanti merenih u razlicitim uslovima i **odbacen je**.

## Kljucni profil

Pocetni cProfile (3layer, 45,55 s ukupno) pokazao je da precizni transformer
dominira: `sigmoid_prime_product_tensor` 32,35 s, od cega `compute_max_error`
19,74 s (provera uglova 8,40 s + provera nelinearne granice 10,80 s). Sve su to
Python petlje koje po neuronu prave nekoliko sicusnih tenzora.

Profil `big` mreze dao je bitno drugaciju sliku: tamo dominira gust matmul
`AffineZonotope` sa **416 s samostalnog vremena (33 %)**, koji skalira sa
width^2 x brojem generatora i predstavlja pravi BLAS, a ne Python reziju. Time
je pokazano da `3layer` **precenjuje** udeo per-neuron orkestracije i
**potcenjuje** gust matmul. Ipak, per-neuron regresija (~300 s) ostaje najveci
*grupabilan* blok i na `big` mrezi.

## Optimizacije

1. **Vektorizacija provere uglova** - sve cetiri ugaone tacke svih neurona
   racunaju se u jednom batched izrazu.
2. **Vektorizacija provere nelinearne granice** - per-neuron petlje za
   maksimum objektivne funkcije zamenjene su batched redukcijama, uz zadrzanu
   postojecu vektorizovanu inverziju kubne jednacine.
3. **Grupisana linearna regresija** - n nezavisnih po-neuronskih regresija
   resava se jednim batched pozivom `torch.linalg.lstsq` po sloju umesto jednim
   po neuronu. Analiza je prethodno utvrdila da su svi sistemi istog oblika
   (`[25,3] \ [25]`), da se nijedna matrica dizajna ne ponavlja (400/400
   razlicitih), te da je batched `gelsd` na ovoj platformi bit-identican
   petlji.
4. **Grupisano generisanje mreze uzoraka (batched grid)** - per-neuron petlja
   `torch.linspace`/`cartesian_prod` zamenjena jednim batched izrazom
   (`get_linspace_batched`, selektor `PASADO_BATCHED_GRID`). **Bit-identicno**:
   `torch.linspace` interno popunjava prvu polovinu unapred od pocetka a drugu
   unazad od kraja; reprodukcija te seme uklanja raniju zabelesku o "~1 ULP
   razlici" - ona je bila artefakt naivne afine formule, ne sustinsko
   ogranicenje. Pored brzine, uklanja i sve `.item()` pozive preciznog puta
   (~385k po 3layer benchmarku, ~7.87M na `big`) - svaki je device-host sync,
   pa je ovo ujedno preduslov za GPU izvrsavanje.

## Validacija korektnosti

Sve cetiri optimizacije su **bit-identicne** ili u granicama jednog ULP-a:

- batched grid: bit-egzaktan naspram petlje (float32 i float64, STEPS  in
  {3,5,7,9}, tackaste kutije, sirine do 1e-12, magnitude do 1e6, negativni
  opsezi); `sigmoid_prime_product_tensor` daje identican zonotop kroz svih
  12 kombinacija selektora, najgore odstupanje 1.4e-17
  (`test_batched_grid.py`);

- jedinicni testovi provera uglova/granice i `compute_max_error`: razlika
  0,000e+00;
- grupisana regresija, 26 konfiguracija (n = 1...1024, float32 i float64,
  degenerisane i rang-deficitne kutije, zasicen sigmoid, lose uslovljeni
  sistemi): **bit-identicno** (`torch.equal`);
- end-to-end (3 slike x 3 epsilona, sa fiksiranim seedom): finalna Lipschitz
  granica <= 5,7e-14, generatori <= 4,5e-14; oblici, dtype, ureaj i pozicije
  NaN/Inf identicni.

**Tokom validacije je otkrivena i ispravljena greska u sopstvenoj
vektorizaciji.** `inverse_poly_tensor` interno racuna kroz `complex64`, pa
korenovi izlaze kao float32. Originalni kod ih je slagao sa float64 granicama,
cime su implicitno promovisani u float64 pre evaluacije sigmoida; nasa prva
verzija ih je zadrzala u float32. Posledica je bila razlika reda 1e-6.
Ispravkom (eksplicitna promocija tipa) razlika je pala na <= 5,7e-14. Raniji
zakljucak da je "razlika ~ 4,8e-7 samo posledica preureivanja operacija" bio
je pogresan - bila je to ova greska, sakrivena float32 podrazumevanim tipom u
testovima.

## Rezultati

**Wall-clock, naizmenicno mereno, minimum od N ponavljanja** (presudan je
odnos, ne apsolutna vrednost - uslovi masine se razlikuju izmeu mreza):

| mreza | originalna | finalna optimizovana | faktor ubrzanja | smanjenje vremena |
|---|---|---|---|---|
| 3layer | 156,25 s | 49,84 s | **3,135x** | **68,1 %** |
| 4layer | 110,98 s | 35,29 s | **3,145x** | **68,2 %** |
| 5layer | 325,91 s | 97,76 s | **3,334x** | **70,0 %** |
| big | 1070,57 s | 648,78 s | **1,650x** | **39,4 %** |

Za `big` je, kao i za gornje tri mreze, primenjen isti protokol: naizmenicno
mereno (`--interleave`), pinovano na performansna jezgra, n=3 ponavljanja po
varijanti. Zdravlje masine je bilo stabilno kroz svih 6 runova (197-226
GFLOP/s), bez znakova kontaminacije. Odnosi po parovima: 1,68x / 1,58x /
1,66x (medijana 1,657x) - blizu jedan drugom, sto je dodatna potvrda da
rezultat nije artefakt merenja. Ovo se poklapa i sa ranijim pojedinacnim
(n=1) merenjem od 1,671x, koje je prethodilo ovom i sada je zamenjeno njime.

Prvi pokusaj merenja `big` mreze (pre ovog protokola) bio je kontaminiran
pozadinskim opterecenjem sistema (2514 s naspram 1745 s za *sporiju*
originalnu varijantu, sto je iz koda nemoguce) i **odbacen**.

Najvazniji nalaz u ovoj tabeli nije nijedna pojedinacna vrednost, vec to sto
3layer i 4layer daju 3,135x i 3,145x iako su njihove masine merile gotovo 3x
razlicitu propusnost (38-48 naspram 93-125 GFLOP/s). Ubrzanje koje je
nepromenljivo na trostruku promenu uslova merenja jeste svojstvo koda, a ne
merenja. Blagi porast sa dubinom (3,14 -> 3,33) je ocekivan: vise sigmoid
slojeva znaci veci udeo vremena u optimizovanom preciznom transformeru.

**Wall-clock, 3layer, mirna masina** (dodatni korak - grupisana regresija):
28,01 s -> 19,36 s, tj. faktor 1,447x i smanjenje od 30,9 %.

**Finalna konfiguracija sa batched grid-om (12.8., direktno mereno  -
naizmenicno, pinovano, odnosi po parovima, n=3):**

| mreza | original / finalna (raspon parova) | inkrement grida naspram prethodne finalne |
|---|---|---|
| 3layer | **3,24x** (3,21-3,30) | 1,34x |
| 4layer | **3,64x** (3,63-4,22) | 1,35x |
| 5layer | **3,37x** (2,79-3,63) | 1,31x |

Ovo su aktuelni end-to-end brojevi za uske mreze i zamenjuju gornju tabelu
(`big` sa grid-om jos nije meren). Napomena: pokusaj da se kumulativ dobije
lancanjem odnosa iz razlicitih sesija (3,1-3,3x x 1,3x ~ 4,2-4,4x) daje vecu
vrednost od direktnog merenja - apsolutni odnos original/optimizovano varira
sa stanjem masine i izmeu sesija (originalni, dispatch-intenzivni put je
osetljiviji na single-core uslove od BLAS-intenzivnog optimizovanog puta).
Zato se, kao i do sada, izvestavaju samo parovi iz iste sesije; direktni
brojevi iznad su merodavni.

**cProfile, 3layer** (vreme profajlera, nije uporedivo sa wall-clock):
45,55 s -> 24,97 s -> 22,46 s. `compute_max_error`: 19,74 s -> 1,97 s.

**Broj poziva (nezavisan od masine - najjaci dokaz):** `linalg_lstsq`
96 000 -> 960 po benchmarku; operatorski pozivi po jednom prolazu
64 738 -> 15 056 (-76,7 %); zbirno self-CPU vreme operatora 604 ms -> 166 ms.

**Mikrobencmark regresije:** 14,6x na n = 100, 6,3x na n = 1024.

## Trenutno usko grlo

`get_linspace` - koji je posle prve tri optimizacije dominirao (192 000
poziva `linspace`, 96 000 `cartesian_prod`, 385 456 `.item()` po 3layer
benchmarku) - resen je optimizacijom 4 (izmereni dobitak 1,31-1,35x na
finalnu konfiguraciju). Sledece usko grlo uskih mreza treba utvrditi novim
profilom (kandidati: batched `lstsq` sam po sebi, boundary kernel, zonotopska
propagacija). Na `big` mrezi apsolutno najveci trosak i dalje ostaje gust
matmul `AffineZonotope` (416 s, 33 %), koji je vec jedan BLAS poziv i ne moze
se ubrzati vektorizacijom - to je GPU teren (v. `PLAN_MLSYS.md` u korenu
projekta, van Pasado repoa).

Odbacene su, uz dokumentaciju: **kesiranje** (izmerena stopa ponavljanja
identicnih ulaza je 0 %), **zatvorena forma umesto `lstsq`** (nepotrebna i
losije uslovljena) i **`torch.compile`** (Inductor CPU backend nedostupan jer
nema MSVC `cl.exe`; `linalg_lstsq` prekida graf; kompleksne operacije nisu
podrzane; a ciljni kerneli su posle vektorizacije prejeftini da bi to imalo
efekta).

## Sledeci pravac

~~Grupisanje `get_linspace`~~ - **uraeno** (optimizacija 4), i to
bit-identicno: navodna "~1 ULP razlika u 14 % slucajeva" bila je artefakt
naivne formule; reprodukcija interne symmetric-halves seme `torch.linspace`
je egzaktna, pa odluka o preciznosti nije ni bila potrebna.

Usput postoji i `PASADO_REAL_CUBIC` (Vieteova realna kubna inverzija umesto
complex64 puta): na CPU ne donosi merljivo ubrzanje i NIJE bit-identican
(pomera granice <= 5,7e-5 relativno), pa ostaje **iskljucen** - parkiran kao
enabler za GPU/`torch.compile` fazu (Inductor ne podrzava kompleksne
operacije). Aktuelni pravac je heterogeni CPU-GPU runtime; plan, faze i
go/no-go kriterijumi su u `PLAN_MLSYS.md` (koren projekta).

Srednjorocno, jedini pravac koji moze da promeni asimptotiku jeste
**smanjenje broja sumnih simbola** - precizni transformer dodaje nove simbole
na svakom sigmoid sloju, sto gura i matmul i racunanje granica. To je
algoritamsko pitanje (koliko se tesnoce gubi po uklonjenom simbolu), izvan
opsega ovog sprinta, ali je tu stvarno vreme.

Za GPU su oba otvorena pitanja u meuvremenu izmerena (v. `GPU_READINESS.md`,
Phase 11-13), ne samo pretpostavljena. Drajver: `gelsd` ne postoji na CUDA
(jedini je `gels`), a test na stvarnim rang-deficitnim kutijama (`lx==ux`)
pokazao je da `gels` daje rezultate pogresne za 15-34 reda velicine - potvreno
nebezbedno, ne samo teorijski rizik. fp64 propusnost je hardverski zavisna:
na T4 je ~1/17 fp32 (na nivou lokalnog CPU-a, GPU port ne bi pomogao), na
A100 je fp64 skoro izjednacen sa fp32 (namenske FP64 Tensor Core jedinice,
~65x brze od CPU-a) - na A100/H100 klasi hardvera bi gust `AffineZonotope`
matmul verovatno dobio ozbiljno ubrzanje. GPU pravac vise nije blokiran
otvorenim pitanjima; ostaje pitanje pouzdanog pristupa takvom hardveru i
inzenjering rang-provere za regresioni korak.
