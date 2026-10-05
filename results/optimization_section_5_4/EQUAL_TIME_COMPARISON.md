# Equal-time poreenje: pri jednakom vremenu Pasado dominira u celom merenom opsegu

## Zasto ovo merenje

Equal-budget poreenje (`LIRPA_COMPARISON.md`, presek na K~7) izjednacava
**broj podproblema** - metriku koja favorizuje skupljeg ucesnika. Cena
podproblema je asimetricna ~30x: Pasado ~0.019/0.028/0.041 s po podproblemu
(3/4/5layer, laptop CPU) naspram ~0.57/0.97/1.49 s auto_LiRPA-i na A100.
Pri jednakom **vremenu** Pasado moze da priusti ~30x vise podela; posto obe
metode konvergiraju ka tacnoj vrednosti sa deljenjem, hipoteza je bila da
pri jednakom wall-clock budzetu Pasado dominira. Sada je izmereno: dominira.

## Postavka

- **Pasado strana:** `n_splits  in  {1,3,5,7,9}` (`pasado_splits.csv`)  union
  `{16, 64, 256}` (`pasado_equal_time.csv`, skripta
  `pasado_equal_time_sweep.py`), CPU, istih 30 slika, eps indeksi 0-4,
  float64, kroz `get_lipschitz.py` subprocess (isti mehanizam kao ranije;
  `--eps-indices` flag dodat da veliki k ne mora kroz svih 16 epsilona).
- **auto_LiRPA strana:** `bab_sweep_gpu.csv` (A100 sweep); vreme na budzetu K
  je K x (ukupno vreme sweep-a / 25 solve-ova).
- **Ograde:** (1) Pasado vremena su sa nepinovane masine - okvirna;
  (2) auto_LiRPA cena po solve-u je izvedena iz ukupnog vremena (optimizacija
  staje posle 13-19 iteracija, pa po-solve cena varira); (3) Pasado vrednosti
  reprodukuju se do ~1e-5 relativno zbog neseedovane
  `np.random.normal(0, 1e-4)` perturbacije u samom Pasadu. Sve tri ograde su
  za redove velicine manje od izmerenih razlika.

## Rezultati (30 slika, po slici po epsilonu)

**Tabela A - najmanji Pasado k koji tuce auto_LiRPA-in najveci mereni budzet
(K=9):**

| mreza | eps | auto_LiRPA K=9 | @ vreme | Pasado k* | vrednost | @ vreme | vremenska prednost |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3layer | 0.632 | 185.1 | 5.2 s | 16 | 181.1 | 0.37 s | 14x |
| 3layer | 0.356 | 163.7 | 5.2 s | 16 | 156.5 | 0.27 s | 19x |
| 3layer | 0.200 | 149.7 | 5.2 s | 16 | 144.1 | 0.29 s | 18x |
| 3layer | 0.112 | 145.3 | 5.2 s | 9 | 144.1 | 0.19 s | 27x |
| 3layer | 0.063 | 142.1 | 5.2 s | 7 | 141.7 | 0.13 s | 39x |
| 4layer | 0.632 | 337.9 | 8.7 s | 16 | 316.8 | 0.59 s | 15x |
| 4layer | 0.356 | 255.5 | 8.7 s | 16 | 245.9 | 0.59 s | 15x |
| 4layer | 0.200 | 225.7 | 8.7 s | 16 | 216.4 | 0.60 s | 15x |
| 4layer | 0.112 | 215.3 | 8.7 s | 16 | 205.8 | 0.50 s | 17x |
| 4layer | 0.063 | 208.1 | 8.7 s | 9 | 205.8 | 0.27 s | 32x |
| 5layer | 0.632 | 358.7 | 13.4 s | 16 | 346.6 | 0.69 s | 19x |
| 5layer | 0.356 | 235.9 | 13.4 s | 16 | 207.9 | 0.76 s | 18x |
| 5layer | 0.200 | 183.6 | 13.4 s | 16 | 163.5 | 0.83 s | 16x |
| 5layer | 0.112 | 164.1 | 13.4 s | 9 | 163.4 | 0.27 s | 50x |
| 5layer | 0.063 | 153.9 | 13.4 s | 7 | 152.3 | 0.29 s | 46x |

**Tabela B - head-to-head pri auto_LiRPA-inom K=9 vremenskom budzetu**
(najveci Pasado k cije izmereno vreme staje u taj budzet):

| mreza | eps | auto_LiRPA K=9 | Pasado (k u budzetu) | odnos tesnoce |
|---|---:|---:|---:|---:|
| 3layer | 0.632 | 185.1 | 139.0 (k=256) | **1.33x** |
| 3layer | 0.356 | 163.7 | 138.8 (k=256) | **1.18x** |
| 3layer | 0.200 | 149.7 | 138.7 (k=256) | **1.08x** |
| 3layer | 0.112 | 145.3 | 138.6 (k=256) | **1.05x** |
| 3layer | 0.063 | 142.1 | 138.7 (k=64) | **1.02x** |
| 4layer | 0.632 | 337.9 | 201.3 (k=256) | **1.68x** |
| 4layer | 0.356 | 255.5 | 200.8 (k=256) | **1.27x** |
| 4layer | 0.200 | 225.7 | 200.6 (k=256) | **1.12x** |
| 4layer | 0.112 | 215.3 | 200.5 (k=256) | **1.07x** |
| 4layer | 0.063 | 208.1 | 200.5 (k=256) | **1.04x** |
| 5layer | 0.632 | 358.7 | 142.1 (k=256) | **2.52x** |
| 5layer | 0.356 | 235.9 | 141.6 (k=256) | **1.67x** |
| 5layer | 0.200 | 183.6 | 141.3 (k=256) | **1.30x** |
| 5layer | 0.112 | 164.1 | 141.3 (k=256) | **1.16x** |
| 5layer | 0.063 | 153.9 | 141.0 (k=256) | **1.09x** |

## Nalazi

1. **Pri jednakom vremenu Pasado je tesnji u svih 15 celija** (1.02-2.52x),
   bez ijednog preseka: u wall-clock terminima krive se ne seku nigde u
   merenom opsegu. Presek na K~7 iz equal-budget tabele je artefakt metrike
   broja podproblema, ne stvarne cene.
2. **Dovoljno je k <= 16 (<= 0.9 s po slici na CPU-u) da se pobedi sve sto je
   auto_LiRPA izmerila do 13.4 s po slici na A100** - vremenska prednost pri
   jednakoj tesnoci je 14-50x.
3. **Pasado na k=256 je prakticno konvergirao:** na 5layer vrednosti kroz sve
   epsilone padaju na ~141-142 (sirina podintervala eps/256 je u rezimu gde je
   jednoprolazna analiza gotovo egzaktna). auto_LiRPA na K=9, pri istom
   vremenu, na velikom epsilonu ostaje 2.5x iznad te vrednosti.
4. Redosled jacine tvrdnji za eventualni tekst: (i) jednoprolazno 1.3-2.8x
   tesnji; (ii) pri jednakom broju podproblema prednost nestaje na K~7;
   (iii) pri jednakom vremenu - jedinoj metrici koja meri ono sto korisnika
   kosta - Pasado dominira svuda, i to CPU protiv A100.

## Fajlovi

- `lipschitz-comparison/pasado_equal_time_sweep.py` - sweep (k  in  {16,64,256})
- `lipschitz-comparison/pasado_equal_time.csv` - sirovi rezultati
- `lipschitz-comparison/analyze_equal_time.py` - spajanje i obe tabele
- `lipschitz-comparison/equal_time_summary.csv` - masinski citljiv sazetak
- `Section_5_4/get_lipschitz.py` - dodat `--eps-indices` (podrazumevano
  ponasanje nepromenjeno)

## Otvoreno

- Vremena potvrditi pinovanim runom (`--pin-pcores` protokol iz
  `FORMAL_SUMMARY.md`) pre upotrebe u tekstu; tesnoce su deterministicne
  (do ~1e-5) i ne zavise od toga.
- `big` mreza i eps indeksi 5-6 nisu pokriveni ni ovde (isti razlozi kao u
  `LIRPA_COMPARISON.md`).
- auto_LiRPA iznad K=9 nije merena; ekstrapolacija kaze da bi joj za Pasadovu
  k=256 tesnocu na 5layer/eps=0.632 trebalo visestruko vise podproblema
  (~minuti po slici), ali to nije izmereno pa se ne tvrdi.
