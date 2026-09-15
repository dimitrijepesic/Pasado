# Realni kubni solver umesto `complex64` (`PASADO_REAL_CUBIC`)

## Šta je izmenjeno i zašto

`inverse_sigmoid_2nd_deriv` rešava `σ''(x) = y`. Sa `s = σ(x)` to je
`2s³ − 3s² + s = y`, a uz `s = u + ½` depresovana kubna **u³ − ¼u − ½y = 0**.
Original je rešava Cardanovom formulom evaluiranom u **`complex64`**.

Za tu kubnu je `p = −¼`, `q = −y/2`, pa su sva tri korena realna kad
`4p³ + 27q² < 0`, što se svodi na `|y| < 1/(6√3) ≈ 0.09623` — a to je **tačno
maksimum `|σ''|`**. Dakle u celom fizički dostižnom opsegu kompleksni put računa
tri realna broja kroz kompleksnu aritmetiku. Viète-ova trigonometrijska forma
daje ih direktno:

    u_k = (1/√3) · cos( arccos(6√3·y)/3 − 2πk/3 ),  k = 0,1,2

i argument `arccos`-a leži u [−1, 1] **tačno kad** su koreni realni, pa provera
opsega dolazi besplatno.

## Izmereno

| | `complex64` | realni Viète |
|---|---:|---:|
| rezidual `|2s³ − 3s² + s − y|`, medijana | 3.8e-06 | **1.2e-16** |
| isti, maksimum | 1.1e-05 | 8.7e-16 |
| najveća relativna razlika u finalnoj Lipšicovoj granici | — | 5.7e-05 |

### Brzina: brže na nivou kernela, neutralno end-to-end

Izolovan mikrobenčmark kernela `check_nonlinear_boundary_tensor` (warmup + 30
ponavljanja, medijana), pri dve gustine živih kandidata jer `filter_range`
odbaci većinu pre kubnog solvera:

| n | živih kandidata | `complex64` | realni Viète | odnos |
|---:|---:|---:|---:|---:|
| 100 | 2 % | 0.778 ms | 0.344 ms | **2.26×** |
| 100 | 100 % | 0.664 ms | 0.414 ms | **1.60×** |
| 1024 | 3 % | 1.475 ms | 1.254 ms | **1.18×** |
| 1024 | 100 % | 1.581 ms | 0.776 ms | **2.04×** |

End-to-end (5layer, harness, naizmenično, 3 ponavljanja): odnosi po parovima
**1.09× / 0.96× / 0.91×**, medijana 0.959× — dakle **unutar šuma merenja**.
Kernel je 2.98 s od 25.9 s profajlerskog vremena, pa bi 2× na njemu dao ~8 %
end-to-end; sa varijansom od ±9 % po parovima efekat te veličine se ovim
merenjem ne može razlučiti.

**Ispravka ranijeg nalaza u ovom projektu.** Prvo merenje je izveštavalo da
realni kubni košta **8 %**. To je bio artefakt poređenja **dva pojedinačna
runa** bez naizmeničnog merenja i bez ponavljanja, na mašini čija varijansa ide
10-20 % — tačno greška zbog koje harness i postoji. Kad se izmeri harnessom,
cene nema.

## Zašto se uvodi

1. **Numerička higijena.** Stari koreni su efektivno **single precision**, jer
   `complex64` to jeste, unutar analize koja je inače float64 i izveštava
   granice na 14 cifara. Gore od samog reziduala: `inv_sigmoid(s) = log(s/(1−s))`
   eksplodira kad `s → 0` ili `1`, pa greška od 1e-6 u `s` postane **~1** u
   vraćenoj lokaciji kandidata — a te lokacije se koriste za evaluaciju
   maksimuma greške.

2. **Otključava TorchInductor.** `complex64` je bio jedan od tri razloga zašto je
   `torch.compile` odbačen (`TORCH_COMPILE_ANALYSIS.md`): Inductor ne generiše kod
   za kompleksne tipove. Preostaju MSVC (lokalno okruženje) i `linalg_lstsq`
   graph-break, a taj je **van** ovog kernela — dakle
   `check_nonlinear_boundary_tensor`, najveći adresabilni blok u profilu
   (38 % preciznog puta), postaje kandidat za fuziju.

3. **Smer promene granica je bezbedniji.** U 10 od 16 epsilona nova granica je
   *malo veća*, tj. konzervativnija — konzistentno sa tim da su netačne lokacije
   kandidata dovodile do blagog **potcenjivanja** maksimuma greške. Efekat
   (≤ 5.7e-05) je premali da bi bio praktičan problem i ne dokazuje da je
   original nesound, ali je to smer koji se očekuje ako stara verzija propušta
   pravi maksimum.

## Rukovanje granicom: tolerancija 1e-7

Dve formule daju korene u **različitom redosledu**, pa se moraju porediti kao
skupovi; poređenje po indeksu daje lažno alarmantne razlike (prvo merenje u ovom
radu je upravo tako pogrešilo).

Poređeno kao skupovi, broj realnih korena se poklapa za **sve** ulaze u opsegu,
van opsega i realistične. Razilaze se samo unutar ~1e-6 od granice, gde je i sam
float32 diskriminant produkcije nepouzdan. Bez tolerancije novi put tamo u ~2.4 %
slučajeva **odbaci** koren koji produkcija nađe — a to je smer koji nosi rizik
nesoundness-a, jer izgubljeni kandidat može da spusti maksimum greške.

Rešenje: NaN samo kad `|arg| > 1 + tol`; unutar pojasa argument se klampuje, što
vraća (skoro) dvostruki koren umesto da ga odbaci. **Zadržan kandidat može samo
da poveća član greške (konzervativnije); odbačen je taj koji bi mogao da naruši
soundness.** Izmereno po tolerancijama (broj odbačenih / dodatih na
adversarijalnom uzorku oko granice):

| tol | odbačeno | dodato |
|---|---:|---:|
| 0 | **491** | 325 |
| 1e-9 | **476** | 337 |
| **1e-7** | **0** | 1867 |
| 1e-6 | 0 | 9843 |

Izabrano **1e-7** — najmanja vrednost koja ne odbacuje ništa, i nije proizvoljna:
to je epsilon float32-a, tj. tačno preciznost na kojoj produkcija donosi istu
odluku o diskriminantu.

## Odluka: selektor podrazumevano ISKLJUČEN

Za razliku od ostalih selektora, ovaj **nije bit-identičan** — namerno menja
numeriku. Pošto cene u brzini nema (v. gore), jedini razlog da ostane isključen
je reproducibilnost: **menja izveštavane granice** (za ≤ 5.7e-05), a artifakt
služi da reprodukuje brojeve iz objavljenog rada. Uključivanje po podrazumevanoj
vrednosti bi značilo da `lipschitz.sh` više ne daje iste brojeve kao rad.

Uključuje se sa `PASADO_REAL_CUBIC=1`. Preporuka: koristiti ga kao osnovu za
`torch.compile` eksperiment (jedini put kojim `check_nonlinear_boundary_tensor`
može da se kompajlira), a za promenu podrazumevane vrednosti tražiti odluku
autora artifakta — to je pitanje reproducibilnosti, ne performansi.

## Neurađeno

`inverse_quadratic_tensor` (softplus put) takođe koristi `complex64`. Nije na
sigmoid putu koji Section 5.4 benchmark koristi, pa nije menjan — ali će morati
ako se ikad bude kompajlirao i softplus transformer.
