# Realni kubni solver umesto `complex64` (`PASADO_REAL_CUBIC`)

## Sta je izmenjeno i zasto

`inverse_sigmoid_2nd_deriv` resava `sigma''(x) = y`. Sa `s = sigma(x)` to je
`2s^3 - 3s^2 + s = y`, a uz `s = u + 1/2` depresovana kubna **u^3 - 1/4u - 1/2y = 0**.
Original je resava Cardanovom formulom evaluiranom u **`complex64`**.

Za tu kubnu je `p = -1/4`, `q = -y/2`, pa su sva tri korena realna kad
`4p^3 + 27q^2 < 0`, sto se svodi na `|y| < 1/(6sqrt3) ~ 0.09623` - a to je **tacno
maksimum `|sigma''|`**. Dakle u celom fizicki dostiznom opsegu kompleksni put racuna
tri realna broja kroz kompleksnu aritmetiku. Viete-ova trigonometrijska forma
daje ih direktno:

    u_k = (1/sqrt3) * cos( arccos(6sqrt3*y)/3 - 2pik/3 ),  k = 0,1,2

i argument `arccos`-a lezi u [-1, 1] **tacno kad** su koreni realni, pa provera
opsega dolazi besplatno.

## Izmereno

| | `complex64` | realni Viete |
|---|---:|---:|
| rezidual `|2s^3 - 3s^2 + s - y|`, medijana | 3.8e-06 | **1.2e-16** |
| isti, maksimum | 1.1e-05 | 8.7e-16 |
| najveca relativna razlika u finalnoj Lipsicovoj granici | - | 5.7e-05 |

### Brzina: brze na nivou kernela, neutralno end-to-end

Izolovan mikrobencmark kernela `check_nonlinear_boundary_tensor` (warmup + 30
ponavljanja, medijana), pri dve gustine zivih kandidata jer `filter_range`
odbaci vecinu pre kubnog solvera:

| n | zivih kandidata | `complex64` | realni Viete | odnos |
|---:|---:|---:|---:|---:|
| 100 | 2 % | 0.778 ms | 0.344 ms | **2.26x** |
| 100 | 100 % | 0.664 ms | 0.414 ms | **1.60x** |
| 1024 | 3 % | 1.475 ms | 1.254 ms | **1.18x** |
| 1024 | 100 % | 1.581 ms | 0.776 ms | **2.04x** |

End-to-end (5layer, harness, naizmenicno, 3 ponavljanja): odnosi po parovima
**1.09x / 0.96x / 0.91x**, medijana 0.959x - dakle **unutar suma merenja**.
Kernel je 2.98 s od 25.9 s profajlerskog vremena, pa bi 2x na njemu dao ~8 %
end-to-end; sa varijansom od +/-9 % po parovima efekat te velicine se ovim
merenjem ne moze razluciti.

**Ispravka ranijeg nalaza u ovom projektu.** Prvo merenje je izvestavalo da
realni kubni kosta **8 %**. To je bio artefakt poreenja **dva pojedinacna
runa** bez naizmenicnog merenja i bez ponavljanja, na masini cija varijansa ide
10-20 % - tacno greska zbog koje harness i postoji. Kad se izmeri harnessom,
cene nema.

## Zasto se uvodi

1. **Numericka higijena.** Stari koreni su efektivno **single precision**, jer
   `complex64` to jeste, unutar analize koja je inace float64 i izvestava
   granice na 14 cifara. Gore od samog reziduala: `inv_sigmoid(s) = log(s/(1-s))`
   eksplodira kad `s -> 0` ili `1`, pa greska od 1e-6 u `s` postane **~1** u
   vracenoj lokaciji kandidata - a te lokacije se koriste za evaluaciju
   maksimuma greske.

2. **Otkljucava TorchInductor.** `complex64` je bio jedan od tri razloga zasto je
   `torch.compile` odbacen (`TORCH_COMPILE_ANALYSIS.md`): Inductor ne generise kod
   za kompleksne tipove. Preostaju MSVC (lokalno okruzenje) i `linalg_lstsq`
   graph-break, a taj je **van** ovog kernela - dakle
   `check_nonlinear_boundary_tensor`, najveci adresabilni blok u profilu
   (38 % preciznog puta), postaje kandidat za fuziju.

3. **Smer promene granica je bezbedniji.** U 10 od 16 epsilona nova granica je
   *malo veca*, tj. konzervativnija - konzistentno sa tim da su netacne lokacije
   kandidata dovodile do blagog **potcenjivanja** maksimuma greske. Efekat
   (<= 5.7e-05) je premali da bi bio praktican problem i ne dokazuje da je
   original nesound, ali je to smer koji se ocekuje ako stara verzija propusta
   pravi maksimum.

## Rukovanje granicom: tolerancija 1e-7

Dve formule daju korene u **razlicitom redosledu**, pa se moraju porediti kao
skupovi; poreenje po indeksu daje lazno alarmantne razlike (prvo merenje u ovom
radu je upravo tako pogresilo).

Poreeno kao skupovi, broj realnih korena se poklapa za **sve** ulaze u opsegu,
van opsega i realisticne. Razilaze se samo unutar ~1e-6 od granice, gde je i sam
float32 diskriminant produkcije nepouzdan. Bez tolerancije novi put tamo u ~2.4 %
slucajeva **odbaci** koren koji produkcija nae - a to je smer koji nosi rizik
nesoundness-a, jer izgubljeni kandidat moze da spusti maksimum greske.

Resenje: NaN samo kad `|arg| > 1 + tol`; unutar pojasa argument se klampuje, sto
vraca (skoro) dvostruki koren umesto da ga odbaci. **Zadrzan kandidat moze samo
da poveca clan greske (konzervativnije); odbacen je taj koji bi mogao da narusi
soundness.** Izmereno po tolerancijama (broj odbacenih / dodatih na
adversarijalnom uzorku oko granice):

| tol | odbaceno | dodato |
|---|---:|---:|
| 0 | **491** | 325 |
| 1e-9 | **476** | 337 |
| **1e-7** | **0** | 1867 |
| 1e-6 | 0 | 9843 |

Izabrano **1e-7** - najmanja vrednost koja ne odbacuje nista, i nije proizvoljna:
to je epsilon float32-a, tj. tacno preciznost na kojoj produkcija donosi istu
odluku o diskriminantu.

## Odluka: selektor podrazumevano ISKLJUCEN

Za razliku od ostalih selektora, ovaj **nije bit-identican** - namerno menja
numeriku. Posto cene u brzini nema (v. gore), jedini razlog da ostane iskljucen
je reproducibilnost: **menja izvestavane granice** (za <= 5.7e-05), a artifakt
sluzi da reprodukuje brojeve iz objavljenog rada. Ukljucivanje po podrazumevanoj
vrednosti bi znacilo da `lipschitz.sh` vise ne daje iste brojeve kao rad.

Ukljucuje se sa `PASADO_REAL_CUBIC=1`. Preporuka: koristiti ga kao osnovu za
`torch.compile` eksperiment (jedini put kojim `check_nonlinear_boundary_tensor`
moze da se kompajlira), a za promenu podrazumevane vrednosti traziti odluku
autora artifakta - to je pitanje reproducibilnosti, ne performansi.

## Neuraeno

`inverse_quadratic_tensor` (softplus put) takoe koristi `complex64`. Nije na
sigmoid putu koji Section 5.4 benchmark koristi, pa nije menjan - ali ce morati
ako se ikad bude kompajlirao i softplus transformer.
