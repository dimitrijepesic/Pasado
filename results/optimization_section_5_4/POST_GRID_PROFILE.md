# Profil posle batch-ovanja grida — gde je usko grlo sada

Merено skriptom `experiments/profile_variants.py` (cProfile, 5layer, 30 slika ×
16 epsilona), poređenjem `vec_both_batched` (stanje posle julskog sprinta) sa
`vec_batched_grid` (posle današnjih izmena).

## Zašto 3layer više nije upotrebljiv za profilisanje

Na 3layer je oko **22 % profila sam start procesa i import** torch-a/sklearn-a
(`_io.open_code` 1.20 s, `BufferedReader.read` 0.74 s, `marshal.loads` 0.40 s,
`_imp.create_dynamic` 0.36 s, `_register_to_dispatcher` 0.49 s,
`nt._getfinalpathname` 0.30 s). Pošto je računanje palo na ~11.6 s wall-clock,
fiksni trošak starta je postao uporediv sa poslom. Sva atribucija ispod je zato
sa 5layer, gde taj udeo pada na ~4 %.

## Efekat batch-ovanog grida

| | pre | posle |
|---|---:|---:|
| `forward_zono_precise` (cum) | 19.73 s | 9.62 s |
| `sigmoid_prime_product_tensor` | 17.83 s | 7.81 s |
| `get_linspace` | 10.17 s | 0.84 s |
| `cartesian_prod` | 8.34 s / 384 000 poziva | uklonjen |
| `linspace` | 12.08 s / 385 926 poziva | 1.45 s / 5 766 |
| `.item()` | 0.79 s / **769 476** poziva | 0.01 s / **1 476** |

**Ne čitati ovo kao 2× ubrzanje.** cProfile naplaćuje fiksni trošak po Python
pozivu, pa precenjuje upravo ono što je uklonjeno (stotine hiljada sitnih
poziva). Wall-clock, naizmenično mereno: **1.245×**. Profil služi za atribuciju,
harness za vreme.

Nusprodukt bitniji od brzine: `.item()` pozivi su pali 521× (a na `big` ih je
bilo 7.87 M). Svaki je device-host sinhronizacija, pa je time otklonjen jedan od
dva glavna GPU blokatora iz `GPU_READINESS.md`. Ostaje `gelsd`.

## Šta je ostalo, po veličini (5layer)

Precizni transformer, `sigmoid_prime_product_tensor` = 7.81 s:

| blok | cum | udeo | priroda |
|---|---:|---:|---|
| `check_nonlinear_boundary_tensor` | 2.98 s | 38 % | naš vektorizovani kod |
| `lin_reg_tensor_batched` | 1.88 s | 24 % | od toga LAPACK `lstsq` 1.76 s |
| `get_linspace_batched` | 0.84 s | 11 % | već batch-ovan |
| `check_corners_tensor` | 0.41 s | 5 % | naš vektorizovani kod |

Van preciznog transformera, gusta zonotopska mašinerija (~5-6 s):
`imatmul` 1.90 s, `get_coeff_abs` 1.38 s (`abs` 0.88 + `sum` 0.69),
`AffineZonotope` 1.08 s, `torch.cat` 0.92 s, `traceify` 0.85 s.

## Zaključak: laki dobici na CPU-u su iscrpljeni

Profil više nije Python-orkestracija. Dominiraju dve stvari:

1. **`check_nonlinear_boundary_tensor`, 38 % preciznog puta** — ali to je **naš
   sopstveni, već vektorizovan kod**. Dalji dobitak ne dolazi od još
   vektorizacije nego od fuzije ili algoritamske zamene. Sadrži
   `inverse_poly_tensor` (0.90 s), koji invertuje kubnu jednačinu kroz
   `complex64`.
2. **Stvarna gusta linearna algebra** — matmul propagacija i `abs`/`sum`
   redukcije za granice. Ne može se vektorizacijom ubrzati, već je jedan BLAS
   poziv.

## Nov nalaz: `Zonotope.expand` raste kvadratno

`torch.cat` (0.92 s) dolazi većinom iz `Zonotope.expand` — **12 480 poziva,
0.45 s**. Posle svakog sigmoid sloja matrica generatora se dopunjuje nulama
(`torch.cat([generators, torch.zeros((n, cols))])`) da se broj noise simbola
poravna između realnog i dualnog zonotopa. Svaki poziv **kopira celu matricu**,
pa je ukupan trošak kvadratan u broju simbola.

Na 5layer je to 1.7 % i ne vredi dirati zbog brzine. Na `big` (1024-široka, više
simbola po sloju) verovatno je znatno više — treba proveriti pre nego što se
odbaci. Nezavisno od brzine, `torch.zeros((n, cols))` bez `device=`/`dtype=` je
i jedan od DEVFIX sajtova iz `GPU_READINESS.md`, pa će se ta linija menjati
ionako.

## Šta ovo znači za sledeći korak

Tri opcije, i profil ih rangira:

1. **Fuzija `check_nonlinear_boundary_tensor`** — najveći adresabilni blok.
   Veza sa originalnim planom prakse (kompajlerske tehnike). Prepreka:
   `complex64` u `inverse_poly_tensor` je jedan od razloga zašto je
   `torch.compile` odbačen (Inductor ne generiše kod za kompleksne tipove). Ako
   se koreni izračunaju bez kompleksne aritmetike, otvara se i ta opcija —
   dvostruki dobitak.
2. **GPU za guste delove** — matmul i redukcije. Blokirano `gelsd`-om samo za
   regresiju; ostalo je slobodno sad kad `.item()` sinhronizacija nema.
3. **Manje noise simbola** — algoritamski, smanjuje i matmul i `expand` i
   granice odjednom. Menja rezultate analize, pa je van pravila "ista
   semantika" koje je sprint do sad držao.
